from django.db.models import Count, Avg
from oauth2_provider.contrib.rest_framework import TokenHasScope
from rest_framework import viewsets, status, permissions, generics
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.pagination import PageNumberPagination
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from django.utils import timezone
from django.db import transaction, models
from django.http import HttpResponse
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes
import os
import logging
from typing import Dict, Any

from .models import (
    Document, DocumentContent, Topic, Section,
    Paragraph, Table, DocumentCategory, Theme, User
)
from .serializers import (
    DocumentListSerializer, DocumentDetailSerializer,
    DocumentContentSerializer, TopicDetailSerializer,
    SectionSerializer, ParagraphSerializer,
    TableSerializer
)
from .services.document_processor import document_processor
from .storage import DocumentStorage
from .permissions import CanViewDocument, CanManageDocument
from .tasks import process_document_content

logger = logging.getLogger(__name__)

class DocumentProcessingPagination(PageNumberPagination):
    """Pagination spécialisée pour le traitement de documents"""
    page_size = 15
    page_size_query_param = 'page_size'
    max_page_size = 50

class DocumentProcessingViewSet(viewsets.ModelViewSet):
    """ViewSet spécialisé pour le traitement et l'extraction de documents"""

    queryset = Document.objects.select_related(
        'theme', 'category', 'uploaded_by', 'content'
    ).prefetch_related('topics', 'topics__sections')

    serializer_class = DocumentDetailSerializer
    pagination_class = DocumentProcessingPagination
    parser_classes = [MultiPartParser, FormParser]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ['title', 'description', 'original_filename']
    ordering_fields = ['created_at', 'title', 'status', 'file_size']
    ordering = ['-created_at']
    permission_classes = [permissions.IsAuthenticated, TokenHasScope]
    required_scopes = ['read', 'write']

    #permission_classes = [permissions.IsAutheticated]


    def get_serializer_class(self):
        """Sélectionne le bon serializer selon l'action"""
        if self.action == 'list':
            return DocumentListSerializer
        elif self.action in ['upload_and_process', 'create']:
            return DocumentListSerializer
        return DocumentDetailSerializer

    def get_permissions(self):
        """Permissions dynamiques selon l'action"""
        if self.action in ['retrieve', 'list']:
            self.permission_classes = [permissions.IsAuthenticated, CanViewDocument]
        elif self.action in ['create', 'upload_and_process']:
            self.permission_classes = [permissions.IsAuthenticated]
        else:
            self.permission_classes = [permissions.IsAuthenticated, CanManageDocument]

        return super().get_permissions()

    @extend_schema(
        summary="Upload et traitement automatique d'un document",
        description="Upload un document (PDF/Word) et lance automatiquement l'extraction du contenu",
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'file': {'type': 'string', 'format': 'binary'},
                    'title': {'type': 'string'},
                    'description': {'type': 'string'},
                    'theme_id': {'type': 'string', 'format': 'uuid'},
                    'category_id': {'type': 'string', 'format': 'uuid'},
                    'visibility': {'type': 'string', 'enum': ['public', 'private', 'restricted']},
                    'language': {'type': 'string', 'enum': ['fr', 'en', 'es']},
                    'auto_process': {'type': 'boolean', 'default': True}
                },
                'required': ['file', 'title', 'theme_id', 'category_id']
            }
        },
        responses={201: OpenApiTypes.OBJECT}
    )
    @action(detail=False, methods=['post'])
    def upload_and_process(self, request):
        """Upload un document et lance automatiquement l'extraction"""
        try:
            with transaction.atomic():
                # Validation des données
                uploaded_file = request.FILES.get('file')
                if not uploaded_file:
                    return Response(
                        {'error': 'Aucun fichier fourni'},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Validation du type de fichier
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                allowed_extensions = ['.pdf', '.doc', '.docx']

                if file_extension not in allowed_extensions:
                    return Response(
                        {
                            'error': f'Type de fichier non supporté. Extensions autorisées: {", ".join(allowed_extensions)}'},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Récupération des données
                title = request.data.get('title')
                description = request.data.get('description', '')
                theme_id = request.data.get('theme_id')
                category_id = request.data.get('category_id')
                visibility = request.data.get('visibility', 'public')
                language = request.data.get('language', 'fr')
                auto_process = request.data.get('auto_process', 'true').lower() == 'true'

                # Validation des champs requis
                if not all([title, theme_id, category_id]):
                    return Response(
                        {'error': 'Titre, thème et catégorie sont requis'},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Validation de l'existence des objets liés
                try:
                    theme = Theme.objects.get(id=theme_id, is_deleted=False, is_active=True)
                    category = DocumentCategory.objects.get(id=category_id)
                except (Theme.DoesNotExist, DocumentCategory.DoesNotExist):
                    return Response(
                        {'error': 'Thème ou catégorie invalide'},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Vérification des quotas utilisateur
                user = request.user
                file_size_mb = uploaded_file.size / (1024 * 1024)

                if user.is_quota_exceeded(file_size_mb):
                    return Response(
                        {'error': f'Quota de stockage dépassé. Limite: {user.storage_quota_mb}MB'},
                        status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                    )

                if not user.can_upload_documents():
                    return Response(
                        {'error': f'Limite de documents atteinte. Maximum: {user.max_documents_upload}'},
                        status=status.HTTP_403_FORBIDDEN
                    )

                # Sauvegarde du fichier avec vérification de duplicata
                try:
                    saved_path, is_duplicate = DocumentStorage.save_document_file(
                        uploaded_file, uploaded_file.name, user
                    )

                    if is_duplicate:
                        return Response(
                            {
                                'error': 'Ce fichier existe déjà dans le système',
                                'duplicate_document_id': is_duplicate.id
                            },
                            status=status.HTTP_409_CONFLICT
                        )

                except ValueError as e:
                    return Response(
                        {'error': str(e)},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Création du document
                document = Document.objects.create(
                    title=title,
                    description=description,
                    original_filename=uploaded_file.name,
                    file_path=saved_path,
                    file_type=file_extension[1:],  # Sans le point
                    file_size=uploaded_file.size,
                    theme=theme,
                    category=category,
                    uploaded_by=user,
                    visibility=visibility,
                    language=language,
                    status=Document.StatusChoices.PENDING
                )

                # Calcul du checksum
                document.file_checksum = DocumentStorage.calculate_file_hash(uploaded_file)
                document.save(update_fields=['file_checksum'])

                if auto_process:
                    try:
                        if hasattr(process_document_content, 'delay'):
                            process_document_content.delay(str(document.id))
                        else:
                            # Traitement synchrone en fallback
                            success = document_processor.process_document(document)
                            if not success:
                                logger.warning(f"Échec du traitement automatique pour le document {document.id}")
                    except Exception as e:
                        logger.error(f"Erreur lors du lancement du traitement: {str(e)}")

                # Réponse avec le document créé
                serializer = DocumentDetailSerializer(document)
                return Response(
                    {
                        'message': 'Document uploadé avec succès',
                        'document': serializer.data,
                        'processing_started': auto_process
                    },
                    status=status.HTTP_201_CREATED
                )

        except Exception as e:
            logger.error(f"Erreur lors de l'upload: {str(e)}")
            return Response(
                {'error': 'Erreur interne du serveur'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        summary="Lance le traitement d'extraction pour un document",
        description="Démarre l'extraction du contenu pour un document spécifique",
        responses={200: DocumentDetailSerializer}
    )
    @action(detail=True, methods=['post'])
    def process_content(self, request, pk=None):
        """Lance le traitement d'extraction du contenu"""
        document = self.get_object()

        # Vérifications
        if document.status == Document.StatusChoices.PROCESSING:
            return Response(
                {'error': 'Le document est déjà en cours de traitement'},
                status=status.HTTP_409_CONFLICT
            )

        if document.status == Document.StatusChoices.PROCESSED:
            force_reprocess = request.data.get('force', False)
            if not force_reprocess:
                return Response(
                    {'error': 'Le document a déjà été traité. Utilisez force=true pour retraiter'},
                    status=status.HTTP_409_CONFLICT
                )

        try:
            # Lancement du traitement
            if hasattr(process_document_content, 'delay'):
                task = process_document_content.delay(str(document.id))
                return Response({
                    'message': 'Traitement lancé',
                    'task_id': task.id,
                    'document_id': document.id
                })
            else:
                # Traitement synchrone
                success = document_processor.process_document(document)
                if success:
                    serializer = DocumentDetailSerializer(document)
                    return Response({
                        'message': 'Traitement terminé avec succès',
                        'document': serializer.data
                    })
                else:
                    return Response(
                        {'error': 'Échec du traitement'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

        except Exception as e:
            logger.error(f"Erreur lors du lancement du traitement: {str(e)}")
            return Response(
                {'error': 'Erreur lors du lancement du traitement'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        summary="Statut du traitement d'un document",
        description="Récupère le statut et les détails du traitement d'un document",
        responses={
            200: OpenApiTypes.OBJECT,
            'example': {
                'document_id': 'uuid',
                'status': 'string',
                'created_at': 'datetime',
                'updated_at': 'datetime',
                'processing_log': 'array',
                'has_content': 'boolean',
                'extraction_metadata': 'object',
                'content_stats': {
                    'word_count': 'integer',
                    'page_count': 'integer',
                    'topics_count': 'integer',
                    'sections_count': 'integer',
                    'paragraphs_count': 'integer',
                    'tables_count': 'integer',
                    'extraction_confidence': 'float',
                    'processing_duration': 'string'
                }
            }
        }
    )
    @action(detail=True, methods=['get'])
    def processing_status(self, request, pk=None):
        """Récupère le statut de traitement du document"""
        document = self.get_object()

        # Construction de la réponse avec détails
        response_data = {
            'document_id': str(document.id),
            'status': document.status,
            'created_at': document.created_at,
            'updated_at': document.updated_at,
            'processing_log': document.processing_log,
            'has_content': hasattr(document, 'content'),
            'extraction_metadata': document.extraction_metadata
        }

        # Ajout des statistiques si le contenu existe
        if hasattr(document, 'content'):
            content = document.content
            response_data.update({
                'content_stats': {
                    'word_count': content.word_count,
                    'page_count': content.page_count,
                    'topics_count': document.topics.count(),
                    'sections_count': Section.objects.filter(topic__document=document).count(),
                    'paragraphs_count': Paragraph.objects.filter(section__topic__document=document).count(),
                    'tables_count': Table.objects.filter(section__topic__document=document).count(),
                    'extraction_confidence': float(content.extraction_confidence),
                    'processing_duration': str(content.processing_duration) if content.processing_duration else None
                }
            })

        return Response(response_data)

    @extend_schema(
        summary="Contenu extrait d'un document",
        description="Récupère le contenu structuré extrait d'un document",
        responses={200: DocumentContentSerializer}
    )
    @action(detail=True, methods=['get'])
    def extracted_content(self, request, pk=None):
        """Récupère le contenu extrait du document"""
        document = self.get_object()

        if not hasattr(document, 'content'):
            return Response(
                {'error': 'Aucun contenu extrait disponible'},
                status=status.HTTP_404_NOT_FOUND
            )

        content = document.content
        serializer = DocumentContentSerializer(content)
        return Response(serializer.data)

    @extend_schema(
        summary="Structure hiérarchique du document",
        description="Récupère la structure complète (topics, sections, paragraphes) du document",
        responses={
            200: OpenApiTypes.OBJECT,
            'example': {
                'document_id': 'uuid',
                'document_title': 'string',
                'structure': [{
                    'id': 'uuid',
                    'title': 'string',
                    'sections': [{
                        'id': 'uuid',
                        'title': 'string',
                        'paragraphs': [{
                            'id': 'uuid',
                            'content': 'string'
                        }]
                    }]
                }],
                'stats': {
                    'topics_count': 'integer',
                    'sections_count': 'integer',
                    'paragraphs_count': 'integer'
                }
            }
        }
    )
    @action(detail=True, methods=['get'])
    def structure(self, request, pk=None):
        """Récupère la structure hiérarchique du document"""
        document = self.get_object()

        topics = document.topics.prefetch_related(
            'sections__paragraphs'
        ).order_by('order_index')

        structure_data = []

        for topic in topics:
            topic_data = TopicDetailSerializer(topic).data
            topic_data['sections'] = []

            sections = topic.sections.filter(is_deleted=False).order_by('order_index')
            for section in sections:
                section_data = SectionSerializer(section).data
                section_data['paragraphs'] = []

                paragraphs = section.paragraphs.all().order_by('order_index')
                for paragraph in paragraphs:
                    paragraph_data = ParagraphSerializer(paragraph).data
                    section_data['paragraphs'].append(paragraph_data)

                topic_data['sections'].append(section_data)

            structure_data.append(topic_data)

        return Response({
            'document_id': str(document.id),
            'document_title': document.title,
            'structure': structure_data,
            'stats': {
                'topics_count': len(structure_data),
                'sections_count': sum(len(topic['sections']) for topic in structure_data),
                'paragraphs_count': sum(
                    len(section['paragraphs'])
                    for topic in structure_data
                    for section in topic['sections']
                )
            }
        })

    @extend_schema(
        summary="Tableaux extraits du document",
        description="Récupère tous les tableaux extraits du document",
        responses={200: TableSerializer(many=True)}
    )
    @action(detail=True, methods=['get'])
    def tables(self, request, pk=None):
        """Récupère les tableaux extraits du document"""
        document = self.get_object()

        tables = Table.objects.filter(
            section__topic__document=document
        ).order_by('order_index')

        serializer = TableSerializer(tables, many=True)
        return Response({
            'document_id': str(document.id),
            'tables': serializer.data,
            'count': tables.count()
        })

    @extend_schema(
        summary="Exporter le contenu en différents formats",
        description="Exporte le contenu extrait en JSON, Markdown ou texte brut",
        parameters=[
            OpenApiParameter(
                name='format',
                type=OpenApiTypes.STR,
                location=OpenApiParameter.QUERY,
                enum=['json', 'markdown', 'txt'],
                default='json'
            )
        ],
        responses={
            200: OpenApiTypes.BINARY,
            400: OpenApiTypes.OBJECT
        }
    )
    @action(detail=True, methods=['get'])
    def export(self, request, pk=None):
        """Exporte le contenu en différents formats"""
        document = self.get_object()
        export_format = request.query_params.get('format', 'json').lower()

        if not hasattr(document, 'content'):
            return Response(
                {'error': 'Aucun contenu à exporter'},
                status=status.HTTP_404_NOT_FOUND
            )

        content = document.content

        if export_format == 'json':
            return self._export_json(document, content)
        elif export_format == 'markdown':
            return self._export_markdown(document, content)
        elif export_format == 'txt':
            return self._export_text(document, content)
        else:
            return Response(
                {'error': 'Format non supporté. Formats disponibles: json, markdown, txt'},
                status=status.HTTP_400_BAD_REQUEST
            )

    def _export_json(self, document: Document, content: DocumentContent) -> Response:
        """Export au format JSON"""
        export_data = {
            'document': DocumentDetailSerializer(document).data,
            'content': DocumentContentSerializer(content).data,
            'structure': self._get_structure_data(document),
            'export_metadata': {
                'exported_at': timezone.now().isoformat(),
                'format': 'json',
                'version': '1.0'
            }
        }

        response = Response(export_data)
        response['Content-Disposition'] = f'attachment; filename="{document.slug}.json"'
        return response

    def _export_markdown(self, document: Document, content: DocumentContent) -> HttpResponse:
        """Export au format Markdown"""
        markdown_content = f"# {document.title}\n\n"

        if document.description:
            markdown_content += f"{document.description}\n\n"

        markdown_content += "---\n\n"

        # Ajout de la structure
        topics = document.topics.prefetch_related('sections__paragraphs').order_by('order_index')

        for topic in topics:
            markdown_content += f"## {topic.title}\n\n"

            for section in topic.sections.filter(is_deleted=False).order_by('order_index'):
                markdown_content += f"### {section.title}\n\n"

                for paragraph in section.paragraphs.all().order_by('order_index'):
                    markdown_content += f"{paragraph.content}\n\n"

        response = HttpResponse(markdown_content, content_type='text/markdown')
        response['Content-Disposition'] = f'attachment; filename="{document.slug}.md"'
        return response

    def _export_text(self, document: Document, content: DocumentContent) -> HttpResponse:
        """Export au format texte brut"""
        text_content = f"{document.title}\n"
        text_content += "=" * len(document.title) + "\n\n"

        if document.description:
            text_content += f"{document.description}\n\n"

        text_content += content.clean_content

        response = HttpResponse(text_content, content_type='text/plain')
        response['Content-Disposition'] = f'attachment; filename="{document.slug}.txt"'
        return response

    def _get_structure_data(self, document: Document) -> Dict[str, Any]:
        """Récupère les données de structure pour l'export"""
        topics = document.topics.prefetch_related('sections__paragraphs').order_by('order_index')

        structure = []
        for topic in topics:
            topic_data = {
                'id': str(topic.id),
                'title': topic.title,
                'content': topic.content,
                'topic_type': topic.topic_type,
                'level': topic.level,
                'sections': []
            }

            for section in topic.sections.filter(is_deleted=False).order_by('order_index'):
                section_data = {
                    'id': str(section.id),
                    'title': section.title,
                    'content': section.content,
                    'section_type': section.section_type,
                    'paragraphs': []
                }

                for paragraph in section.paragraphs.all().order_by('order_index'):
                    paragraph_data = {
                        'id': str(paragraph.id),
                        'content': paragraph.content,
                        'paragraph_type': paragraph.paragraph_type,
                        'word_count': paragraph.word_count
                    }
                    section_data['paragraphs'].append(paragraph_data)

                topic_data['sections'].append(section_data)

            structure.append(topic_data)

        return structure

class ProcessingStatisticsView(generics.GenericAPIView):
    """APIView pour les statistiques de traitement des documents"""
    serializer_class = DocumentContentSerializer
    permission_classes = [permissions.IsAuthenticated]

    @extend_schema(
        summary="Statistiques de traitement des documents",
        description="Récupère les statistiques globales de traitement des documents",
        responses={
            200: OpenApiTypes.OBJECT,
            'example': {
                'summary': {
                    'total_documents': 'integer',
                    'processed_documents': 'integer',
                    'processing_documents': 'integer',
                    'error_documents': 'integer'
                },
                'status_distribution': [{'status': 'string', 'count': 'integer'}],
                'file_type_distribution': [{'file_type': 'string', 'count': 'integer'}],
                'content_statistics': {
                    'avg_word_count': 'float',
                    'avg_page_count': 'float',
                    'avg_extraction_confidence': 'float',
                    'total_processed': 'integer'
                },
                'extraction_methods': [{'extraction_method': 'string', 'count': 'integer'}],
                'recent_processed': []  # example list of documents
            }
        }
    )
    def get(self, request):
        """Statistiques globales de traitement"""
        # Statistiques par statut
        status_stats = Document.objects.values('status').annotate(
            count=Count('id')
        ).order_by('status')

        # Statistiques par type de fichier
        file_type_stats = Document.objects.values('file_type').annotate(
            count=Count('id')
        ).order_by('file_type')

        # Statistiques de contenu
        content_stats = DocumentContent.objects.aggregate(
            avg_word_count=Avg('word_count'),
            avg_page_count=Avg('page_count'),
            avg_extraction_confidence=Avg('extraction_confidence'),
            total_processed=Count('id')
        )

        extraction_stats = DocumentContent.objects.values('extraction_method').annotate(
            count=Count('id')
        ).order_by('extraction_method')

        # Documents récemment traités
        recent_processed = Document.objects.filter(
            status=Document.StatusChoices.PROCESSED
        ).order_by('-updated_at')[:5]

        return Response({
            'summary': {
                'total_documents': Document.objects.count(),
                'processed_documents': Document.objects.filter(status=Document.StatusChoices.PROCESSED).count(),
                'processing_documents': Document.objects.filter(status=Document.StatusChoices.PROCESSING).count(),
                'error_documents': Document.objects.filter(status=Document.StatusChoices.ERROR).count(),
            },
            'status_distribution': list(status_stats),
            'file_type_distribution': list(file_type_stats),
            'content_statistics': content_stats,
            'extraction_methods': list(extraction_stats),
            'recent_processed': DocumentListSerializer(recent_processed, many=True).data
        })

class SearchContentView(generics.GenericAPIView):
    """APIView pour rechercher du contenu dans les documents"""
    serializer_class = DocumentContentSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = DocumentProcessingPagination

    @extend_schema(
        summary="Recherche dans le contenu extrait",
        description="Recherche full-text dans le contenu extrait des documents",
        parameters=[
            OpenApiParameter(
                name='q',
                type=OpenApiTypes.STR,
                location=OpenApiParameter.QUERY,
                required=True,
                description='Terme de recherche'
            )
        ],
        responses={
            200: OpenApiTypes.OBJECT,
            'example': {
                'query': 'string',
                'results': [{
                    'document': {},
                    'content_id': 'uuid',
                    'excerpt': 'string',
                    'rank': 'float',
                    'word_count': 'integer',
                    'extraction_confidence': 'float'
                }],
                'count': 'integer'
            }
        }
    )
    def get(self, request):
        """Recherche dans le contenu extrait"""
        query = request.query_params.get('q', '').strip()

        if not query:
            return Response(
                {'error': 'Paramètre de recherche "q" requis'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if len(query) < 3:
            return Response(
                {'error': 'La recherche doit contenir au moins 3 caractères'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Recherche dans le contenu
        from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank

        search_vector = SearchVector('clean_content', weight='A') + \
                        SearchVector('keywords_extracted', weight='B')

        search_query = SearchQuery(query, search_type='websearch')

        results = DocumentContent.objects.annotate(
            search=search_vector,
            rank=SearchRank(search_vector, search_query)
        ).filter(
            search=search_query
        ).filter(
            document__is_deleted=False,
            document__status=Document.StatusChoices.PROCESSED
        ).select_related('document').order_by('-rank')

        # Pagination
        paginator = self.pagination_class()
        page = paginator.paginate_queryset(results, request)

        if page is not None:
            # Construction des résultats avec extraits
            search_results = []
            for content in page:
                # Extraction d'un extrait pertinent
                excerpt = self._extract_search_excerpt(content.clean_content, query)

                search_results.append({
                    'document': DocumentListSerializer(content.document).data,
                    'content_id': str(content.id),
                    'excerpt': excerpt,
                    'rank': float(content.rank),
                    'word_count': content.word_count,
                    'extraction_confidence': float(content.extraction_confidence)
                })

            return paginator.get_paginated_response({
                'query': query,
                'results': search_results
            })

        return Response({
            'query': query,
            'results': [],
            'count': 0
        })

    def _extract_search_excerpt(self, content: str, query: str, max_length: int = 200) -> str:
        """Extrait un passage pertinent du contenu pour la recherche"""
        import re

        # Recherche de la première occurrence du terme
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        match = pattern.search(content)

        if match:
            start = max(0, match.start() - max_length // 2)
            end = min(len(content), match.end() + max_length // 2)

            excerpt = content[start:end].strip()

            if start > 0:
                excerpt = '...' + excerpt
            if end < len(content):
                excerpt = excerpt + '...'

            excerpt = pattern.sub(f'**{match.group()}**', excerpt)

            return excerpt

        return content[:max_length] + ('...' if len(content) > max_length else '')