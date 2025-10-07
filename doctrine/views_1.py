from django.db.models import Count, Avg
from oauth2_provider.contrib.rest_framework import TokenHasScope
from rest_framework import viewsets, status, permissions, generics
from rest_framework.views import APIView
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.pagination import PageNumberPagination
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from django.utils import timezone
from django.db import transaction, models, connection
from django.db.utils import OperationalError
from django.http import HttpResponse
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse
from drf_spectacular.types import OpenApiTypes
import redis
import json
import os
import logging
from typing import Dict, Any, List

from .models import (
    Document, DocumentContent, Topic, Section,
    Paragraph, Table, DocumentCategory, Theme, User
)
from .serializers import (
    DocumentListSerializer, DocumentDetailSerializer, DocumentCreateSerializer,
    DocumentContentSerializer, DocumentContentLightSerializer, DocumentContentWithPreviewSerializer,
    TopicDetailSerializer, SectionSerializer, ParagraphSerializer,
    TableSerializer, RAGQuerySerializer, HuggingFaceRAGGenerationSerializer,
    ChatCompletionSerializer, RAGResponseSerializer, ModelInfoSerializer,
    PageContentSerializer, PageContentResponseSerializer
)
from .services.document_processor import document_processor
from .services.rag_service import PDFRAGService
from .storage import DocumentStorage
from .permissions import CanViewDocument, CanManageDocument
from .tasks import process_document_content

logger = logging.getLogger(__name__)


def execute_with_retry(query_func, max_retries=3):
    """Exécute une requête avec retry en cas d'erreur de connexion PostgreSQL"""
    for attempt in range(max_retries):
        try:
            return query_func()
        except OperationalError as e:
            error_msg = str(e).lower()
            if 'server closed the connection' in error_msg or 'connection' in error_msg:
                logger.warning(f"Erreur connexion PostgreSQL (tentative {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    # Fermer et rouvrir la connexion
                    try:
                        connection.close()
                        connection.connect()
                        logger.info(f"Connexion PostgreSQL rétablie (tentative {attempt + 1})")
                        # Petite pause avant retry
                        import time
                        time.sleep(0.5 * (attempt + 1))
                    except Exception as reconnect_error:
                        logger.error(f"Erreur reconnexion: {reconnect_error}")
                else:
                    # Dernière tentative échouée
                    logger.error(f"Impossible de rétablir la connexion après {max_retries} tentatives")
                    raise
            else:
                # Autre type d'erreur OperationalError
                raise
        except Exception as e:
            # Autres erreurs non liées à la connexion
            logger.error(f"Erreur non-connexion lors de l'exécution: {e}")
            raise

    # Ne devrait jamais arriver ici
    raise OperationalError("Toutes les tentatives de reconnexion ont échoué")


class DocumentContentCache:
    """Gestionnaire de cache Redis pour les contenus de documents"""

    def __init__(self):
        self.redis_client = None
        self.cache_prefix = "dgi:extracted_content"
        self.cache_timeout = 3600  # 1 heure
        self._init_redis()

    def _init_redis(self):
        """Initialise la connexion Redis"""
        try:
            # Configuration Redis depuis les variables d'environnement
            redis_host = os.getenv('REDIS_HOST', '37.60.239.221')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            redis_db = int(os.getenv('REDIS_DB', 1))

            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_timeout=5
            )
            # Test de connexion
            self.redis_client.ping()
            logger.info("Connexion Redis établie pour le cache de documents")
        except Exception as e:
            logger.warning(f"Erreur connexion Redis: {e}")
            self.redis_client = None

    def get_cache_key(self, page: int = 1, page_size: int = 10, search: str = None) -> str:
        """Génère une clé de cache unique"""
        key = f"{self.cache_prefix}:page_{page}:size_{page_size}"
        if search:
            key += f":search_{hash(search)}"
        return key

    def get_cached_data(self, page: int = 1, page_size: int = 10, search: str = None) -> Dict:
        """Récupère les données depuis le cache Redis"""
        if not self.redis_client:
            return None

        try:
            cache_key = self.get_cache_key(page, page_size, search)
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                logger.info(f"Cache hit pour {cache_key}")
                return json.loads(cached_data)

            logger.info(f"Cache miss pour {cache_key}")
            return None
        except Exception as e:
            logger.error(f"Erreur lecture cache Redis: {e}")
            return None

    def set_cached_data(self, data: Dict, page: int = 1, page_size: int = 10, search: str = None):
        """Stocke les données dans le cache Redis"""
        if not self.redis_client:
            return

        try:
            cache_key = self.get_cache_key(page, page_size, search)
            self.redis_client.setex(
                cache_key,
                self.cache_timeout,
                json.dumps(data, default=str)  # default=str pour les dates/UUID
            )
            logger.info(f"Données mises en cache: {cache_key}")
        except Exception as e:
            logger.error(f"Erreur écriture cache Redis: {e}")

    def invalidate_cache(self):
        """Invalide tout le cache des contenus extraits"""
        if not self.redis_client:
            return

        try:
            pattern = f"{self.cache_prefix}:*"
            keys = self.redis_client.keys(pattern)

            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cache invalidé: {len(keys)} clés supprimées")
        except Exception as e:
            logger.error(f"Erreur invalidation cache Redis: {e}")

    def populate_cache_from_db(self, page_size: int = 10, max_pages: int = 10):
        """Peuple le cache depuis la base de données"""
        if not self.redis_client:
            return

        try:
            # S'assurer que la connexion DB est active
            if connection.connection is None or connection.connection.closed:
                connection.connect()
            logger.info("Début du peuplement du cache depuis la DB...")

            queryset = DocumentContent.objects.select_related(
                'document'
            ).filter(
                document__status=Document.StatusChoices.PROCESSED,
                processing_status=DocumentContent.ProcessingStatus.COMPLETED
            ).order_by('-created_at')

            for page in range(1, max_pages + 1):
                try:
                    def execute_page_query():
                        offset = (page - 1) * page_size
                        return list(queryset[offset:offset + page_size])

                    # Exécuter avec retry
                    items = self._execute_with_retry(execute_page_query)

                    if not items:
                        break

                except OperationalError as e:
                    logger.error(f"Erreur PostgreSQL lors du peuplement cache page {page}: {e}")
                    break  # Arrêter le peuplement en cas d'erreur de connexion
                except Exception as e:
                    logger.error(f"Erreur lors du peuplement cache page {page}: {e}")
                    continue  # Continuer avec la page suivante

                # Sérialiser les données avec aperçu du contenu
                serializer = DocumentContentWithPreviewSerializer(items, many=True)

                # Préparer la réponse
                response_data = {
                    'count': len(items),
                    'page': page,
                    'page_size': page_size,
                    'has_next': len(items) == page_size,
                    'has_previous': page > 1,
                    'results': serializer.data,
                    'cached_at': timezone.now().isoformat()
                }

                # Mettre en cache
                self.set_cached_data(response_data, page, page_size)

                logger.info(f"Cache peuplé pour la page {page} ({len(items)} éléments)")

            logger.info("Peuplement du cache terminé")

        except Exception as e:
            logger.error(f"Erreur peuplement cache: {e}")


# Instance globale du gestionnaire de cache
document_cache = DocumentContentCache()


class AllExtractedContentView(generics.GenericAPIView):
    """
    Endpoint non-protégé pour récupérer tous les contenus extraits de tous les documents
    Utilise Redis comme cache pour éviter les requêtes lourdes à la base de données
    """
    permission_classes = []  # Aucune protection comme demandé
    authentication_classes = []  # Pas d'authentification requise
    queryset = DocumentContent.objects.none()  # Queryset vide pour éviter l'erreur DRF
    filter_backends = []  # Désactiver les filtres DRF pour éviter les erreurs
    serializer_class = DocumentContentWithPreviewSerializer  # Classe de sérialiseur pour DRF

    def _ensure_db_connection(self):
        """S'assure que la connexion à la base de données est active"""
        try:
            if connection.connection is None or connection.connection.closed:
                connection.connect()
                logger.info("Connexion base de données rétablie")
        except Exception as e:
            logger.warning(f"Impossible de rétablir la connexion DB: {e}")
            connection.close()
            connection.connect()

    def _execute_with_retry(self, query_func, max_retries=3):
        """Exécute une requête avec retry en cas d'erreur de connexion PostgreSQL"""
        for attempt in range(max_retries):
            try:
                return query_func()
            except OperationalError as e:
                error_msg = str(e).lower()
                if 'server closed the connection' in error_msg or 'connection' in error_msg:
                    logger.warning(f"Erreur connexion PostgreSQL (tentative {attempt + 1}/{max_retries}): {e}")

                    if attempt < max_retries - 1:
                        # Fermer et rouvrir la connexion
                        try:
                            connection.close()
                            connection.connect()
                            logger.info(f"Connexion PostgreSQL rétablie (tentative {attempt + 1})")
                            # Petite pause avant retry
                            import time
                            time.sleep(0.5 * (attempt + 1))
                        except Exception as reconnect_error:
                            logger.error(f"Erreur reconnexion: {reconnect_error}")
                    else:
                        # Dernière tentative échouée
                        logger.error(f"Impossible de rétablir la connexion après {max_retries} tentatives")
                        raise
                else:
                    # Autre type d'erreur OperationalError
                    raise
            except Exception as e:
                # Autres erreurs non liées à la connexion
                logger.error(f"Erreur non-connexion lors de l'exécution: {e}")
                raise

        # Ne devrait jamais arriver ici
        raise OperationalError("Toutes les tentatives de reconnexion ont échoué")

    @extend_schema(
        summary="Récupère tous les contenus extraits",
        description="Endpoint public pour récupérer tous les contenus extraits de tous les documents traités via cache Redis",
        parameters=[
            OpenApiParameter(
                name='search',
                description='Recherche dans le contenu',
                required=False,
                type=OpenApiTypes.STR
            ),
            OpenApiParameter(
                name='page_size',
                description='Nombre d\'éléments par page (max 50)',
                required=False,
                type=OpenApiTypes.INT
            ),
            OpenApiParameter(
                name='page',
                description='Numéro de page',
                required=False,
                type=OpenApiTypes.INT
            ),
            OpenApiParameter(
                name='refresh_cache',
                description='Force le rafraîchissement du cache',
                required=False,
                type=OpenApiTypes.BOOL
            )
        ]
    )
    def get(self, request, *args, **kwargs):
        """Liste tous les contenus extraits depuis le cache Redis"""
        try:
            # S'assurer que la connexion DB est active
            self._ensure_db_connection()
            # Paramètres de pagination
            page_size = min(int(request.query_params.get('page_size', 20)), 50)
            page_number = max(int(request.query_params.get('page', 1)), 1)
            search = request.query_params.get('search', '').strip() or None
            refresh_cache = request.query_params.get('refresh_cache', '').lower() in ('true', '1')

            # Si refresh_cache demandé, invalider le cache
            if refresh_cache:
                document_cache.invalidate_cache()
                logger.info("Cache invalidé sur demande")

            # Essayer de récupérer depuis le cache Redis
            cached_data = document_cache.get_cached_data(page_number, page_size, search)

            if cached_data:
                # Construire les URLs de navigation
                base_url = request.build_absolute_uri().split('?')[0]

                # Ajouter les URLs de navigation
                if cached_data.get('has_next'):
                    next_params = f"page={page_number + 1}&page_size={page_size}"
                    if search:
                        next_params += f"&search={search}"
                    cached_data['next'] = f"{base_url}?{next_params}"
                else:
                    cached_data['next'] = None

                if cached_data.get('has_previous'):
                    prev_params = f"page={page_number - 1}&page_size={page_size}"
                    if search:
                        prev_params += f"&search={search}"
                    cached_data['previous'] = f"{base_url}?{prev_params}"
                else:
                    cached_data['previous'] = None

                cached_data['cache_hit'] = True
                return Response(cached_data)

            # Cache miss : récupérer depuis la DB et mettre en cache
            logger.info(f"Cache miss - récupération depuis DB pour page {page_number}")

            # Fonction pour exécuter la requête
            def execute_query():
                # Requête à la base de données optimisée pour éviter les timeouts
                queryset = DocumentContent.objects.select_related(
                    'document'
                ).filter(
                    document__status=Document.StatusChoices.PROCESSED,
                    processing_status=DocumentContent.ProcessingStatus.COMPLETED
                ).order_by('-created_at')

                # Appliquer la recherche si nécessaire
                if search:
                    queryset = queryset.filter(
                        models.Q(document__title__icontains=search) |
                        models.Q(document__description__icontains=search)
                    )

                # Pagination
                offset = (page_number - 1) * page_size
                return list(queryset[offset:offset + page_size])

            # Exécuter avec retry
            items = self._execute_with_retry(execute_query)

            # Sérialiser les données avec aperçu du contenu
            serializer = DocumentContentWithPreviewSerializer(items, many=True)

            # Préparer la réponse
            base_url = request.build_absolute_uri().split('?')[0]
            has_next = len(items) == page_size
            has_previous = page_number > 1

            next_url = None
            previous_url = None

            if has_next:
                next_params = f"page={page_number + 1}&page_size={page_size}"
                if search:
                    next_params += f"&search={search}"
                next_url = f"{base_url}?{next_params}"

            if has_previous:
                prev_params = f"page={page_number - 1}&page_size={page_size}"
                if search:
                    prev_params += f"&search={search}"
                previous_url = f"{base_url}?{prev_params}"

            response_data = {
                'count': len(items),
                'page': page_number,
                'page_size': page_size,
                'next': next_url,
                'previous': previous_url,
                'has_next': has_next,
                'has_previous': has_previous,
                'results': serializer.data,
                'cache_hit': False,
                'cached_at': timezone.now().isoformat()
            }

            document_cache.set_cached_data(response_data, page_number, page_size, search)

            return Response(response_data)

        except OperationalError as e:
            logger.exception(f"Erreur PostgreSQL dans AllExtractedContentView: {e}")

            # Tentative de fallback vers le cache
            try:
                cached_data = document_cache.get_cached_data(page_number, page_size, search)
                if cached_data:
                    cached_data['cache_hit'] = True
                    cached_data['fallback_reason'] = 'PostgreSQL connection error'
                    logger.info("Fallback vers le cache Redis après erreur PostgreSQL")
                    return Response(cached_data)
            except Exception as cache_error:
                logger.error(f"Erreur cache lors du fallback: {cache_error}")

            return Response({
                'error': 'Erreur de connexion à la base de données PostgreSQL',
                'details': 'Le serveur de base de données a fermé la connexion de manière inattendue',
                'count': 0,
                'results': [],
                'cache_hit': False,
                'suggestion': 'Problème de connexion PostgreSQL. Contactez l\'administrateur système.',
                'technical_details': str(e)
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        except Exception as e:
            logger.exception(f"Erreur générale dans AllExtractedContentView: {e}")

            return Response({
                'error': 'Erreur lors de la récupération des données',
                'details': str(e),
                'count': 0,
                'results': [],
                'cache_hit': False,
                'suggestion': 'Erreur inattendue. Essayez de réduire page_size.'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocumentStructureView(generics.GenericAPIView):
    """
    Endpoint public pour récupérer la structure complète d'un document spécifique
    Retourne toutes les sections et paragraphes associés au document
    """
    permission_classes = []  # Aucune protection - endpoint public
    authentication_classes = []  # Pas d'authentification requise
    queryset = Document.objects.none()  # Queryset vide pour éviter l'erreur DRF
    filter_backends = []  # Désactiver les filtres DRF pour éviter les erreurs
    serializer_class = DocumentDetailSerializer  # Classe de sérialiseur pour DRF

    @extend_schema(
        summary="Structure complète d'un document",
        description="Endpoint public pour récupérer la structure hiérarchique complète d'un document (topics, sections, paragraphes)",
        parameters=[
            OpenApiParameter(
                name='document_id',
                description='UUID du document',
                required=True,
                type=OpenApiTypes.STR,
                location=OpenApiParameter.PATH
            ),
            OpenApiParameter(
                name='include_content',
                description='Inclure le contenu complet des paragraphes (défaut: true)',
                required=False,
                type=OpenApiTypes.BOOL
            ),
            OpenApiParameter(
                name='max_content_length',
                description='Longueur maximale du contenu des paragraphes (défaut: 1000)',
                required=False,
                type=OpenApiTypes.INT
            )
        ],
        responses={
            200: OpenApiTypes.OBJECT,
            404: OpenApiResponse(description='Document non trouvé'),
            'example': {
                'document_id': 'uuid',
                'document_title': 'string',
                'document_description': 'string',
                'document_status': 'string',
                'total_topics': 'integer',
                'total_sections': 'integer',
                'total_paragraphs': 'integer',
                'structure': [{
                    'topic_id': 'uuid',
                    'topic_title': 'string',
                    'topic_content': 'string',
                    'topic_type': 'string',
                    'level': 'integer',
                    'order_index': 'integer',
                    'sections': [{
                        'section_id': 'uuid',
                        'section_title': 'string',
                        'section_content': 'string',
                        'section_type': 'string',
                        'order_index': 'integer',
                        'paragraphs': [{
                            'paragraph_id': 'uuid',
                            'content': 'string',
                            'paragraph_type': 'string',
                            'word_count': 'integer',
                            'order_index': 'integer'
                        }]
                    }]
                }]
            }
        }
    )
    def get(self, request, document_id, *args, **kwargs):
        """Récupère la structure complète d'un document avec ses sections et paragraphes"""
        try:
            # Paramètres de la requête
            include_content = request.query_params.get('include_content', 'true').lower() in ('true', '1')
            max_content_length = int(request.query_params.get('max_content_length', 1000))

            # Récupérer le document avec ses relations
            try:
                document = Document.objects.select_related('content').prefetch_related(
                    'topics__sections__paragraphs'
                ).get(id=document_id, is_deleted=False)
            except Document.DoesNotExist:
                return Response({
                    'error': 'Document non trouvé',
                    'document_id': document_id
                }, status=status.HTTP_404_NOT_FOUND)

            # Vérifier que le document est traité
            if document.status != Document.StatusChoices.PROCESSED:
                return Response({
                    'error': 'Le document n\'a pas encore été traité ou est en erreur',
                    'document_id': str(document.id),
                    'current_status': document.status
                }, status=status.HTTP_400_BAD_REQUEST)

            # Construire la structure
            structure_data = []
            total_sections = 0
            total_paragraphs = 0

            # Récupérer les topics avec leurs sections et paragraphes
            topics = document.topics.filter(is_deleted=False).order_by('order_index')

            for topic in topics:
                topic_data = {
                    'topic_id': str(topic.id),
                    'topic_title': topic.title,
                    'topic_content': topic.content if include_content else None,
                    'topic_type': topic.topic_type,
                    'level': topic.level,
                    'order_index': topic.order_index,
                    'sections': []
                }

                # Récupérer les sections du topic
                sections = list(topic.sections.filter(is_deleted=False).order_by('order_index'))
                total_sections += len(sections)

                for section in sections:
                    section_data = {
                        'section_id': str(section.id),
                        'section_title': section.title,
                        'section_content': section.content if include_content else None,
                        'section_type': section.section_type,
                        'order_index': section.order_index,
                        'paragraphs': []
                    }

                    # Récupérer les paragraphes de la section
                    paragraphs = list(section.paragraphs.all().order_by('order_index'))
                    total_paragraphs += len(paragraphs)

                    for paragraph in paragraphs:
                        # Tronquer le contenu si nécessaire
                        content = paragraph.content
                        if include_content and max_content_length > 0 and len(content) > max_content_length:
                            content = content[:max_content_length] + '...'

                        paragraph_data = {
                            'paragraph_id': str(paragraph.id),
                            'content': content if include_content else None,
                            'paragraph_type': paragraph.paragraph_type,
                            'word_count': paragraph.word_count,
                            'order_index': paragraph.order_index
                        }
                        section_data['paragraphs'].append(paragraph_data)

                    topic_data['sections'].append(section_data)

                structure_data.append(topic_data)

            # Préparer la réponse
            response_data = {
                'document_id': str(document.id),
                'document_title': document.title,
                'document_description': document.description,
                'document_status': document.status,
                'document_language': document.language,
                'document_created_at': document.created_at.isoformat(),
                'document_updated_at': document.updated_at.isoformat(),
                'total_topics': len(structure_data),
                'total_sections': total_sections,
                'total_paragraphs': total_paragraphs,
                'structure': structure_data,
                'content_included': include_content,
                'max_content_length': max_content_length if include_content else None,
                'retrieved_at': timezone.now().isoformat()
            }

            # Ajouter des statistiques du contenu si disponible
            if hasattr(document, 'content') and document.content:
                content = document.content
                response_data['content_stats'] = {
                    'word_count': content.word_count,
                    'page_count': content.page_count,
                    'extraction_confidence': float(content.extraction_confidence),
                    'processing_status': content.processing_status
                }

            return Response(response_data, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response({
                'error': 'Paramètres invalides',
                'details': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.exception(f"Erreur dans DocumentStructureView pour document {document_id}: {e}")
            return Response({
                'error': 'Erreur lors de la récupération de la structure du document',
                'details': str(e),
                'document_id': document_id
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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
    pagination_class = None
    parser_classes = [MultiPartParser, FormParser]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ['title', 'description', 'original_filename']
    ordering_fields = ['created_at', 'title', 'status', 'file_size']
    ordering = ['-created_at']
    permission_classes = [permissions.IsAuthenticated, TokenHasScope]
    required_scopes = ['read', 'write']

    # permission_classes = [permissions.IsAuthenticated]


    def get_serializer_class(self):
        """Sélectionne le bon serializer selon l'action"""
        if self.action == 'list':
            return DocumentListSerializer
        elif self.action == 'upload_and_process':
            return DocumentListSerializer
        elif self.action == 'create':
            return DocumentCreateSerializer
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
                auto_process_raw = request.data.get('auto_process', True)
                auto_process = str(auto_process_raw).lower() in ('true', '1', 'yes', 'on')

                # Validation de la taille du fichier
                file_size = uploaded_file.size
                if file_size is None or file_size <= 0:
                    return Response(
                        {'error': 'Impossible de déterminer la taille du fichier'},
                        status=status.HTTP_400_BAD_REQUEST
                    )

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
                file_size_mb = file_size / (1024 * 1024)

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
                    saved_or_doc, is_duplicate = DocumentStorage.save_document_file(
                        uploaded_file, uploaded_file.name, user
                    )

                    if is_duplicate:
                        # When duplicate, the first return value is the existing Document instance
                        existing_doc = saved_or_doc
                        return Response(
                            {
                                'error': 'Ce fichier existe déjà dans le système',
                                'duplicate_document_id': str(existing_doc.id)
                            },
                            status=status.HTTP_409_CONFLICT
                        )
                    else:
                        saved_path = saved_or_doc

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
                    file_size=file_size,
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
                        # Traitement synchrone pour retourner le contenu extrait immédiatement
                        success = document_processor.process_document(document)
                        if not success:
                            logger.warning(f"Échec du traitement automatique pour le document {document.id}")
                    except Exception as e:
                        logger.error(f"Erreur lors du lancement du traitement: {str(e)}")

                document_data = DocumentDetailSerializer(document).data

                content_data = None
                if hasattr(document, 'content'):
                    content_data = DocumentContentSerializer(document.content).data

                structure_data = self._get_structure_data(document)

                tables_qs = Table.objects.filter(section__topic__document=document).order_by('order_index')
                tables_data = TableSerializer(tables_qs, many=True).data

                return Response(
                    {
                        'message': 'Document uploadé avec succès',
                        'document': document_data,
                        'content': content_data,
                        'structure': structure_data,
                        'tables': tables_data,
                        'processing_started': bool(auto_process)
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
    pagination_class = None

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



PDF_PATH = "media/doctrine_fiscale.pdf"


class RAGQueryView(APIView):
    """APIView RAG unifié pour PDF: recherche sémantique avec génération Gemini"""
    permission_classes = []  # Accès public
    authentication_classes = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rag_service = PDFRAGService(media_folder="media")

    @extend_schema(
        summary="RAG - Recherche sémantique sur PDF avec génération",
        description=(
            "Récupère les top-k passages pertinents du PDF spécifié avec génération Gemini.\n"
            "Mode toujours 'pdf_gemini_rag' (unifié).\n"
            "Requiert google-generativeai et langchain pour le traitement PDF."
        ),
        request=RAGQuerySerializer,
        responses={200: RAGResponseSerializer}
    )
    def post(self, request):
        """Recherche RAG sur PDF avec génération"""
        serializer = RAGQuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data
        query = validated_data['query']
        k = validated_data.get('k', 5)

        data = self.rag_service.query(question=query, k=k)

        adapted_data = {
            "query": data["query"],
            "scope": "pdf",  # Fixed for PDF
            "k": k,
            "count": data["count"],
            "took_ms": data["took_ms"],
            "mode": data["mode"],
            "results": data["context_documents"],
            "generated_response": data["answer"],
            "generation_metadata": data["generation_metadata"]
        }

        status_code = status.HTTP_200_OK
        if data.get('generation_metadata', {}).get('error'):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        return Response(adapted_data, status=status_code)

    @extend_schema(
        summary="RAG - Recherche simple sur PDF (GET, rétrocompatibilité)",
        description="Endpoint GET pour rétrocompatibilité avec l'ancienne API",
        parameters=[
            OpenApiParameter(name='q', type=OpenApiTypes.STR, required=True, location=OpenApiParameter.QUERY,
                             description='Question / requête utilisateur'),
            OpenApiParameter(name='k', type=OpenApiTypes.INT, required=False, location=OpenApiParameter.QUERY,
                             description='Nombre de passages à retourner (top-k)'),
        ],
        responses={200: OpenApiTypes.OBJECT}
    )
    def get(self, request):
        """Endpoint GET pour rétrocompatibilité"""
        q = (request.query_params.get('q') or '').strip()
        if not q:
            return Response({'error': 'Paramètre q requis'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            k = int(request.query_params.get('k', 5))
        except Exception:
            k = 5
        k = max(1, min(50, k))

        data = self.rag_service.query(question=q, k=k)

        adapted_data = {
            "query": data["query"],
            "scope": "pdf",
            "k": k,
            "count": data["count"],
            "took_ms": data["took_ms"],
            "mode": data["mode"],
            "results": data["context_documents"],
            "generated_response": data["answer"],
            "generation_metadata": data["generation_metadata"]
        }

        status_code = status.HTTP_200_OK
        if data.get('generation_metadata', {}).get('error'):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Response(adapted_data, status=status_code)





class HuggingFaceRAGView(generics.GenericAPIView):
    """Vue dédiée à la génération RAG Hugging Face"""
    permission_classes = []  # Accès public
    authentication_classes = []
    serializer_class = HuggingFaceRAGGenerationSerializer

    @extend_schema(
        summary="Génération RAG Hugging Face",
        description=(
            "Génère une réponse complète à une question en utilisant Hugging Face RAG.\n"
            "Combine retrieval (depuis votre base de données) + génération pour une réponse naturelle.\n"
            "Requiert l'installation de transformers et torch."
        ),
        request=HuggingFaceRAGGenerationSerializer,
        responses={200: OpenApiTypes.OBJECT}
    )
    def post(self, request):
        """Génère une réponse complète avec Hugging Face RAG"""
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data

        try:
            from .services.huggingface_rag import huggingface_rag_service

            # Paramètres de génération
            generation_kwargs = {
                'max_new_tokens': validated_data.get('max_new_tokens', 100),
                'num_beams': validated_data.get('num_beams', 4),
                'do_sample': validated_data.get('do_sample', False),
                'temperature': validated_data.get('temperature', 1.0),
                'top_p': validated_data.get('top_p', 1.0),
                'early_stopping': validated_data.get('early_stopping', True),
            }

            # Génération
            response_data = huggingface_rag_service.generate_response(
                validated_data['query'],
                use_custom_context=validated_data.get('use_custom_context', True),
                **generation_kwargs
            )

            return Response(response_data, status=status.HTTP_200_OK)

        except ImportError:
            return Response({
                'error': 'Service Hugging Face RAG non disponible',
                'details': 'Installez: pip install transformers torch accelerate'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        except Exception as e:
            logger.exception('Erreur génération Hugging Face RAG: %s', e)
            return Response({
                'error': f'Erreur de génération: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatCompletionView(generics.GenericAPIView):
    """Vue de conversation compatible OpenAI avec Hugging Face RAG"""
    permission_classes = []  # Accès public
    authentication_classes = []
    serializer_class = ChatCompletionSerializer

    @extend_schema(
        summary="Chat completion avec RAG (compatible OpenAI)",
        description=(
            "Interface de conversation compatible avec l'API OpenAI.\n"
            "Utilise Hugging Face RAG avec le contexte de votre base de données.\n"
            "Format de réponse similaire à OpenAI pour faciliter l'intégration."
        ),
        request=ChatCompletionSerializer,
        responses={200: OpenApiTypes.OBJECT}
    )
    def post(self, request):
        """Conversation style OpenAI avec RAG backend"""
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data

        try:
            from .services.huggingface_rag import huggingface_rag_service

            # Paramètres de génération
            generation_kwargs = {
                'max_new_tokens': validated_data.get('max_new_tokens', 100),
                'temperature': validated_data.get('temperature', 1.0),
                'top_p': validated_data.get('top_p', 1.0),
                'num_beams': validated_data.get('num_beams', 4),
            }

            # Génération avec interface compatible OpenAI
            response_data = huggingface_rag_service.chat_completion(
                validated_data['messages'],
                **generation_kwargs
            )

            return Response(response_data, status=status.HTTP_200_OK)

        except ImportError:
            return Response({
                'error': {
                    'message': 'Service Hugging Face RAG non disponible',
                    'type': 'service_unavailable',
                    'code': 'hf_rag_missing'
                }
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        except Exception as e:
            logger.exception('Erreur chat completion: %s', e)
            return Response({
                'error': {
                    'message': f'Erreur de génération: {str(e)}',
                    'type': 'generation_error',
                    'code': 'generation_failed'
                }
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)







class RAGModelInfoView(generics.GenericAPIView):
    """Vue pour obtenir des informations sur les modèles RAG chargés"""
    permission_classes = []  # Accès public
    authentication_classes = []
    serializer_class = ModelInfoSerializer

    @extend_schema(
        summary="Informations sur les modèles RAG",
        description="Retourne des informations sur l'état des modèles RAG (chargement, device, etc.)",
        responses={200: ModelInfoSerializer}
    )
    def get(self, request):
        """Informations sur les modèles RAG"""
        try:
            from .services.huggingface_rag import huggingface_rag_service

            model_info = huggingface_rag_service.get_model_info()
            return Response(model_info, status=status.HTTP_200_OK)

        except ImportError:
            return Response({
                'model_name': 'N/A',
                'retriever_name': 'N/A',
                'device': 'N/A',
                'model_loaded': False,
                'use_custom_retriever': True,
                'use_gpu': False,
                'dependencies_available': False,
                'error': 'Dépendances Hugging Face non installées'
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception('Erreur récupération info modèle: %s', e)
            return Response({
                'error': f'Erreur: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocumentPageContentView(generics.GenericAPIView):
    """Vue pour récupérer le contenu de documents par page"""
    permission_classes = []  # Accès public
    authentication_classes = []

    @extend_schema(
        operation_id='get_document_page_content',
        description='Récupère le contenu d\'un document par page spécifique ou plage de pages',
        parameters=[
            OpenApiParameter(
                name='page_number',
                description='Numéro de page spécifique (1-N)',
                required=False,
                type=OpenApiTypes.INT
            ),
            OpenApiParameter(
                name='start_page',
                description='Page de début pour une plage',
                required=False,
                type=OpenApiTypes.INT
            ),
            OpenApiParameter(
                name='end_page',
                description='Page de fin pour une plage',
                required=False,
                type=OpenApiTypes.INT
            )
        ],
        responses={
            200: PageContentResponseSerializer,
            400: OpenApiResponse(description='Paramètres de page invalides'),
            404: OpenApiResponse(description='Document ou contenu non trouvé')
        }
    )
    def get(self, request, document_id):
        """Récupère le contenu par page d'un document"""
        try:
            # Récupérer le document
            document = Document.objects.select_related('content').get(id=document_id)

            if not document.content:
                return Response({
                    'error': 'Aucun contenu trouvé pour ce document'
                }, status=status.HTTP_404_NOT_FOUND)

            # Valider les paramètres de page
            serializer = PageContentSerializer(data=request.query_params)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            validated_data = serializer.validated_data
            content = document.content

            # Vérifier que le contenu est organisé par pages
            if not content.has_page_based_content():
                return Response({
                    'error': 'Ce document n\'a pas de contenu organisé par pages'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Récupérer le contenu selon les paramètres
            page_number = validated_data.get('page_number')
            start_page = validated_data.get('start_page')
            end_page = validated_data.get('end_page')

            if page_number:
                # Page spécifique
                page_content = content.get_page_content(page_number)
                if not page_content:
                    return Response({
                        'error': f'Page {page_number} non trouvée'
                    }, status=status.HTTP_404_NOT_FOUND)

                requested_pages = {str(page_number): page_content}
            else:
                # Plage de pages
                requested_pages = content.get_pages_content(start_page, end_page)
                if not requested_pages:
                    return Response({
                        'error': f'Aucune page trouvée dans la plage {start_page}-{end_page or "fin"}'
                    }, status=status.HTTP_404_NOT_FOUND)

            # Préparer la réponse
            response_data = {
                'document_id': str(document.id),
                'document_title': document.title,
                'total_pages': content.get_total_pages(),
                'requested_pages': requested_pages
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Document.DoesNotExist:
            return Response({
                'error': 'Document non trouvé'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.exception('Erreur récupération contenu par page: %s', e)
            return Response({
                'error': f'Erreur: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
