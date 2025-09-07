from django.utils import timezone
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from django.db import models
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.http import HttpResponse
from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank


from .models import (
    User, Theme, DocumentCategory, Document, DocumentContent,
    Topic, Section, Paragraph, Table
)
from .serializers import (
    # User serializers
    UserListSerializer, UserDetailSerializer, UserCreateSerializer, UserUpdateSerializer,
    # Theme serializers
    ThemeListSerializer, ThemeDetailSerializer, ThemeCreateUpdateSerializer,
    # Document Category serializers
    DocumentCategoryListSerializer, DocumentCategoryDetailSerializer,
    # Document serializers
    DocumentListSerializer, DocumentDetailSerializer, DocumentCreateSerializer, DocumentUpdateSerializer,
    # Document Content serializers
    DocumentContentSerializer,
    # Topic serializers
    TopicListSerializer, TopicDetailSerializer,
    # Section serializers
    SectionListSerializer, SectionDetailSerializer,
    # Paragraph serializers
    ParagraphListSerializer, ParagraphDetailSerializer,
    # Table serializers
    TableListSerializer, TableDetailSerializer
)
from .filters import (
    DocumentProcessingFilter, DocumentContentFilter, TopicFilter,
    SectionFilter, ParagraphFilter, TableFilter
)
from .permissions import (
    CanManageDocument, CanViewDocument, IsOwnerOrManagerOrReadOnly,
    DocumentViewPermissions, DocumentManagePermissions, HasAPIAccess
)

User = get_user_model()


class StandardPagination(PageNumberPagination):
    """Pagination standard pour toutes les vues"""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

    def get_paginated_response(self, data):
        """Réponse paginée personnalisée"""
        return Response({
            'links': {
                'next': self.get_next_link(),
                'previous': self.get_previous_link()
            },
            'count': self.page.paginator.count,
            'total_pages': self.page.paginator.num_pages,
            'page_size': self.get_page_size(self.request),
            'current_page': self.page.number,
            'results': data
        })


class LargeResultsSetPagination(PageNumberPagination):
    """Pagination pour de gros volumes de données"""
    page_size = 50
    page_size_query_param = 'page_size'
    max_page_size = 200


class SmallResultsSetPagination(PageNumberPagination):
    """Pagination pour de petits volumes de données"""
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 50


class BaseModelViewSet(viewsets.ModelViewSet):
    """ViewSet de base avec configuration commune"""
    pagination_class = StandardPagination
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Queryset de base avec optimisations"""
        queryset = super().get_queryset()

        # Exclure les éléments supprimés logiquement si applicable
        if hasattr(self.queryset.model, 'is_deleted'):
            queryset = queryset.filter(is_deleted=False)

        return queryset

    def get_serializer_class(self):
        """Sélection du serializer selon l'action"""
        if hasattr(self, 'serializer_classes'):
            return self.serializer_classes.get(self.action, self.serializer_class)
        return super().get_serializer_class()

    def perform_create(self, serializer):
        """Actions lors de la création"""
        # Assigner l'utilisateur courant si applicable
        if hasattr(serializer.Meta.model, 'created_by'):
            serializer.save(created_by=self.request.user)
        elif hasattr(serializer.Meta.model, 'uploaded_by'):
            serializer.save(uploaded_by=self.request.user)
        else:
            serializer.save()

    def perform_destroy(self, instance):
        """Suppression logique si disponible, sinon suppression physique"""
        if hasattr(instance, 'soft_delete'):
            instance.soft_delete(user=self.request.user)
        else:
            instance.delete()

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Action pour obtenir des statistiques"""
        cache_key = f"{self.__class__.__name__}_stats_{request.user.id}"
        cached_stats = cache.get(cache_key)

        if cached_stats:
            return Response(cached_stats)

        queryset = self.get_queryset()
        stats = {
            'total': queryset.count(),
        }

        # Ajouter des statistiques spécifiques selon le modèle
        if hasattr(queryset.model, 'is_active'):
            stats['active'] = queryset.filter(is_active=True).count()

        if hasattr(queryset.model, 'status'):
            status_distribution = queryset.values('status').annotate(
                count=models.Count('id')
            ).annotate(
                count=models.Count('id')
            ).order_by('status')
            stats['status_distribution'] = list(status_distribution)

        # Cache pour 5 minutes
        cache.set(cache_key, stats, 300)
        return Response(stats)


class UserViewSet(BaseModelViewSet):
    """ViewSet pour la gestion des utilisateurs"""
    queryset = User.objects.select_related('manager').prefetch_related('team_members')
    search_fields = ['first_name', 'last_name', 'email', 'department', 'position']
    ordering_fields = ['created_at', 'last_name', 'email', 'hire_date']
    ordering = ['-created_at']

    serializer_classes = {
        'list': UserListSerializer,
        'retrieve': UserDetailSerializer,
        'create': UserCreateSerializer,
        'update': UserUpdateSerializer,
        'partial_update': UserUpdateSerializer,
    }
    serializer_class = UserDetailSerializer

    permission_classes = [permissions.IsAuthenticated, IsOwnerOrManagerOrReadOnly]

    def get_queryset(self):
        """Queryset optimisé avec gestion des permissions"""
        queryset = super().get_queryset()
        user = self.request.user

        # Handle schema generation by drf-yasg
        if getattr(self, 'swagger_fake_view', False):
            return queryset.none()

        if not user.is_authenticated:
            return queryset.none()

        if user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
            return queryset
        elif user.role == User.RoleChoices.ANALYST:
            return queryset.filter(department=user.department)
        else:
            return queryset.filter(id=user.id)

    @action(detail=True, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def set_password(self, request, pk=None):
        """Changer le mot de passe d'un utilisateur"""
        user = self.get_object()

        # Vérifications de permissions
        if request.user != user and not request.user.is_staff:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        password = request.data.get('password')
        if not password:
            return Response({'error': 'Password required'}, status=status.HTTP_400_BAD_REQUEST)

        # Validation de la force du mot de passe
        if len(password) < 8:
            return Response({'error': 'Password must be at least 8 characters'}, status=status.HTTP_400_BAD_REQUEST)

        user.set_password(password)
        user.save()
        return Response({'message': 'Password updated successfully'})

    @action(detail=True, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def generate_api_key(self, request, pk=None):
        """Générer une nouvelle clé API"""
        user = self.get_object()

        # Vérifications de permissions
        if request.user != user and not request.user.is_staff:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        if not user.api_access_enabled:
            return Response({'error': 'API access not enabled'}, status=status.HTTP_403_FORBIDDEN)

        api_key = user.generate_api_key()
        return Response({'api_key': api_key})

    @action(detail=False, methods=['get'])
    def me(self, request):
        """Profil de l'utilisateur courant"""
        serializer = UserDetailSerializer(request.user)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def team(self, request):
        """Équipe de l'utilisateur courant"""
        team_members = request.user.team_members.all()
        serializer = UserListSerializer(team_members, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def documents(self, request, pk=None):
        """Documents uploadés par l'utilisateur"""
        user = self.get_object()
        documents = user.uploaded_documents.filter(is_deleted=False)

        page = self.paginate_queryset(documents)
        if page is not None:
            serializer = DocumentListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = DocumentListSerializer(documents, many=True)
        return Response(serializer.data)


class ThemeViewSet(BaseModelViewSet):
    """ViewSet pour la gestion des thèmes"""
    queryset = Theme.objects.select_related('parent_theme', 'created_by').prefetch_related('children', 'managed_by')
    search_fields = ['name', 'description', 'code']
    ordering_fields = ['name', 'level', 'order_index', 'documents_count', 'created_at']
    ordering = ['level', 'order_index', 'name']

    serializer_classes = {
        'list': ThemeListSerializer,
        'retrieve': ThemeDetailSerializer,
        'create': ThemeCreateUpdateSerializer,
        'update': ThemeCreateUpdateSerializer,
        'partial_update': ThemeCreateUpdateSerializer,
    }
    serializer_class = ThemeDetailSerializer

    def get_queryset(self):
        """Queryset avec filtrage par visibilité"""
        queryset = super().get_queryset()
        user = self.request.user

        # Handle schema generation by drf-yasg
        if getattr(self, 'swagger_fake_view', False):
            return queryset.none()

        if not user.is_authenticated:
            return queryset.none()

        if user.is_staff or user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
            return queryset
        else:
            return queryset.filter(is_public=True, is_active=True)

    @action(detail=True, methods=['get'])
    def children(self, request, pk=None):
        """Obtenir les sous-thèmes d'un thème"""
        theme = self.get_object()
        children = theme.get_children()
        serializer = ThemeListSerializer(children, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def ancestors(self, request, pk=None):
        """Obtenir les thèmes parents d'un thème"""
        theme = self.get_object()
        ancestors = theme.get_ancestors()
        serializer = ThemeListSerializer(ancestors, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def documents(self, request, pk=None):
        """Obtenir les documents d'un thème"""
        theme = self.get_object()
        documents = theme.documents.filter(is_deleted=False, status=Document.StatusChoices.PUBLISHED)

        # Pagination
        page = self.paginate_queryset(documents)
        if page is not None:
            serializer = DocumentListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = DocumentListSerializer(documents, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def tree(self, request):
        """Structure hiérarchique des thèmes"""
        cache_key = f"theme_tree_{request.user.id}"
        cached_tree = cache.get(cache_key)

        if cached_tree:
            return Response(cached_tree)

        root_themes = self.get_queryset().filter(parent_theme=None)
        serializer = ThemeDetailSerializer(root_themes, many=True)
        tree_data = serializer.data

        # Cache pour 30 minutes
        cache.set(cache_key, tree_data, 1800)
        return Response(tree_data)


class DocumentCategoryViewSet(BaseModelViewSet):
    """ViewSet pour la gestion des catégories de documents"""
    queryset = DocumentCategory.objects.all()
    search_fields = ['name', 'description']
    ordering_fields = ['name', 'category_type', 'created_at']
    ordering = ['name']

    serializer_classes = {
        'list': DocumentCategoryListSerializer,
        'retrieve': DocumentCategoryDetailSerializer,
    }
    serializer_class = DocumentCategoryDetailSerializer

    @action(detail=True, methods=['get'])
    def documents(self, request, pk=None):
        """Documents d'une catégorie"""
        category = self.get_object()
        documents = category.documents.filter(is_deleted=False)

        page = self.paginate_queryset(documents)
        if page is not None:
            serializer = DocumentListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = DocumentListSerializer(documents, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def stats(self, request, pk=None):
        """Statistiques d'une catégorie"""
        category = self.get_object()

        stats = {
            'total_documents': category.documents.filter(is_deleted=False).count(),
            'published_documents': category.documents.filter(
                is_deleted=False,
                status=Document.StatusChoices.PUBLISHED
            ).count(),
            'pending_documents': category.documents.filter(
                is_deleted=False,
                status=Document.StatusChoices.PENDING
            ).count(),
            'avg_file_size_mb': category.documents.filter(
                is_deleted=False
            ).aggregate(
                avg_size=models.Avg('file_size')
            )['avg_size'] or 0,
        }

        # Convertir en MB
        stats['avg_file_size_mb'] = round(stats['avg_file_size_mb'] / (1024 * 1024), 2)

        return Response(stats)


class DocumentViewSet(BaseModelViewSet):
    """ViewSet pour la gestion des documents"""
    queryset = Document.objects.select_related(
        'theme', 'category', 'uploaded_by', 'reviewed_by', 'approved_by', 'parent_document'
    ).prefetch_related('access_groups', 'authorized_users', 'related_documents', 'versions')

    filterset_class = DocumentProcessingFilter
    search_fields = ['title', 'description', 'original_filename', 'legal_reference']
    ordering_fields = ['title', 'created_at', 'updated_at', 'publication_date', 'view_count', 'rating_score']
    ordering = ['-created_at']

    serializer_classes = {
        'list': DocumentListSerializer,
        'retrieve': DocumentDetailSerializer,
        'create': DocumentCreateSerializer,
        'update': DocumentUpdateSerializer,
        'partial_update': DocumentUpdateSerializer,
    }
    serializer_class = DocumentDetailSerializer

    permission_classes = [permissions.IsAuthenticated, DocumentViewPermissions]

    def get_queryset(self):
        """Queryset avec gestion de la visibilité"""
        queryset = super().get_queryset()
        user = self.request.user

        # Handle schema generation by drf-yasg
        if getattr(self, 'swagger_fake_view', False):
            return queryset.none()

        if not user.is_authenticated:
            return queryset.none()

        if user.is_staff or user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
            return queryset

        # Filtrage par visibilité et autorisations
        return queryset.filter(
            models.Q(visibility=Document.VisibilityChoices.PUBLIC) |
            models.Q(uploaded_by=user) |
            models.Q(authorized_users=user) |
            models.Q(access_groups__user=user)
        ).distinct()

    def retrieve(self, request, *args, **kwargs):
        """Récupération avec incrémentation du compteur de vues"""
        instance = self.get_object()

        # Incrémenter le compteur de vues (éviter les double-comptes avec cache)
        view_cache_key = f"document_view_{instance.id}_{request.user.id}"
        if not cache.get(view_cache_key):
            Document.objects.filter(pk=instance.pk).update(view_count=models.F('view_count') + 1)
            cache.set(view_cache_key, True, 3600)  # 1 heure

        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    @action(detail=True, methods=['post'], permission_classes=[permissions.IsAuthenticated, CanManageDocument])
    def approve(self, request, pk=None):
        """Approuver un document"""
        document = self.get_object()

        if not request.user.has_perm('documents.can_approve_document'):
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        document.status = Document.StatusChoices.PUBLISHED
        document.approved_by = request.user
        document.approved_at = timezone.now()
        document.save(update_fields=['status', 'approved_by', 'approved_at'])

        # Notification asynchrone
        from .tasks import send_approval_notification
        if hasattr(send_approval_notification, 'delay'):
            send_approval_notification.delay(str(document.id), str(request.user.id))

        return Response({'message': 'Document approved successfully'})

    @action(detail=True, methods=['post'], permission_classes=[permissions.IsAuthenticated, CanManageDocument])
    def reject(self, request, pk=None):
        """Rejeter un document"""
        document = self.get_object()
        reason = request.data.get('reason', '')

        if not request.user.has_perm('documents.can_reject_document'):
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        document.status = Document.StatusChoices.REJECTED
        document.rejection_reason = reason
        document.reviewed_by = request.user
        document.reviewed_at = timezone.now()
        document.save(update_fields=['status', 'rejection_reason', 'reviewed_by', 'reviewed_at'])

        # Notification de rejet
        from .tasks import send_processing_notification
        if hasattr(send_processing_notification, 'delay'):
            send_processing_notification.delay(
                str(document.id),
                'error',
                f"Votre document a été rejeté. Raison: {reason}"
            )

        return Response({'message': 'Document rejected successfully'})

    @action(detail=True, methods=['get'])
    def download(self, request, pk=None):
        """Télécharger un document"""
        document = self.get_object()

        if not document.can_be_downloaded_by(request.user):
            return Response({'error': 'Download not allowed'}, status=status.HTTP_403_FORBIDDEN)

        # Incrémenter le compteur de téléchargements
        Document.objects.filter(pk=document.pk).update(download_count=models.F('download_count') + 1)

        # Retourner l'URL de téléchargement ou le fichier directement
        try:
            response = HttpResponse(document.file_path.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{document.original_filename}"'
            return response
        except Exception:
            return Response({'download_url': document.file_path.url})

    @action(detail=True, methods=['get'])
    def versions(self, request, pk=None):
        """Obtenir les versions d'un document"""
        document = self.get_object()
        versions = document.versions.all().order_by('-created_at')
        serializer = DocumentListSerializer(versions, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def related(self, request, pk=None):
        """Documents liés"""
        document = self.get_object()
        related = document.related_documents.filter(is_deleted=False)
        serializer = DocumentListSerializer(related, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def featured(self, request):
        """Documents en vedette"""
        cache_key = f"featured_documents_{request.user.id}"
        cached_docs = cache.get(cache_key)

        if cached_docs:
            return Response(cached_docs)

        featured_docs = self.get_queryset().filter(
            is_featured=True,
            status=Document.StatusChoices.PUBLISHED
        )[:10]

        serializer = DocumentListSerializer(featured_docs, many=True)
        data = serializer.data

        # Cache pour 15 minutes
        cache.set(cache_key, data, 900)
        return Response(data)

    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Documents récents"""
        recent_docs = self.get_queryset().filter(
            status=Document.StatusChoices.PUBLISHED
        ).order_by('-created_at')[:10]

        serializer = DocumentListSerializer(recent_docs, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def toggle_favorite(self, request, pk=None):
        """Basculer le statut favori d'un document"""
        document = self.get_object()
        user = request.user

        # Implémentation simple avec un champ metadata
        favorites = user.metadata.get('favorite_documents', [])
        doc_id = str(document.id)

        if doc_id in favorites:
            favorites.remove(doc_id)
            is_favorite = False
        else:
            favorites.append(doc_id)
            is_favorite = True

        user.metadata['favorite_documents'] = favorites
        user.save(update_fields=['metadata'])

        return Response({
            'is_favorite': is_favorite,
            'message': 'Document added to favorites' if is_favorite else 'Document removed from favorites'
        })

    @action(detail=False, methods=['get'])
    def favorites(self, request):
        """Documents favoris de l'utilisateur"""
        user = request.user
        favorite_ids = user.metadata.get('favorite_documents', [])

        if not favorite_ids:
            return Response({'results': [], 'count': 0})

        favorites = self.get_queryset().filter(id__in=favorite_ids)

        page = self.paginate_queryset(favorites)
        if page is not None:
            serializer = DocumentListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = DocumentListSerializer(favorites, many=True)
        return Response(serializer.data)


class DocumentContentViewSet(BaseModelViewSet):
    """ViewSet pour la gestion du contenu des documents"""
    queryset = DocumentContent.objects.select_related('document')
    filterset_class = DocumentContentFilter
    search_fields = ['clean_content', 'keywords_extracted']
    ordering_fields = ['word_count', 'page_count', 'extraction_confidence', 'processed_at']
    ordering = ['-created_at']

    serializer_class = DocumentContentSerializer
    permission_classes = [permissions.IsAuthenticated, DocumentViewPermissions]

    def get_queryset(self):
        """Queryset basé sur les permissions des documents"""
        queryset = super().get_queryset()
        user = self.request.user

        # Handle schema generation by drf-yasg
        if getattr(self, 'swagger_fake_view', False):
            return queryset.none()

        if not user.is_authenticated:
            return queryset.none()

        if user.is_staff or user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
            return queryset

        # Filtrer selon les documents accessibles
        return queryset.filter(
            models.Q(document__visibility=Document.VisibilityChoices.PUBLIC) |
            models.Q(document__uploaded_by=user) |
            models.Q(document__authorized_users=user)
        ).distinct()

    @action(detail=True, methods=['get'])
    def summary(self, request, pk=None):
        """Résumé du contenu"""
        content = self.get_object()
        max_words = int(request.query_params.get('max_words', 100))
        summary = content.get_summary(max_words=max_words)
        return Response({'summary': summary, 'word_count': len(summary.split())})

    @action(detail=True, methods=['get'])
    def keywords(self, request, pk=None):
        """Mots-clés extraits du contenu"""
        content = self.get_object()
        return Response({
            'keywords_extracted': content.keywords_extracted,
            'topics_detected': content.topics_detected,
            'entities_extracted': content.entities_extracted
        })

    @action(detail=False, methods=['get'])
    def search(self, request):
        """Recherche full-text dans le contenu"""
        query = request.query_params.get('q', '').strip()

        if not query or len(query) < 3:
            return Response(
                {'error': 'Query must be at least 3 characters long'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Recherche PostgreSQL full-text
        search_vector = SearchVector('clean_content', weight='A') + \
                        SearchVector('keywords_extracted', weight='B')
        search_query = SearchQuery(query, search_type='websearch')

        results = self.get_queryset().annotate(
            search=search_vector,
            rank=SearchRank(search_vector, search_query)
        ).filter(search=search_query).order_by('-rank')

        page = self.paginate_queryset(results)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(results, many=True)
        return Response(serializer.data)


class TopicViewSet(BaseModelViewSet):
    """ViewSet pour la gestion des topics"""
    queryset = Topic.objects.select_related('document', 'parent_topic').prefetch_related('subtopics')
    filterset_class = TopicFilter
    search_fields = ['title', 'content']
    ordering_fields = ['order_index', 'title', 'level', 'word_count']
    ordering = ['document', 'order_index']

    serializer_classes = {
        'list': TopicListSerializer,
        'retrieve': TopicDetailSerializer,
    }
    serializer_class = TopicDetailSerializer

    permission_classes = [permissions.IsAuthenticated, DocumentViewPermissions]

    def get_queryset(self):
        """Queryset filtré par document accessible"""
        queryset = super().get_queryset()
        user = self.request.user

        # Handle schema generation by drf-yasg
        if getattr(self, 'swagger_fake_view', False):
            return queryset.none()

        if not user.is_authenticated:
            return queryset.none()

        if user.is_staff or user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
            return queryset

        return queryset.filter(
            models.Q(document__visibility=Document.VisibilityChoices.PUBLIC) |
            models.Q(document__uploaded_by=user) |
            models.Q(document__authorized_users=user)
        ).distinct()

    @action(detail=True, methods=['get'])
    def subtopics(self, request, pk=None):
        """Sous-topics d'un topic"""
        topic = self.get_object()
        subtopics = topic.get_children()
        serializer = TopicListSerializer(subtopics, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def sections(self, request, pk=None):
        """Sections d'un topic"""
        topic = self.get_object()
        sections = topic.sections.filter(is_deleted=False).order_by('order_index')

        page = self.paginate_queryset(sections)
        if page is not None:
            serializer = SectionListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = SectionListSerializer(sections, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def tree(self, request, pk=None):
        """Arbre complet du topic avec ses sous-topics"""
        topic = self.get_object()

        def build_tree(topic_obj):
            data = TopicDetailSerializer(topic_obj).data
            children = topic_obj.get_children()
            if children:
                data['subtopics'] = [build_tree(child) for child in children]
            return data

        tree = build_tree(topic)
        return Response(tree)


class SectionViewSet(BaseModelViewSet):
    """ViewSet pour la gestion des sections"""
    queryset = Section.objects.select_related('topic').prefetch_related('paragraphs')
    filterset_class = SectionFilter
    search_fields = ['title', 'content']
    ordering_fields = ['order_index', 'title', 'word_count']
    ordering = ['topic', 'order_index']

    serializer_classes = {
        'list': SectionListSerializer,
        'retrieve': SectionDetailSerializer,
    }
    serializer_class = SectionDetailSerializer

    permission_classes = [permissions.IsAuthenticated, CanViewDocument]

    def get_queryset(self):
        """Queryset filtré par document accessible"""
        queryset = super().get_queryset()
        user = self.request.user

        # Handle schema generation by drf-yasg
        if getattr(self, 'swagger_fake_view', False):
            return queryset.none()

        if not user.is_authenticated:
            return queryset.none()

        if user.is_staff or user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
            return queryset

        return queryset.filter(
            models.Q(topic__document__visibility=Document.VisibilityChoices.PUBLIC) |
            models.Q(topic__document__uploaded_by=user) |
            models.Q(topic__document__authorized_users=user)
        ).distinct()

    @action(detail=True, methods=['get'])
    def paragraphs(self, request, pk=None):
        """Paragraphes d'une section"""
        section = self.get_object()
        paragraphs = section.paragraphs.all().order_by('order_index')
        serializer = ParagraphListSerializer(paragraphs, many=True)
        return Response(serializer.data)


class ParagraphViewSet(BaseModelViewSet):
    """ViewSet pour la gestion des paragraphes"""
    queryset = Paragraph.objects.select_related('section')
    filterset_class = ParagraphFilter
    search_fields = ['content']
    ordering_fields = ['order_index', 'word_count', 'sentence_count']
    ordering = ['section', 'order_index']

    serializer_classes = {
        'list': ParagraphListSerializer,
        'retrieve': ParagraphDetailSerializer,
    }
    serializer_class = ParagraphDetailSerializer

    permission_classes = [permissions.IsAuthenticated, CanViewDocument]

    def get_queryset(self):
        """Queryset filtré par document accessible"""
        queryset = super().get_queryset()
        user = self.request.user

        # Handle schema generation by drf-yasg
        if getattr(self, 'swagger_fake_view', False):
            return queryset.none()

        if not user.is_authenticated:
            return queryset.none()

        if user.is_staff or user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
            return queryset

        return queryset.filter(
            models.Q(section__topic__document__visibility=Document.VisibilityChoices.PUBLIC) |
            models.Q(section__topic__document__uploaded_by=user) |
            models.Q(section__topic__document__authorized_users=user)
        ).distinct()


class TableViewSet(BaseModelViewSet):
    """ViewSet pour la gestion des tableaux"""
    queryset = Table.objects.all()
    filterset_class = TableFilter
    search_fields = ['title', 'caption']
    ordering_fields = ['order_index', 'title', 'row_count', 'column_count', 'extraction_confidence']
    ordering = ['order_index']

    serializer_classes = {
        'list': TableListSerializer,
        'retrieve': TableDetailSerializer,
    }
    serializer_class = TableDetailSerializer

    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Queryset filtré par document accessible"""
        queryset = super().get_queryset()
        user = self.request.user

        # Handle schema generation by drf-yasg
        if getattr(self, 'swagger_fake_view', False):
            return queryset.none()

        if not user.is_authenticated:
            return queryset.none()

        if user.is_staff or user.role in [User.RoleChoices.ADMIN, User.RoleChoices.MODERATOR]:
            return queryset

        return queryset.filter(
            models.Q(section__topic__document__visibility=Document.VisibilityChoices.PUBLIC) |
            models.Q(section__topic__document__uploaded_by=user) |
            models.Q(section__topic__document__authorized_users=user)
        ).distinct()

    @action(detail=True, methods=['get'])
    def export_csv(self, request, pk=None):
        """Exporter un tableau en CSV"""
        table = self.get_object()

        return Response({
            'message': 'CSV export would be implemented here',
            'table_id': table.id,
            'headers': table.headers,
            'row_count': table.row_count
        })

    @action(detail=True, methods=['get'])
    def export_excel(self, request, pk=None):
        """Exporter un tableau en Excel"""
        table = self.get_object()

        return Response({
            'message': 'Excel export would be implemented here',
            'table_id': table.id
        })