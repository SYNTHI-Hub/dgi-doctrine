from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field

from django.contrib.auth import get_user_model
from .models import (
    User, Theme, DocumentCategory, Document, DocumentContent,
    Topic, Section, Paragraph, Table
)
from enum import Enum

class RAGMode(Enum):
    """Modes de traitement RAG"""
    SIMILARITY_ONLY = "similarity_only"
    HUGGINGFACE_RAG = "huggingface_rag"
    LANGCHAIN_RAG = "langchain_rag"
    HYBRID = "hybrid"

User = get_user_model()


class BaseModelSerializer(serializers.ModelSerializer):
    """Serializer de base avec gestion des timestamps"""
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)


class UserMinimalSerializer(serializers.ModelSerializer):
    """Serializer minimal pour les références utilisateur"""
    full_name = serializers.CharField(read_only=True)

    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'full_name']
        read_only_fields = ['id']


class UserListSerializer(BaseModelSerializer):
    """Serializer pour la liste des utilisateurs"""
    full_name = serializers.CharField(read_only=True)

    class Meta:
        model = User
        fields = [
            'id', 'email', 'first_name', 'last_name', 'full_name',
            'role', 'status', 'department', 'position', 'is_active',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class UserDetailSerializer(BaseModelSerializer):
    """Serializer détaillé pour un utilisateur"""
    full_name = serializers.CharField(read_only=True)
    manager = UserMinimalSerializer(read_only=True)
    team_members = UserMinimalSerializer(many=True, read_only=True)

    class Meta:
        model = User
        fields = [
            'id', 'email', 'first_name', 'last_name', 'full_name',
            'role', 'status', 'phone_number', 'birth_date', 'address',
            'city', 'country', 'postal_code', 'department', 'position',
            'employee_id', 'manager', 'team_members', 'hire_date',
            'salary_grade', 'profile_picture', 'preferred_language',
            'timezone', 'theme_preference', 'notifications_enabled',
            'email_notifications', 'email_verified', 'two_factor_enabled',
            'api_access_enabled', 'api_rate_limit', 'max_documents_upload',
            'storage_quota_mb', 'current_storage_mb', 'bio', 'skills',
            'interests', 'social_links', 'metadata', 'is_active',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'full_name', 'team_members', 'email_verified',
            'current_storage_mb', 'created_at', 'updated_at'
        ]
        extra_kwargs = {
            'password': {'write_only': True},
            'api_key': {'write_only': True},
            'two_factor_secret': {'write_only': True},
        }


class UserCreateSerializer(serializers.ModelSerializer):
    """Serializer pour la création d'utilisateur"""
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = [
            'email', 'first_name', 'last_name', 'password', 'password_confirm',
            'role', 'department', 'position', 'phone_number', 'manager'
        ]

    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError("Les mots de passe ne correspondent pas.")
        return attrs

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        password = validated_data.pop('password')
        user = User.objects.create_user(**validated_data)
        user.set_password(password)
        user.save()
        return user


class UserUpdateSerializer(serializers.ModelSerializer):
    """Serializer pour la mise à jour d'utilisateur"""

    class Meta:
        model = User
        fields = [
            'first_name', 'last_name', 'phone_number', 'birth_date',
            'address', 'city', 'country', 'postal_code', 'department',
            'position', 'manager', 'profile_picture', 'preferred_language',
            'timezone', 'theme_preference', 'notifications_enabled',
            'email_notifications', 'bio', 'skills', 'interests', 'social_links'
        ]


class ThemeMinimalSerializer(serializers.ModelSerializer):
    """Serializer minimal pour les références de thème"""

    class Meta:
        model = Theme
        fields = ['id', 'name', 'slug', 'color', 'icon']


class ThemeListSerializer(BaseModelSerializer):
    """Serializer pour la liste des thèmes"""
    parent_theme = ThemeMinimalSerializer(read_only=True)
    children_count = serializers.SerializerMethodField()

    class Meta:
        model = Theme
        fields = [
            'id', 'name', 'slug', 'description', 'code', 'theme_type',
            'parent_theme', 'icon', 'color', 'level', 'order_index',
            'is_active', 'is_featured', 'is_public', 'documents_count',
            'views_count', 'children_count', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'slug', 'level', 'path', 'created_at', 'updated_at']

    def get_children_count(self, obj):
        return obj.children.filter(is_active=True, is_deleted=False).count()


class ThemeDetailSerializer(BaseModelSerializer):
    """Serializer détaillé pour un thème"""
    parent_theme = ThemeMinimalSerializer(read_only=True)
    children = ThemeMinimalSerializer(many=True, read_only=True)
    ancestors = serializers.SerializerMethodField()
    created_by = UserMinimalSerializer(read_only=True)
    managed_by = UserMinimalSerializer(many=True, read_only=True)

    class Meta:
        model = Theme
        fields = [
            'id', 'name', 'slug', 'description', 'code', 'theme_type',
            'parent_theme', 'children', 'ancestors', 'icon', 'color',
            'background_color', 'text_color', 'order_index', 'level',
            'path', 'is_active', 'is_featured', 'is_public', 'metadata',
            'keywords', 'aliases', 'documents_count', 'views_count',
            'created_by', 'managed_by', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'slug', 'level', 'path', 'children', 'ancestors',
            'documents_count', 'views_count', 'created_at', 'updated_at'
        ]

    @extend_schema_field(ThemeMinimalSerializer(many=True))
    def get_ancestors(self, obj):
        return ThemeMinimalSerializer(obj.get_ancestors(), many=True).data


class ThemeCreateUpdateSerializer(BaseModelSerializer):
    """Serializer pour création/mise à jour de thème"""

    class Meta:
        model = Theme
        fields = [
            'name', 'description', 'theme_type', 'parent_theme',
            'icon', 'color', 'background_color', 'text_color',
            'order_index', 'is_active', 'is_featured', 'is_public',
            'metadata', 'keywords', 'aliases'
        ]

    def validate_parent_theme(self, value):
        if value and self.instance and value == self.instance:
            raise serializers.ValidationError("Un thème ne peut pas être son propre parent.")
        return value


class DocumentCategoryListSerializer(BaseModelSerializer):
    """Serializer pour la liste des catégories de documents"""

    class Meta:
        model = DocumentCategory
        fields = [
            'id', 'name', 'slug', 'description', 'category_type',
            'color', 'icon', 'requires_approval', 'default_visibility',
            'retention_days', 'max_file_size_mb', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'slug', 'created_at', 'updated_at']


class DocumentCategoryDetailSerializer(BaseModelSerializer):
    """Serializer détaillé pour une catégorie de document"""
    documents_count = serializers.SerializerMethodField()

    class Meta:
        model = DocumentCategory
        fields = [
            'id', 'name', 'slug', 'description', 'category_type',
            'color', 'icon', 'requires_approval', 'default_visibility',
            'retention_days', 'metadata_schema', 'allowed_file_types',
            'max_file_size_mb', 'documents_count', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'slug', 'documents_count', 'created_at', 'updated_at']

    def get_documents_count(self, obj):
        return obj.documents.filter(is_deleted=False).count()


class TableMinimalSerializer(BaseModelSerializer):
    """Serializer minimal pour les tableaux"""

    class Meta:
        model = Table
        fields = ['id', 'title', 'table_type', 'row_count', 'column_count']


class DocumentMinimalSerializer(serializers.ModelSerializer):
    """Serializer minimal pour les références de document"""
    file_size_mb = serializers.FloatField(read_only=True)

    class Meta:
        model = Document
        fields = [
            'id', 'title', 'slug', 'status', 'visibility',
            'file_size_mb', 'created_at'
        ]


class DocumentListSerializer(BaseModelSerializer):
    """Serializer pour la liste des documents"""
    theme = ThemeMinimalSerializer(read_only=True)
    category = serializers.StringRelatedField(read_only=True)
    uploaded_by = UserMinimalSerializer(read_only=True)
    file_size_mb = serializers.FloatField(read_only=True)
    is_expired = serializers.BooleanField(read_only=True)
    needs_review = serializers.BooleanField(read_only=True)

    class Meta:
        model = Document
        fields = [
            'id', 'title', 'slug', 'description', 'original_filename',
            'file_type', 'file_size_mb', 'theme', 'category', 'uploaded_by',
            'status', 'visibility', 'language', 'priority', 'publication_date',
            'effective_date', 'expiration_date', 'is_expired', 'needs_review',
            'view_count', 'download_count', 'rating_score', 'rating_count',
            'is_featured', 'version', 'is_latest_version', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'slug', 'file_size_mb', 'is_expired', 'needs_review',
            'view_count', 'download_count', 'rating_score', 'rating_count',
            'created_at', 'updated_at'
        ]


class DocumentDetailSerializer(BaseModelSerializer):
    """Serializer détaillé pour un document"""
    theme = ThemeDetailSerializer(read_only=True)
    category = DocumentCategoryListSerializer(read_only=True)
    uploaded_by = UserMinimalSerializer(read_only=True)
    reviewed_by = UserMinimalSerializer(read_only=True)
    approved_by = UserMinimalSerializer(read_only=True)
    access_groups = serializers.StringRelatedField(many=True, read_only=True)
    authorized_users = UserMinimalSerializer(many=True, read_only=True)
    related_documents = DocumentMinimalSerializer(many=True, read_only=True)
    parent_document = DocumentMinimalSerializer(read_only=True)
    versions = DocumentMinimalSerializer(many=True, read_only=True)
    file_size_mb = serializers.FloatField(read_only=True)
    is_expired = serializers.BooleanField(read_only=True)
    needs_review = serializers.BooleanField(read_only=True)

    class Meta:
        model = Document
        fields = [
            'id', 'title', 'slug', 'description', 'summary', 'abstract',
            'original_filename', 'file_path', 'file_type', 'file_size',
            'file_size_mb', 'file_checksum', 'mime_type', 'theme', 'category',
            'uploaded_by', 'status', 'reviewed_by', 'approved_by', 'reviewed_at',
            'approved_at', 'rejection_reason', 'visibility', 'security_classification',
            'access_groups', 'authorized_users', 'language', 'priority',
            'publication_date', 'effective_date', 'expiration_date', 'review_date',
            'is_expired', 'needs_review', 'legal_reference', 'regulation_number',
            'jurisdiction', 'legal_status', 'compliance_requirements',
            'view_count', 'download_count', 'share_count', 'comment_count',
            'rating_score', 'rating_count', 'metadata', 'extraction_metadata',
            'search_keywords', 'auto_generated_tags', 'is_featured',
            'is_searchable', 'is_downloadable', 'is_shareable', 'is_commentable',
            'is_ratable', 'requires_approval', 'allow_anonymous_access',
            'version', 'parent_document', 'versions', 'is_latest_version',
            'version_notes', 'related_documents', 'processing_log',
            'quality_score', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'slug', 'file_size', 'file_size_mb', 'file_checksum',
            'mime_type', 'uploaded_by', 'reviewed_by', 'approved_by',
            'reviewed_at', 'approved_at', 'is_expired', 'needs_review',
            'view_count', 'download_count', 'share_count', 'comment_count',
            'rating_score', 'rating_count', 'versions', 'processing_log',
            'created_at', 'updated_at'
        ]


class DocumentCreateSerializer(BaseModelSerializer):
    """Serializer pour la création de document"""

    class Meta:
        model = Document
        fields = [
            'title', 'description', 'summary', 'original_filename',
            'file_path', 'file_size', 'theme', 'category', 'visibility', 'language',
            'priority', 'publication_date', 'effective_date', 'expiration_date',
            'legal_reference', 'regulation_number', 'jurisdiction',
            'compliance_requirements', 'metadata', 'search_keywords',
            'is_featured', 'is_searchable', 'is_downloadable', 'is_shareable',
            'is_commentable', 'allow_anonymous_access', 'version_notes'
        ]

    def validate_file_path(self, value):
        if value:
            # Validation de la taille du fichier selon la catégorie
            max_size = getattr(self.initial_data.get('category'), 'max_file_size_mb', 50) * 1024 * 1024
            if value.size > max_size:
                raise serializers.ValidationError(
                    f"Le fichier est trop volumineux. Taille maximum: {max_size / 1024 / 1024}MB")
        return value

    def validate(self, attrs):
        # Ensure file_size is set if file_path is provided
        file_path = attrs.get('file_path')
        if file_path and not attrs.get('file_size'):
            file_size = getattr(file_path, 'size', None)
            if file_size is None or file_size <= 0:
                raise serializers.ValidationError({
                    'file_path': 'Impossible de déterminer la taille du fichier'
                })
            attrs['file_size'] = file_size
        return attrs

    def create(self, validated_data):
        # Set uploaded_by to current user
        validated_data['uploaded_by'] = self.context['request'].user
        return super().create(validated_data)


class DocumentUpdateSerializer(BaseModelSerializer):
    """Serializer pour la mise à jour de document"""

    class Meta:
        model = Document
        fields = [
            'title', 'description', 'summary', 'abstract', 'theme',
            'category', 'visibility', 'security_classification',
            'language', 'priority', 'publication_date', 'effective_date',
            'expiration_date', 'review_date', 'legal_reference',
            'regulation_number', 'jurisdiction', 'legal_status',
            'compliance_requirements', 'metadata', 'search_keywords',
            'is_featured', 'is_searchable', 'is_downloadable',
            'is_shareable', 'is_commentable', 'is_ratable',
            'allow_anonymous_access', 'version_notes'
        ]


class DocumentContentSerializer(BaseModelSerializer):
    """Serializer pour le contenu de document"""
    document = DocumentMinimalSerializer(read_only=True)
    total_pages = serializers.SerializerMethodField()
    has_page_based_content = serializers.SerializerMethodField()

    class Meta:
        model = DocumentContent
        fields = [
            'id', 'document', 'raw_content', 'structured_content',
            'html_content', 'markdown_content', 'clean_content',
            'content_type', 'extraction_method', 'processing_status',
            'extraction_confidence', 'word_count', 'character_count',
            'sentence_count', 'paragraph_count', 'page_count',
            'table_count', 'image_count', 'entities_extracted',
            'keywords_extracted', 'topics_detected', 'sentiment_score',
            'readability_score', 'complexity_score', 'processed_at',
            'processing_duration', 'processing_errors', 'quality_checks',
            'encoding_detected', 'language_detected', 'language_confidence',
            'total_pages', 'has_page_based_content', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'document', 'word_count', 'character_count',
            'sentence_count', 'paragraph_count', 'processed_at',
            'processing_duration', 'encoding_detected', 'language_detected',
            'language_confidence', 'total_pages', 'has_page_based_content',
            'created_at', 'updated_at'
        ]

    @extend_schema_field(serializers.IntegerField)
    def get_total_pages(self, obj):
        """Retourne le nombre total de pages"""
        return obj.get_total_pages()

    @extend_schema_field(serializers.BooleanField)
    def get_has_page_based_content(self, obj):
        """Indique si le contenu est organisé par pages"""
        return obj.has_page_based_content()


class PageContentSerializer(serializers.Serializer):
    """Serializer pour récupérer le contenu d'une ou plusieurs pages"""
    page_number = serializers.IntegerField(required=False, help_text="Numéro de page spécifique")
    start_page = serializers.IntegerField(required=False, help_text="Page de début (pour une plage)")
    end_page = serializers.IntegerField(required=False, help_text="Page de fin (pour une plage)")

    def validate(self, data):
        """Validation des paramètres de page"""
        page_number = data.get('page_number')
        start_page = data.get('start_page')
        end_page = data.get('end_page')

        if page_number and (start_page or end_page):
            raise serializers.ValidationError(
                "Utilisez soit 'page_number' soit 'start_page'/'end_page', pas les deux"
            )

        if not page_number and not start_page:
            raise serializers.ValidationError(
                "Spécifiez soit 'page_number' soit 'start_page'"
            )

        if start_page and start_page < 1:
            raise serializers.ValidationError("start_page doit être >= 1")

        if end_page and end_page < 1:
            raise serializers.ValidationError("end_page doit être >= 1")

        if start_page and end_page and start_page > end_page:
            raise serializers.ValidationError("start_page doit être <= end_page")

        return data


class PageContentResponseSerializer(serializers.Serializer):
    """Serializer pour la réponse de contenu par page"""
    document_id = serializers.UUIDField()
    document_title = serializers.CharField()
    total_pages = serializers.IntegerField()
    requested_pages = serializers.DictField(
        child=serializers.CharField(),
        help_text="Dictionnaire {page_number: page_content}"
    )


class DocumentContentLightSerializer(BaseModelSerializer):
    """Serializer allégé pour le contenu de document (pour pagination rapide)"""
    document = DocumentMinimalSerializer(read_only=True)

    class Meta:
        model = DocumentContent
        fields = [
            'id', 'document', 'content_type', 'extraction_method',
            'processing_status', 'word_count', 'character_count',
            'page_count', 'processed_at', 'created_at'
        ]
        read_only_fields = [
            'id', 'document', 'word_count', 'character_count',
            'processed_at', 'created_at'
        ]


class SectionPreviewSerializer(serializers.ModelSerializer):
    """Serializer pour l'aperçu des sections - uniquement title et content"""

    class Meta:
        model = Section
        fields = ['title', 'content']


class DocumentContentWithPreviewSerializer(BaseModelSerializer):
    """Serializer pour le contenu de document avec aperçu des premières sections"""
    document = DocumentMinimalSerializer(read_only=True)
    content_preview = serializers.SerializerMethodField()

    class Meta:
        model = DocumentContent
        fields = [
            'id', 'document', 'content_type', 'extraction_method',
            'processing_status', 'word_count', 'character_count',
            'page_count', 'processed_at', 'created_at', 'content_preview'
        ]
        read_only_fields = [
            'id', 'document', 'word_count', 'character_count',
            'processed_at', 'created_at'
        ]

    @extend_schema_field({
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'title': {'type': 'string', 'description': 'Titre de la section'},
                'content': {'type': 'string', 'description': 'Contenu de la section'}
            }
        },
        'description': 'Aperçu des 3 premières sections du document (title et content uniquement)'
    })
    def get_content_preview(self, obj):
        """Retourne les 3 premières sections du document pour aperçu (title et content uniquement)"""
        try:
            # Récupérer les 3 premières sections du document - optimisé sans paragraphes
            sections = Section.objects.filter(
                topic__document=obj.document,
                is_deleted=False
            ).select_related('topic').only(
                'title', 'content', 'topic__order_index', 'order_index'
            ).order_by(
                'topic__order_index', 'order_index'
            )[:3]

            return SectionPreviewSerializer(sections, many=True).data
        except Exception:
            return []


class TopicMinimalSerializer(serializers.ModelSerializer):
    """Serializer minimal pour les topics"""

    class Meta:
        model = Topic
        fields = ['id', 'title', 'slug', 'topic_type', 'level', 'order_index']


class TopicListSerializer(BaseModelSerializer):
    """Serializer pour la liste des topics"""
    document = DocumentMinimalSerializer(read_only=True)
    parent_topic = TopicMinimalSerializer(read_only=True)
    subtopics_count = serializers.SerializerMethodField()

    class Meta:
        model = Topic
        fields = [
            'id', 'document', 'parent_topic', 'title', 'slug', 'summary',
            'topic_type', 'order_index', 'level', 'numbering', 'start_page',
            'end_page', 'word_count', 'reading_time_minutes', 'importance_score',
            'is_highlighted', 'is_key_section', 'requires_attention',
            'subtopics_count', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'slug', 'level', 'word_count', 'reading_time_minutes',
            'subtopics_count', 'created_at', 'updated_at'
        ]

    def get_subtopics_count(self, obj):
        return obj.subtopics.filter(is_deleted=False).count()


class TopicDetailSerializer(BaseModelSerializer):
    """Serializer détaillé pour un topic"""
    document = DocumentMinimalSerializer(read_only=True)
    parent_topic = TopicMinimalSerializer(read_only=True)
    subtopics = TopicMinimalSerializer(many=True, read_only=True)

    class Meta:
        model = Topic
        fields = [
            'id', 'document', 'parent_topic', 'subtopics', 'title', 'slug',
            'content', 'summary', 'topic_type', 'order_index', 'level',
            'numbering', 'start_page', 'end_page', 'start_position',
            'end_position', 'metadata', 'keywords', 'entities', 'concepts',
            'word_count', 'reading_time_minutes', 'importance_score',
            'is_highlighted', 'is_key_section', 'requires_attention',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'slug', 'level', 'subtopics', 'word_count',
            'reading_time_minutes', 'created_at', 'updated_at'
        ]


class SectionMinimalSerializer(serializers.ModelSerializer):
    """Serializer minimal pour les sections"""

    class Meta:
        model = Section
        fields = ['id', 'title', 'section_type', 'order_index']


class SectionListSerializer(BaseModelSerializer):
    """Serializer pour la liste des sections"""
    topic = TopicMinimalSerializer(read_only=True)

    class Meta:
        model = Section
        fields = [
            'id', 'topic', 'title', 'subtitle', 'section_type',
            'order_index', 'alignment', 'start_page', 'end_page',
            'word_count', 'character_count', 'is_highlighted',
            'is_critical', 'requires_translation', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'word_count', 'character_count', 'created_at', 'updated_at'
        ]


class SectionSerializer(BaseModelSerializer):
    """Serializer détaillé pour une section"""
    topic = TopicMinimalSerializer(read_only=True)

    class Meta:
        model = Section
        fields = [
            'id', 'topic', 'title', 'subtitle', 'content', 'raw_content',
            'section_type', 'order_index', 'formatting', 'css_classes',
            'alignment', 'font_size', 'font_weight', 'start_page', 'end_page',
            'bbox_coordinates', 'word_count', 'character_count',
            'is_highlighted', 'is_critical', 'requires_translation',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'word_count', 'character_count', 'created_at', 'updated_at'
        ]


class ParagraphListSerializer(BaseModelSerializer):
    """Serializer pour la liste des paragraphes"""
    section = SectionMinimalSerializer(read_only=True)

    class Meta:
        model = Paragraph
        fields = [
            'id', 'section', 'paragraph_type', 'order_index', 'alignment',
            'indentation', 'word_count', 'sentence_count', 'readability_score',
            'is_key_paragraph', 'requires_review', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'word_count', 'sentence_count', 'readability_score',
            'created_at', 'updated_at'
        ]


class ParagraphSerializer(BaseModelSerializer):
    """Serializer détaillé pour un paragraphe"""
    section = SectionMinimalSerializer(read_only=True)

    class Meta:
        model = Paragraph
        fields = [
            'id', 'section', 'content', 'original_content', 'paragraph_type',
            'order_index', 'formatting', 'alignment', 'indentation',
            'entities', 'keywords', 'concepts', 'named_entities',
            'word_count', 'sentence_count', 'readability_score',
            'is_key_paragraph', 'requires_review', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'word_count', 'sentence_count', 'readability_score',
            'created_at', 'updated_at'
        ]


class TableListSerializer(BaseModelSerializer):
    """Serializer pour la liste des tableaux"""

    class Meta:
        model = Table
        fields = [
            'id', 'title', 'caption', 'table_type', 'row_count',
            'column_count', 'order_index', 'extraction_confidence',
            'extraction_method', 'has_header_row', 'has_totals_row',
            'is_transposed', 'is_complex', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'row_count', 'column_count', 'extraction_confidence',
            'created_at', 'updated_at'
        ]


class TableSerializer(BaseModelSerializer):
    """Serializer détaillé pour un tableau"""

    class Meta:
        model = Table
        fields = [
            'id', 'title', 'caption', 'table_type', 'headers', 'data',
            'raw_data', 'processed_data', 'row_count', 'column_count',
            'order_index', 'extraction_confidence', 'extraction_method',
            'bbox_coordinates', 'styling', 'borders', 'column_widths',
            'has_header_row', 'has_totals_row', 'is_transposed', 'is_complex',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'row_count', 'column_count', 'extraction_confidence',
            'created_at', 'updated_at'
        ]


# ==========================================
# RAG Serializers
# ==========================================

class RAGQuerySerializer(serializers.Serializer):
    """Serializer pour les requêtes RAG"""
    query = serializers.CharField(
        max_length=1000,
        required=True,
        help_text="La question ou requête à traiter"
    )
    mode = serializers.ChoiceField(
        choices=[(mode.value, mode.value) for mode in RAGMode],
        default=RAGMode.SIMILARITY_ONLY.value,
        required=False,
        help_text="Mode de traitement RAG"
    )
    k = serializers.IntegerField(
        min_value=1,
        max_value=20,
        default=5,
        required=False,
        help_text="Nombre de documents à récupérer"
    )
    scope = serializers.ChoiceField(
        choices=[('all', 'Tous les documents'), ('document', 'Document spécifique')],
        default='all',
        required=False,
        help_text="Portée de la recherche"
    )
    document_id = serializers.UUIDField(
        required=False,
        allow_null=True,
        help_text="ID du document spécifique (si scope='document')"
    )
    max_new_tokens = serializers.IntegerField(
        min_value=10,
        max_value=500,
        default=100,
        required=False,
        help_text="Nombre maximum de tokens à générer (pour mode generation)"
    )
    num_beams = serializers.IntegerField(
        min_value=1,
        max_value=10,
        default=4,
        required=False,
        help_text="Nombre de beams pour la génération"
    )
    temperature = serializers.FloatField(
        min_value=0.1,
        max_value=2.0,
        default=1.0,
        required=False,
        help_text="Température pour la génération (créativité)"
    )
    use_custom_context = serializers.BooleanField(
        default=True,
        required=False,
        help_text="Utiliser la base de données comme contexte"
    )

    def validate(self, attrs):
        """Validation inter-champs"""
        scope = attrs.get('scope', 'all')
        document_id = attrs.get('document_id')

        if scope == 'document' and not document_id:
            raise serializers.ValidationError(
                "document_id est requis quand scope='document'"
            )

        return attrs


class HuggingFaceRAGGenerationSerializer(serializers.Serializer):
    """Serializer pour la génération Hugging Face RAG complète"""
    query = serializers.CharField(
        max_length=1000,
        required=True,
        help_text="La question à traiter"
    )
    max_new_tokens = serializers.IntegerField(
        min_value=10,
        max_value=500,
        default=100,
        required=False
    )
    num_beams = serializers.IntegerField(
        min_value=1,
        max_value=10,
        default=4,
        required=False
    )
    do_sample = serializers.BooleanField(
        default=False,
        required=False,
        help_text="Utiliser l'échantillonnage au lieu de beam search"
    )
    temperature = serializers.FloatField(
        min_value=0.1,
        max_value=2.0,
        default=1.0,
        required=False
    )
    top_p = serializers.FloatField(
        min_value=0.1,
        max_value=1.0,
        default=1.0,
        required=False,
        help_text="Nucleus sampling"
    )
    early_stopping = serializers.BooleanField(
        default=True,
        required=False
    )
    use_custom_context = serializers.BooleanField(
        default=True,
        required=False,
        help_text="Utiliser notre base de données pour le contexte"
    )


class ChatMessage(serializers.Serializer):
    """Serializer pour un message de chat"""
    role = serializers.ChoiceField(
        choices=[('user', 'User'), ('assistant', 'Assistant'), ('system', 'System')],
        required=True,
        help_text="Rôle du message"
    )
    content = serializers.CharField(
        max_length=2000,
        required=True,
        help_text="Contenu du message"
    )


class ChatCompletionSerializer(serializers.Serializer):
    """Serializer pour les conversations chat (compatible OpenAI)"""
    messages = serializers.ListSerializer(
        child=ChatMessage(),
        min_length=1,
        help_text="Liste des messages de conversation"
    )
    max_new_tokens = serializers.IntegerField(
        min_value=10,
        max_value=500,
        default=100,
        required=False
    )
    temperature = serializers.FloatField(
        min_value=0.1,
        max_value=2.0,
        default=1.0,
        required=False
    )
    top_p = serializers.FloatField(
        min_value=0.1,
        max_value=1.0,
        default=1.0,
        required=False
    )
    num_beams = serializers.IntegerField(
        min_value=1,
        max_value=10,
        default=4,
        required=False
    )

    def validate_messages(self, value):
        """Valider qu'il y a au moins un message utilisateur"""
        user_messages = [msg for msg in value if msg.get('role') == 'user']
        if not user_messages:
            raise serializers.ValidationError(
                "Au moins un message avec role='user' est requis"
            )
        return value


class RAGResponseSerializer(serializers.Serializer):
    """Serializer pour les réponses RAG (en lecture seule)"""
    query = serializers.CharField(read_only=True)
    mode = serializers.CharField(read_only=True)
    scope = serializers.CharField(read_only=True)
    k = serializers.IntegerField(read_only=True)
    count = serializers.IntegerField(read_only=True)
    took_ms = serializers.IntegerField(read_only=True)
    results = serializers.ListField(read_only=True)
    generated_response = serializers.CharField(read_only=True, required=False)
    generation_metadata = serializers.DictField(read_only=True, required=False)
    error = serializers.CharField(read_only=True, required=False)


class ModelInfoSerializer(serializers.Serializer):
    """Serializer pour les informations sur les modèles RAG"""
    model_name = serializers.CharField(read_only=True)
    retriever_name = serializers.CharField(read_only=True)
    device = serializers.CharField(read_only=True)
    model_loaded = serializers.BooleanField(read_only=True)
    use_custom_retriever = serializers.BooleanField(read_only=True)
    use_gpu = serializers.BooleanField(read_only=True)
    dependencies_available = serializers.BooleanField(read_only=True)