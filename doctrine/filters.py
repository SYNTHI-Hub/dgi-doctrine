import django_filters
from django import forms
from django_filters import rest_framework as filters
from django.db import models
from django.utils import timezone
from datetime import timedelta

from packaging.utils import _

from .models import (
    Document, DocumentContent, Topic, Section,
    Paragraph, Table, Theme, DocumentCategory, User
)


class DocumentProcessingFilter(django_filters.FilterSet):
    """Filtres avancés pour le traitement des documents"""

    # Filtres de base
    status = filters.ChoiceFilter(choices=Document.StatusChoices.choices)
    file_type = filters.CharFilter(field_name='file_type', lookup_expr='iexact')
    language = filters.ChoiceFilter(choices=Document.LanguageChoices.choices)
    visibility = filters.ChoiceFilter(choices=Document.VisibilityChoices.choices)

    # Filtres par relations
    theme = filters.ModelChoiceFilter(queryset=Theme.objects.filter(is_active=True, is_deleted=False))
    theme_name = filters.CharFilter(field_name='theme__name', lookup_expr='icontains')
    category = filters.ModelChoiceFilter(queryset=DocumentCategory.objects.all())
    category_type = filters.ChoiceFilter(field_name='category__category_type',
                                         choices=DocumentCategory.CategoryType.choices)
    uploaded_by = filters.ModelChoiceFilter(queryset=User.objects.filter(is_active=True))

    # Filtres de dates
    created_after = filters.DateTimeFilter(field_name='created_at', lookup_expr='gte')
    created_before = filters.DateTimeFilter(field_name='created_at', lookup_expr='lte')
    updated_after = filters.DateTimeFilter(field_name='updated_at', lookup_expr='gte')
    updated_before = filters.DateTimeFilter(field_name='updated_at', lookup_expr='lte')

    # Filtres de période prédéfinie
    created_period = filters.ChoiceFilter(
        method='filter_created_period',
        choices=[
            ('today', 'Aujourd\'hui'),
            ('week', 'Cette semaine'),
            ('month', 'Ce mois'),
            ('quarter', 'Ce trimestre'),
            ('year', 'Cette année')
        ]
    )

    # Filtres de taille
    file_size_min = filters.NumberFilter(field_name='file_size', lookup_expr='gte')
    file_size_max = filters.NumberFilter(field_name='file_size', lookup_expr='lte')
    file_size_mb_min = filters.NumberFilter(method='filter_file_size_mb_min')
    file_size_mb_max = filters.NumberFilter(method='filter_file_size_mb_max')

    # Filtres booléens
    has_content = filters.BooleanFilter(method='filter_has_content')
    is_processed = filters.BooleanFilter(method='filter_is_processed')
    has_errors = filters.BooleanFilter(method='filter_has_errors')
    is_featured = filters.BooleanFilter()

    # Filtres de contenu
    has_tables = filters.BooleanFilter(method='filter_has_tables')
    min_word_count = filters.NumberFilter(method='filter_min_word_count')
    max_word_count = filters.NumberFilter(method='filter_max_word_count')
    min_page_count = filters.NumberFilter(method='filter_min_page_count')
    max_page_count = filters.NumberFilter(method='filter_max_page_count')

    # Filtres de qualité
    min_extraction_confidence = filters.NumberFilter(method='filter_min_extraction_confidence')
    extraction_method = filters.ChoiceFilter(
        field_name='content__extraction_method',
        choices=DocumentContent.ExtractionMethod.choices
    )

    class Meta:
        model = Document
        fields = [
            'status', 'file_type', 'language', 'visibility', 'theme', 'category',
            'uploaded_by', 'is_featured', 'is_searchable', 'is_downloadable'
        ]

    def filter_created_period(self, queryset, name, value):
        """Filtre par période de création"""
        now = timezone.now()

        if value == 'today':
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif value == 'week':
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif value == 'month':
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif value == 'quarter':
            quarter_start_month = ((now.month - 1) // 3) * 3 + 1
            start = now.replace(month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif value == 'year':
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return queryset

        return queryset.filter(created_at__gte=start)

    def filter_file_size_mb_min(self, queryset, name, value):
        """Filtre par taille minimale en MB"""
        return queryset.filter(file_size__gte=value * 1024 * 1024)

    def filter_file_size_mb_max(self, queryset, name, value):
        """Filtre par taille maximale en MB"""
        return queryset.filter(file_size__lte=value * 1024 * 1024)

    def filter_has_content(self, queryset, name, value):
        """Filtre les documents qui ont du contenu extrait"""
        if value:
            return queryset.filter(content__isnull=False)
        else:
            return queryset.filter(content__isnull=True)

    def filter_is_processed(self, queryset, name, value):
        """Filtre les documents traités"""
        if value:
            return queryset.filter(status=Document.StatusChoices.PROCESSED)
        else:
            return queryset.exclude(status=Document.StatusChoices.PROCESSED)

    def filter_has_errors(self, queryset, name, value):
        """Filtre les documents avec erreurs"""
        if value:
            return queryset.filter(status=Document.StatusChoices.ERROR)
        else:
            return queryset.exclude(status=Document.StatusChoices.ERROR)

    def filter_has_tables(self, queryset, name, value):
        """Filtre les documents qui contiennent des tableaux"""
        if value:
            return queryset.filter(content__table_count__gt=0)
        else:
            return queryset.filter(
                models.Q(content__table_count=0) |
                models.Q(content__isnull=True)
            )

    def filter_min_word_count(self, queryset, name, value):
        """Filtre par nombre minimum de mots"""
        return queryset.filter(content__word_count__gte=value)

    def filter_max_word_count(self, queryset, name, value):
        """Filtre par nombre maximum de mots"""
        return queryset.filter(content__word_count__lte=value)

    def filter_min_page_count(self, queryset, name, value):
        """Filtre par nombre minimum de pages"""
        return queryset.filter(content__page_count__gte=value)

    def filter_max_page_count(self, queryset, name, value):
        """Filtre par nombre maximum de pages"""
        return queryset.filter(content__page_count__lte=value)

    def filter_min_extraction_confidence(self, queryset, name, value):
        """Filtre par confiance d'extraction minimale"""
        return queryset.filter(content__extraction_confidence__gte=value)


class DocumentContentFilter(django_filters.FilterSet):
    """Filtres pour le contenu des documents"""

    # Filtres de base
    content_type = filters.ChoiceFilter(choices=DocumentContent.ContentType.choices)
    extraction_method = filters.ChoiceFilter(choices=DocumentContent.ExtractionMethod.choices)
    processing_status = filters.ChoiceFilter(choices=DocumentContent.ProcessingStatus.choices)

    # Filtres numériques
    word_count_min = filters.NumberFilter(field_name='word_count', lookup_expr='gte')
    word_count_max = filters.NumberFilter(field_name='word_count', lookup_expr='lte')
    page_count_min = filters.NumberFilter(field_name='page_count', lookup_expr='gte')
    page_count_max = filters.NumberFilter(field_name='page_count', lookup_expr='lte')
    table_count_min = filters.NumberFilter(field_name='table_count', lookup_expr='gte')
    table_count_max = filters.NumberFilter(field_name='table_count', lookup_expr='lte')

    # Filtres de qualité
    extraction_confidence_min = filters.NumberFilter(field_name='extraction_confidence', lookup_expr='gte')
    extraction_confidence_max = filters.NumberFilter(field_name='extraction_confidence', lookup_expr='lte')
    readability_score_min = filters.NumberFilter(field_name='readability_score', lookup_expr='gte')
    readability_score_max = filters.NumberFilter(field_name='readability_score', lookup_expr='lte')

    # Filtres de document parent
    document_status = filters.ChoiceFilter(field_name='document__status',
                                           choices=Document.StatusChoices.choices)
    document_language = filters.ChoiceFilter(field_name='document__language',
                                             choices=Document.LanguageChoices.choices)

    # Filtres de dates de traitement
    processed_after = filters.DateTimeFilter(field_name='processed_at', lookup_expr='gte')
    processed_before = filters.DateTimeFilter(field_name='processed_at', lookup_expr='lte')

    class Meta:
        model = DocumentContent
        fields = [
            'content_type', 'extraction_method', 'processing_status',
            'language_detected', 'encoding_detected'
        ]


class TopicFilter(django_filters.FilterSet):
    """Filtres pour les topics"""

    # Filtres de base
    topic_type = filters.ChoiceFilter(choices=Topic.TopicType.choices)
    level = filters.NumberFilter()
    level_min = filters.NumberFilter(field_name='level', lookup_expr='gte')
    level_max = filters.NumberFilter(field_name='level', lookup_expr='lte')

    # Filtres de document
    document = filters.ModelChoiceFilter(queryset=Document.objects.all())
    document_status = filters.ChoiceFilter(field_name='document__status',
                                           choices=Document.StatusChoices.choices)

    # Filtres de contenu
    word_count_min = filters.NumberFilter(field_name='word_count', lookup_expr='gte')
    word_count_max = filters.NumberFilter(field_name='word_count', lookup_expr='lte')
    importance_score_min = filters.NumberFilter(field_name='importance_score', lookup_expr='gte')

    # Filtres booléens
    is_highlighted = filters.BooleanFilter()
    is_key_section = filters.BooleanFilter()
    requires_attention = filters.BooleanFilter()

    # Filtres hiérarchiques
    has_subtopics = filters.BooleanFilter(method='filter_has_subtopics')
    is_root_topic = filters.BooleanFilter(method='filter_is_root_topic')

    class Meta:
        model = Topic
        fields = [
            'topic_type', 'level', 'is_highlighted', 'is_key_section',
            'requires_attention', 'start_page', 'end_page'
        ]

    def filter_has_subtopics(self, queryset, name, value):
        """Filtre les topics qui ont des sous-topics"""
        if value:
            return queryset.filter(subtopics__isnull=False).distinct()
        else:
            return queryset.filter(subtopics__isnull=True)

    def filter_is_root_topic(self, queryset, name, value):
        """Filtre les topics racines (sans parent)"""
        if value:
            return queryset.filter(parent_topic__isnull=True)
        else:
            return queryset.filter(parent_topic__isnull=False)


class TableFilter(django_filters.FilterSet):
    """Filtres pour les tableaux"""

    # Filtres de base
    table_type = filters.ChoiceFilter(choices=Table.TableType.choices)

    # Filtres de taille
    row_count_min = filters.NumberFilter(field_name='row_count', lookup_expr='gte')
    row_count_max = filters.NumberFilter(field_name='row_count', lookup_expr='lte')
    column_count_min = filters.NumberFilter(field_name='column_count', lookup_expr='gte')
    column_count_max = filters.NumberFilter(field_name='column_count', lookup_expr='lte')

    # Filtres de qualité
    extraction_confidence_min = filters.NumberFilter(field_name='extraction_confidence', lookup_expr='gte')
    extraction_method = filters.CharFilter(field_name='extraction_method', lookup_expr='icontains')

    # Filtres booléens
    has_header_row = filters.BooleanFilter()
    has_totals_row = filters.BooleanFilter()
    is_transposed = filters.BooleanFilter()
    is_complex = filters.BooleanFilter()

    class Meta:
        model = Table
        fields = [
            'table_type', 'has_header_row', 'has_totals_row',
            'is_transposed', 'is_complex', 'order_index'
        ]


class SectionFilter(django_filters.FilterSet):
    """Filtres minimalistes pour les sections"""

    topic = django_filters.ModelChoiceFilter(
        queryset=Topic.objects.filter(is_deleted=False),
        label=_("Topic"),
        widget=forms.Select(attrs={'class': 'form-control'}),
        empty_label=_("Tous")
    )

    section_type = django_filters.ChoiceFilter(
        choices=Section.SectionType.choices,
        label=_("Type de section"),
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    is_highlighted = django_filters.BooleanFilter(
        label=_("Mis en évidence"),
        widget=forms.CheckboxInput()
    )

    is_critical = django_filters.BooleanFilter(
        label=_("Critique"),
        widget=forms.CheckboxInput()
    )

    class Meta:
        model = Section
        fields = ['topic', 'section_type', 'is_highlighted', 'is_critical']

    ordering = django_filters.OrderingFilter(
        fields=(
            ('order_index', 'order_index'),
            ('title', 'title'),
            ('word_count', 'word_count'),
        ),
        field_labels={
            'order_index': _('Ordre'),
            'title': _('Titre'),
            'word_count': _('Nombre de mots'),
        }
    )

    def filter_search(self, queryset, name, value):
        return queryset.filter(
            models.Q(title__icontains=value) |
            models.Q(content__icontains=value)
        )


class ParagraphFilter(django_filters.FilterSet):
    """Filtres minimalistes pour les paragraphes"""

    section = django_filters.ModelChoiceFilter(
        queryset=Section.objects.filter(is_deleted=False),
        label=_("Section"),
        widget=forms.Select(attrs={'class': 'form-control'}),
        empty_label=_("Toutes")
    )

    paragraph_type = django_filters.ChoiceFilter(
        choices=Paragraph.ParagraphType.choices,
        label=_("Type de paragraphe"),
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    is_key_paragraph = django_filters.BooleanFilter(
        label=_("Paragraphe clé"),
        widget=forms.CheckboxInput()
    )

    class Meta:
        model = Paragraph
        fields = ['section', 'paragraph_type', 'is_key_paragraph']

    ordering = django_filters.OrderingFilter(
        fields=(
            ('order_index', 'order_index'),
            ('word_count', 'word_count'),
            ('sentence_count', 'sentence_count'),
        ),
        field_labels={
            'order_index': _('Ordre'),
            'word_count': _('Nombre de mots'),
            'sentence_count': _('Nombre de phrases'),
        }
    )

    def filter_search(self, queryset, name, value):
        return queryset.filter(content__icontains=value)


class TableFilter(django_filters.FilterSet):
    """Filtres minimalistes pour les tableaux"""

    table_type = django_filters.ChoiceFilter(
        choices=Table.TableType.choices,
        label=_("Type de tableau"),
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    extraction_method = django_filters.CharFilter(
        lookup_expr='icontains',
        label=_("Méthode d'extraction"),
        widget=forms.TextInput(attrs={'placeholder': 'Méthode...'})
    )

    has_header_row = django_filters.BooleanFilter(
        label=_("Ligne d'en-tête"),
        widget=forms.CheckboxInput()
    )

    is_complex = django_filters.BooleanFilter(
        label=_("Complexe"),
        widget=forms.CheckboxInput()
    )

    class Meta:
        model = Table
        fields = ['table_type', 'extraction_method', 'has_header_row', 'is_complex']

    ordering = django_filters.OrderingFilter(
        fields=(
            ('order_index', 'order_index'),
            ('title', 'title'),
            ('row_count', 'row_count'),
            ('column_count', 'column_count'),
            ('extraction_confidence', 'extraction_confidence'),
        ),
        field_labels={
            'order_index': _('Ordre'),
            'title': _('Titre'),
            'row_count': _('Nombre de lignes'),
            'column_count': _('Nombre de colonnes'),
            'extraction_confidence': _('Confiance'),
        }
    )

    def filter_search(self, queryset, name, value):
        return queryset.filter(
            models.Q(title__icontains=value) |
            models.Q(caption__icontains=value)
        )