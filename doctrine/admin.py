from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import (
    User, Theme, DocumentCategory, Document, DocumentContent,
    Topic, Section, Paragraph, Table
)


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['email', 'first_name', 'last_name', 'role', 'status', 'is_active', 'created_at']
    list_filter = ['role', 'status', 'is_active', 'department']
    search_fields = ['email', 'first_name', 'last_name', 'department']
    ordering = ['-created_at']

    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Informations personnelles', {'fields': ('first_name', 'last_name', 'phone_number')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Rôle et statut', {'fields': ('role', 'status', 'department', 'position')}),
        ('Dates importantes', {'fields': ('last_login', 'created_at')}),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'first_name', 'last_name', 'password1', 'password2'),
        }),
    )

    readonly_fields = ['created_at']


@admin.register(Theme)
class ThemeAdmin(admin.ModelAdmin):
    list_display = ['name', 'theme_type', 'parent_theme', 'level', 'is_active', 'documents_count']
    list_filter = ['theme_type', 'level', 'is_active', 'is_public']
    search_fields = ['name', 'description', 'code']
    ordering = ['level', 'order_index', 'name']
    readonly_fields = ['slug', 'level', 'path']


@admin.register(DocumentCategory)
class DocumentCategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'category_type', 'requires_approval', 'max_file_size_mb']
    list_filter = ['category_type', 'requires_approval']
    search_fields = ['name', 'description']
    readonly_fields = ['slug']


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'theme', 'category', 'uploaded_by', 'status', 'visibility', 'created_at']
    list_filter = ['status', 'visibility', 'language', 'theme', 'category', 'file_type']
    search_fields = ['title', 'description', 'original_filename']
    ordering = ['-created_at']
    readonly_fields = ['slug', 'file_size', 'file_checksum']

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('theme', 'category', 'uploaded_by')


@admin.register(DocumentContent)
class DocumentContentAdmin(admin.ModelAdmin):
    list_display = ['document', 'content_type', 'processing_status', 'word_count', 'extraction_confidence']
    list_filter = ['content_type', 'extraction_method', 'processing_status']
    search_fields = ['document__title']
    readonly_fields = ['word_count', 'character_count', 'processed_at']

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('document')


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ['title', 'document', 'topic_type', 'level', 'order_index', 'is_key_section']
    list_filter = ['topic_type', 'level', 'is_highlighted', 'is_key_section']
    search_fields = ['title', 'content']
    ordering = ['document', 'order_index']
    readonly_fields = ['slug', 'level']

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('document', 'parent_topic')


@admin.register(Section)
class SectionAdmin(admin.ModelAdmin):
    list_display = ['title', 'topic', 'section_type', 'order_index', 'word_count', 'is_highlighted']
    list_filter = ['section_type', 'is_highlighted', 'is_critical']
    search_fields = ['title', 'content']
    ordering = ['topic', 'order_index']
    readonly_fields = ['word_count', 'character_count']

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('topic')


@admin.register(Paragraph)
class ParagraphAdmin(admin.ModelAdmin):
    list_display = ['section', 'paragraph_type', 'order_index', 'word_count', 'is_key_paragraph']
    list_filter = ['paragraph_type', 'is_key_paragraph']
    search_fields = ['content']
    ordering = ['section', 'order_index']
    readonly_fields = ['word_count', 'sentence_count']

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('section')


@admin.register(Table)
class TableAdmin(admin.ModelAdmin):
    list_display = ['title', 'table_type', 'row_count', 'column_count', 'extraction_confidence', 'has_header_row']
    list_filter = ['table_type', 'has_header_row', 'is_complex']
    search_fields = ['title', 'caption']
    ordering = ['order_index']
    readonly_fields = ['row_count', 'column_count']