# signals.py
from django.db.models.signals import post_save, pre_delete, post_delete
from django.dispatch import receiver
from django.core.cache import cache
from .models import Document, DocumentContent, User, Theme
from .tasks import process_document_content, update_search_index


@receiver(post_save, sender=Document)
def document_post_save(sender, instance, created, **kwargs):
    '''
    Actions après sauvegarde d'un document
    '''
    if created:
        user = instance.uploaded_by
        user.current_storage_mb += instance.file_size_mb
        user.save(update_fields=['current_storage_mb'])

        if hasattr(process_document_content, 'delay'):
            process_document_content.delay(instance.id)

        cache.delete_pattern('stats:*')

    if hasattr(update_search_index, 'delay'):
        update_search_index.delay(instance.id)


@receiver(post_delete, sender=Document)
def document_post_delete(sender, instance, **kwargs):
    '''
    Actions après suppression d'un document
    '''
    user = instance.uploaded_by
    user.current_storage_mb = max(0, user.current_storage_mb - instance.file_size_mb)
    user.save(update_fields=['current_storage_mb'])

    if instance.file_path:
        instance.file_path.delete(save=False)

    cache.delete_pattern('stats:*')


@receiver(post_save, sender=Theme)
def theme_post_save(sender, instance, created, **kwargs):
    '''
    Actions après sauvegarde d'un thème
    '''
    if created or not kwargs.get('update_fields'):
        if instance.parent_theme:
            parent = instance.parent_theme
            parent.documents_count = parent.documents.filter(is_deleted=False).count()
            parent.save(update_fields=['documents_count'])


@receiver(post_save, sender=User)
def user_post_save(sender, instance, created, **kwargs):
    '''
    Actions après sauvegarde d'un utilisateur
    '''
    if created:
        if instance.api_access_enabled and not instance.api_key:
            instance.generate_api_key()

