from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from django.db import transaction
from django.core.cache import cache
from datetime import datetime, timedelta
import logging
import os

from doctrine import models

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2, default_retry_delay=120, soft_time_limit=3600, time_limit=7200)
def process_document_content(self, document_id):
    """
    Traite le contenu d'un document en arrière-plan avec retry automatique
    """
    from .models import Document, DocumentContent
    from .services.document_processor import document_processor

    try:
        with transaction.atomic():
            document = Document.objects.select_for_update().get(id=document_id)

            if document.status == Document.StatusChoices.PROCESSING:
                logger.warning(f"Document {document_id} déjà en cours de traitement")
                return {"status": "already_processing", "document_id": document_id}

            document.status = Document.StatusChoices.PROCESSING
            document.processing_log.append({
                'timestamp': timezone.now().isoformat(),
                'action': 'processing_started',
                'task_id': self.request.id,
                'retry_count': self.request.retries
            })
            document.save(update_fields=['status', 'processing_log'])

        # Traitement du document
        success = document_processor.process_document(document)

        if success:
            logger.info(f"Document {document_id} traité avec succès")

            update_search_index.delay(document_id)

            if document.uploaded_by.email_notifications:
                send_processing_notification.delay(
                    document_id,
                    'success',
                    f"Votre document '{document.title}' a été traité avec succès."
                )

            # Invalider les caches liés
            cache_keys = [
                f'document_stats_{document.uploaded_by.id}',
                f'document_content_{document_id}',
                'global_processing_stats'
            ]
            cache.delete_many(cache_keys)

            return {
                "status": "success",
                "document_id": document_id,
                "processing_time": document.processing_log[-1].get('processing_time')
            }
        else:
            raise Exception("Échec du traitement du document")

    except Document.DoesNotExist:
        logger.error(f"Document {document_id} introuvable")
        return {"status": "error", "message": "Document introuvable", "document_id": document_id}

    except Exception as e:
        logger.error(f"Erreur lors du traitement du document {document_id}: {str(e)}")

        # Retry automatique
        if self.request.retries < self.max_retries:
            logger.info(f"Retry {self.request.retries + 1}/{self.max_retries} pour le document {document_id}")
            raise self.retry(countdown=60 * (2 ** self.request.retries))

        # Échec définitif après tous les retries
        try:
            document = Document.objects.get(id=document_id)
            document.status = Document.StatusChoices.ERROR
            document.processing_log.append({
                'timestamp': timezone.now().isoformat(),
                'action': 'processing_failed',
                'error': str(e),
                'final_failure': True,
                'retries_exhausted': True
            })
            document.save(update_fields=['status', 'processing_log'])

            # Notification d'échec
            if document.uploaded_by.email_notifications:
                send_processing_notification.delay(
                    document_id,
                    'error',
                    f"Le traitement de votre document '{document.title}' a échoué: {str(e)}"
                )

        except Exception as save_error:
            logger.error(f"Impossible de sauvegarder l'état d'erreur: {str(save_error)}")

        return {
            "status": "error",
            "message": str(e),
            "document_id": document_id,
            "retries_exhausted": True
        }


@shared_task(bind=True, max_retries=2)
def update_search_index(self, document_id):
    """
    Met à jour l'index de recherche pour un document
    """
    from .models import Document
    from django.contrib.postgres.search import SearchVector

    try:
        document = Document.objects.select_related('content').get(id=document_id)

        search_vector = SearchVector('title', weight='A')

        if document.description:
            search_vector += SearchVector('description', weight='B')

        if hasattr(document, 'content') and document.content.clean_content:
            search_vector += SearchVector('content__clean_content', weight='C')

        if document.search_keywords:
            keywords_text = ' '.join(document.search_keywords)
            search_vector += SearchVector(keywords_text, weight='B')

        # Mise à jour atomique
        Document.objects.filter(id=document_id).update(search_vector=search_vector)

        logger.info(f"Index de recherche mis à jour pour le document {document_id}")
        return {"status": "success", "document_id": document_id}

    except Document.DoesNotExist:
        logger.error(f"Document {document_id} introuvable pour la mise à jour de l'index")
        return {"status": "error", "message": "Document introuvable"}

    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de l'index pour le document {document_id}: {str(e)}")

        # Retry automatique
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=30)

        return {"status": "error", "message": str(e), "document_id": document_id}


@shared_task
def send_processing_notification(document_id, notification_type, message):
    """
    Envoie une notification de traitement par email
    """
    from .models import Document

    try:
        document = Document.objects.select_related('uploaded_by').get(id=document_id)
        user = document.uploaded_by

        if not user.email_notifications:
            return {"status": "skipped", "reason": "Notifications désactivées"}

        subject_map = {
            'success': f"Document traité avec succès: {document.title}",
            'error': f"Erreur de traitement: {document.title}",
            'warning': f"Attention requise: {document.title}"
        }

        subject = subject_map.get(notification_type, f"Notification: {document.title}")

        html_message = f"""
        <html>
        <body>
            <h2>Notification de traitement de document</h2>
            <p>Bonjour {user.first_name},</p>

            <p>{message}</p>

            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <h3>Détails du document:</h3>
                <ul>
                    <li><strong>Titre:</strong> {document.title}</li>
                    <li><strong>Fichier:</strong> {document.original_filename}</li>
                    <li><strong>Statut:</strong> {document.get_status_display()}</li>
                    <li><strong>Uploadé le:</strong> {document.created_at.strftime('%d/%m/%Y à %H:%M')}</li>
                </ul>
            </div>

            {f'<p><a href="{settings.FRONTEND_URL}/documents/{document.slug}" style="background-color: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Consulter le document</a></p>' if hasattr(settings, 'FRONTEND_URL') else ''}

            <p style="color: #6c757d; font-size: 0.9em;">
                Cordialement,<br>
                L'équipe de gestion documentaire
            </p>
        </body>
        </html>
        """

        plain_message = f"""
        Bonjour {user.first_name},

        {message}

        Détails du document:
        - Titre: {document.title}
        - Fichier: {document.original_filename}
        - Statut: {document.get_status_display()}
        - Uploadé le: {document.created_at.strftime('%d/%m/%Y à %H:%M')}

        Cordialement,
        L'équipe de gestion documentaire
        """

        send_mail(
            subject=subject,
            message=plain_message,
            html_message=html_message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )

        logger.info(f"Notification {notification_type} envoyée pour le document {document_id}")
        return {"status": "success", "notification_type": notification_type}

    except Document.DoesNotExist:
        logger.error(f"Document {document_id} introuvable pour l'envoi de notification")
        return {"status": "error", "message": "Document introuvable"}

    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de notification: {str(e)}")
        return {"status": "error", "message": str(e)}


@shared_task
def send_approval_notification(document_id, approver_id):
    """
    Envoie une notification d'approbation
    """
    from .models import Document, User

    try:
        document = Document.objects.select_related('uploaded_by').get(id=document_id)
        approver = User.objects.get(id=approver_id)
        user = document.uploaded_by

        if not user.email_notifications:
            return {"status": "skipped", "reason": "Notifications désactivées"}

        subject = f"Document approuvé: {document.title}"

        html_message = f"""
        <html>
        <body>
            <h2>Document approuvé</h2>
            <p>Bonjour {user.first_name},</p>

            <p>Votre document "<strong>{document.title}</strong>" a été approuvé par {approver.get_full_name()}.</p>

            <div style="background-color: #d4edda; border-color: #c3e6cb; color: #155724; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <p><strong> Votre document est maintenant publié et accessible.</strong></p>
            </div>

            {f'<p><a href="{settings.FRONTEND_URL}/documents/{document.slug}" style="background-color: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Consulter le document</a></p>' if hasattr(settings, 'FRONTEND_URL') else ''}

            <p style="color: #6c757d; font-size: 0.9em;">
                Cordialement,<br>
                L'équipe de gestion documentaire
            </p>
        </body>
        </html>
        """

        plain_message = f"""
        Bonjour {user.first_name},

        Votre document "{document.title}" a été approuvé par {approver.get_full_name()}.

        Votre document est maintenant publié et accessible.

        Cordialement,
        L'équipe de gestion documentaire
        """

        send_mail(
            subject=subject,
            message=plain_message,
            html_message=html_message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )

        logger.info(f"Notification d'approbation envoyée pour le document {document_id}")
        return {"status": "success"}

    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de notification d'approbation: {str(e)}")
        return {"status": "error", "message": str(e)}


@shared_task
def cleanup_old_files():
    """
    Nettoie les anciens fichiers et documents supprimés
    """
    from .models import Document
    from django.core.files.storage import default_storage

    try:
        cutoff_date = timezone.now() - timedelta(days=30)

        # Documents supprimés logiquement depuis plus de 30 jours
        old_deleted_docs = Document.objects.filter(
            is_deleted=True,
            deleted_at__lt=cutoff_date
        )

        cleaned_count = 0
        errors = []

        for doc in old_deleted_docs:
            try:
                # Supprimer le fichier physique
                if doc.file_path and default_storage.exists(doc.file_path.name):
                    default_storage.delete(doc.file_path.name)

                # Supprimer l'enregistrement de la base
                doc.delete()
                cleaned_count += 1

            except Exception as e:
                error_msg = f"Erreur lors de la suppression du document {doc.id}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        # Nettoyage des fichiers orphelins
        orphaned_files = cleanup_orphaned_files()

        logger.info(
            f"Nettoyage terminé: {cleaned_count} documents supprimés, {orphaned_files} fichiers orphelins supprimés")

        result = {
            "status": "success",
            "documents_cleaned": cleaned_count,
            "orphaned_files_cleaned": orphaned_files,
            "errors": errors
        }

        cache.set('last_cleanup_stats', result, 86400)  # 24h

        return result

    except Exception as e:
        error_msg = f"Erreur lors du nettoyage: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}


def cleanup_orphaned_files():
    """
    Nettoie les fichiers orphelins (sans document correspondant)
    """
    from .models import Document
    from django.core.files.storage import default_storage
    import os

    try:
        orphaned_count = 0

        db_file_paths = set(
            Document.objects.exclude(file_path='').values_list('file_path', flat=True)
        )

        if hasattr(default_storage, 'listdir'):
            for root in ['documents']:
                try:
                    dirs, files = default_storage.listdir(root)
                    for file_name in files:
                        file_path = os.path.join(root, file_name)

                        # Vérifier si le fichier existe dans la base
                        if file_path not in db_file_paths:
                            try:
                                default_storage.delete(file_path)
                                orphaned_count += 1
                                logger.info(f"Fichier orphelin supprimé: {file_path}")
                            except Exception as e:
                                logger.warning(f"Impossible de supprimer le fichier orphelin {file_path}: {str(e)}")

                except Exception as e:
                    logger.warning(f"Erreur lors du parcours du répertoire {root}: {str(e)}")

        return orphaned_count

    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des fichiers orphelins: {str(e)}")
        return 0


@shared_task
def generate_daily_statistics():
    """
    Génère les statistiques quotidiennes et les met en cache
    """
    from .models import Document, User, Theme, DocumentContent
    from django.db.models import Count, Avg, Sum, Q

    try:
        today = timezone.now().date()
        yesterday = today - timedelta(days=1)
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)

        # Statistiques générales
        total_documents = Document.objects.filter(is_deleted=False).count()
        processed_documents = Document.objects.filter(
            status=Document.StatusChoices.PROCESSED,
            is_deleted=False
        ).count()

        # Statistiques quotidiennes
        documents_created_today = Document.objects.filter(
            created_at__date=today,
            is_deleted=False
        ).count()

        documents_processed_today = Document.objects.filter(
            status=Document.StatusChoices.PROCESSED,
            updated_at__date=today,
            is_deleted=False
        ).count()

        documents_created_week = Document.objects.filter(
            created_at__date__gte=week_ago,
            is_deleted=False
        ).count()

        documents_created_month = Document.objects.filter(
            created_at__date__gte=month_ago,
            is_deleted=False
        ).count()

        # Statistiques par statut
        status_distribution = Document.objects.filter(is_deleted=False).values('status').annotate(
            count=Count('id')
        ).order_by('status')

        # Statistiques par type de fichier
        file_type_distribution = Document.objects.filter(is_deleted=False).values('file_type').annotate(
            count=Count('id')
        ).order_by('file_type')

        # Statistiques de contenu
        content_stats = DocumentContent.objects.aggregate(
            avg_word_count=Avg('word_count'),
            avg_page_count=Avg('page_count'),
            avg_extraction_confidence=Avg('extraction_confidence'),
            total_words=Sum('word_count'),
            total_pages=Sum('page_count')
        )

        # Top utilisateurs par uploads
        top_uploaders = User.objects.annotate(
            upload_count=Count('uploaded_documents', filter=Q(uploaded_documents__is_deleted=False))
        ).filter(upload_count__gt=0).order_by('-upload_count')[:10]

        # Top thèmes par nombre de documents
        top_themes = Theme.objects.annotate(
            doc_count=Count('documents', filter=Q(documents__is_deleted=False))
        ).filter(doc_count__gt=0).order_by('-doc_count')[:10]

        # Statistiques d'utilisation du stockage
        storage_stats = Document.objects.filter(is_deleted=False).aggregate(
            total_size_bytes=Sum('file_size'),
            avg_file_size=Avg('file_size'),
            max_file_size=models.Max('file_size')
        )

        total_size_mb = (storage_stats['total_size_bytes'] or 0) / (1024 * 1024)
        avg_file_size_mb = (storage_stats['avg_file_size'] or 0) / (1024 * 1024)
        max_file_size_mb = (storage_stats['max_file_size'] or 0) / (1024 * 1024)

        # Compilation des statistiques
        stats = {
            'date': today.isoformat(),
            'generated_at': timezone.now().isoformat(),

            # Totaux
            'totals': {
                'documents': total_documents,
                'processed_documents': processed_documents,
                'processing_rate': round((processed_documents / total_documents * 100) if total_documents > 0 else 0,
                                         2),
                'active_users': User.objects.filter(is_active=True).count(),
                'themes': Theme.objects.filter(is_deleted=False).count(),
            },

            # Activité quotidienne
            'daily': {
                'documents_created': documents_created_today,
                'documents_processed': documents_processed_today,
                'date': today.isoformat()
            },

            # Activité hebdomadaire
            'weekly': {
                'documents_created': documents_created_week,
                'period_start': week_ago.isoformat(),
                'period_end': today.isoformat()
            },

            # Activité mensuelle
            'monthly': {
                'documents_created': documents_created_month,
                'period_start': month_ago.isoformat(),
                'period_end': today.isoformat()
            },

            # Distributions
            'distributions': {
                'status': list(status_distribution),
                'file_types': list(file_type_distribution)
            },

            # Statistiques de contenu
            'content': {
                'avg_word_count': round(content_stats['avg_word_count'] or 0, 0),
                'avg_page_count': round(content_stats['avg_page_count'] or 0, 1),
                'avg_extraction_confidence': round(float(content_stats['avg_extraction_confidence'] or 0), 4),
                'total_words': content_stats['total_words'] or 0,
                'total_pages': content_stats['total_pages'] or 0
            },

            # Stockage
            'storage': {
                'total_size_mb': round(total_size_mb, 2),
                'avg_file_size_mb': round(avg_file_size_mb, 2),
                'max_file_size_mb': round(max_file_size_mb, 2)
            },

            # Top listes
            'top_uploaders': [
                {
                    'user_id': str(user.id),
                    'name': user.get_full_name(),
                    'upload_count': user.upload_count
                }
                for user in top_uploaders
            ],

            'top_themes': [
                {
                    'theme_id': str(theme.id),
                    'name': theme.name,
                    'document_count': theme.doc_count
                }
                for theme in top_themes
            ]
        }

        # Mise en cache des statistiques
        cache_keys = {
            f'daily_stats_{today.isoformat()}': stats,
            'latest_daily_stats': stats,
            'global_processing_stats': stats
        }

        for key, value in cache_keys.items():
            cache.set(key, value, 86400)  # 24h

        logger.info(f"Statistiques quotidiennes générées pour {today}")
        return {"status": "success", "date": today.isoformat(), "stats_summary": stats['totals']}

    except Exception as e:
        error_msg = f"Erreur lors de la génération des statistiques: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}


@shared_task
def batch_process_documents(document_ids, priority='normal'):
    """
    Traite plusieurs documents en lot
    """
    from .models import Document

    try:
        results = {
            'total': len(document_ids),
            'success': 0,
            'errors': 0,
            'skipped': 0,
            'details': []
        }

        for doc_id in document_ids:
            try:
                document = Document.objects.get(id=doc_id)

                if document.status == Document.StatusChoices.PROCESSING:
                    results['skipped'] += 1
                    results['details'].append({
                        'document_id': doc_id,
                        'status': 'skipped',
                        'reason': 'Already processing'
                    })
                    continue

                # Lancer le traitement
                task = process_document_content.delay(doc_id)
                results['success'] += 1
                results['details'].append({
                    'document_id': doc_id,
                    'status': 'started',
                    'task_id': task.id
                })

            except Document.DoesNotExist:
                results['errors'] += 1
                results['details'].append({
                    'document_id': doc_id,
                    'status': 'error',
                    'reason': 'Document not found'
                })

            except Exception as e:
                results['errors'] += 1
                results['details'].append({
                    'document_id': doc_id,
                    'status': 'error',
                    'reason': str(e)
                })

        logger.info(
            f"Traitement en lot terminé: {results['success']} succès, {results['errors']} erreurs, {results['skipped']} ignorés")
        return results

    except Exception as e:
        error_msg = f"Erreur lors du traitement en lot: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}


@shared_task
def monitor_processing_queue():
    """
    Surveille la file d'attente de traitement et génère des alertes
    """
    from .models import Document
    from django.core.mail import mail_admins

    try:
        threshold = timezone.now() - timedelta(hours=2)
        stuck_documents = Document.objects.filter(
            status=Document.StatusChoices.PROCESSING,
            updated_at__lt=threshold
        )

        # Documents en erreur récents
        error_threshold = timezone.now() - timedelta(hours=24)
        recent_errors = Document.objects.filter(
            status=Document.StatusChoices.ERROR,
            updated_at__gte=error_threshold
        )

        # Statistiques de la file
        queue_stats = {
            'pending': Document.objects.filter(status=Document.StatusChoices.PENDING).count(),
            'processing': Document.objects.filter(status=Document.StatusChoices.PROCESSING).count(),
            'stuck': stuck_documents.count(),
            'recent_errors': recent_errors.count()
        }

        # Alertes si nécessaire
        alerts = []

        if queue_stats['stuck'] > 0:
            alerts.append(f"{queue_stats['stuck']} documents bloqués en traitement depuis plus de 2h")

        if queue_stats['recent_errors'] > 10:
            alerts.append(f"{queue_stats['recent_errors']} documents en erreur dans les dernières 24h")

        if queue_stats['pending'] > 100:
            alerts.append(f"{queue_stats['pending']} documents en attente de traitement")

        # Envoi d'alertes aux administrateurs
        if alerts and hasattr(settings, 'ADMINS') and settings.ADMINS:
            alert_message = f"""
            Alerte de surveillance du traitement de documents:

            {chr(10).join(f'- {alert}' for alert in alerts)}

            Statistiques actuelles:
            - En attente: {queue_stats['pending']}
            - En traitement: {queue_stats['processing']}
            - Bloqués: {queue_stats['stuck']}
            - Erreurs récentes: {queue_stats['recent_errors']}
            """

            mail_admins(
                subject="Alerte - File de traitement des documents",
                message=alert_message,
                fail_silently=True
            )

        # Mise en cache des statistiques de surveillance
        cache.set('processing_queue_stats', queue_stats, 300)  # 5 minutes

        logger.info(f"Surveillance de la file: {queue_stats}")
        return {"status": "success", "queue_stats": queue_stats, "alerts": alerts}

    except Exception as e:
        error_msg = f"Erreur lors de la surveillance: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}