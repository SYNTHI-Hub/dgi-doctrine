import os
from celery import Celery
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

app = Celery('dgi_extractor')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.autodiscover_tasks()


app.conf.update(
    # Optimisations worker
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=True,
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',

    task_soft_time_limit=3600,
    task_time_limit=7200,
    task_acks_late=True,
    worker_prefetch_multiplier=1,

    # Résultats
    result_expires=3600,  # 1 heure
    result_backend_transport_options={
        'master_name': 'mymaster',
        'visibility_timeout': 3600,
        'retry_policy': {
            'timeout': 5.0
        }
    },

    # Sérialisation
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Routes pour les queues spécialisées
    task_routes={
        'doctrine.tasks.process_document_content': {'queue': 'document_processing'},
        'doctrine.tasks.*': {'queue': 'default'},
    },

    # Configuration des queues
    task_default_queue='default',
    task_queues={
        'default': {
            'exchange': 'default',
            'exchange_type': 'direct',
            'routing_key': 'default',
        },
        'document_processing': {
            'exchange': 'document_processing',
            'exchange_type': 'direct',
            'routing_key': 'document_processing',
        },
        'rag_processing': {
            'exchange': 'rag_processing',
            'exchange_type': 'direct',
            'routing_key': 'rag_processing',
        },
    },
)

app.conf.beat_schedule = {
    'cleanup-old-files': {
        'task': 'doctrine.tasks.cleanup_old_files',
        'schedule': 86400.0,
    },
    'weekly-stats': {
        'task': 'doctrine.tasks.generate_weekly_stats',
        'schedule': 604800.0,  # 7 jours
        'args': (),
    },
}

@app.task(bind=True)
def debug_task(self):
    """Tâche de debug pour tester Celery"""
    print(f'Request: {self.request!r}')

# Configuration pour le monitoring avec Flower
if hasattr(settings, 'FLOWER_BASIC_AUTH'):
    app.conf.flower_basic_auth = settings.FLOWER_BASIC_AUTH