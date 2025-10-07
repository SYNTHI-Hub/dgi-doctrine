#!/bin/bash

# ==========================================
# Docker Entrypoint pour DGI-EXTRACTOR
# Support multi-services (Django, Celery, Flower)
# ==========================================

set -e

# Couleurs pour les logs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[DOCKER-INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[DOCKER-WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[DOCKER-ERROR]${NC} $1"
}

# Fonction d'attente pour les services externes
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}

    log_info "Attente de $service_name ($host:$port)..."

    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" >/dev/null 2>&1; then
            log_info " $service_name disponible"
            return 0
        fi
        echo -n "."
        sleep 1
    done

    log_error " Timeout: $service_name non disponible après ${timeout}s"
    exit 1
}

# Fonction de vérification de santé
health_check() {
    log_info "🏥 Vérification de santé..."

    # Vérifier Django
    python manage.py check --deploy

    # Vérifier la base de données
    python manage.py showmigrations --verbosity=0 >/dev/null

    log_info " Vérification de santé OK"
}

# Migrations et setup Django
setup_django() {
    log_info " Configuration Django..."

    # Créer les répertoires si ils n'existent pas
    mkdir -p /app/logs /app/pids /app/media /app/staticfiles

    # Appliquer les migrations
    log_info " Application des migrations..."
    python manage.py makemigrations --noinput
    python manage.py migrate --noinput

    # Collecter les fichiers statiques
    log_info " Collecte des fichiers statiques..."
    python manage.py collectstatic --noinput --clear

    log_info " Django configuré"
}

# Fonction principale
main() {
    local command="${1:-web}"

    log_info " Démarrage DGI-EXTRACTOR - Mode: $command"
    log_info "  Version: $(cat /app/VERSION 2>/dev/null || echo 'dev')"

    # Attendre les services externes selon la configuration
    if [ "${WAIT_FOR_REDIS:-true}" = "true" ]; then
        redis_host=$(echo "$CELERY_BROKER_URL" | sed 's/redis:\/\/\([^:]*\).*/\1/' || echo "${REDIS_HOST:-redis}")
        redis_port=$(echo "$CELERY_BROKER_URL" | sed 's/.*:\([0-9]*\).*/\1/' || echo "${REDIS_PORT:-6379}")
        wait_for_service "$redis_host" "$redis_port" "Redis"
    fi

    if [ "${WAIT_FOR_DB:-true}" = "true" ]; then
        wait_for_service "${DB_HOST:-postgres}" "${DB_PORT:-5432}" "PostgreSQL"
    fi

    # Configuration selon le mode
    case "$command" in
        "web"|"django")
            setup_django
            health_check
            log_info " Démarrage serveur web Django..."
            log_info "Disponible sur http://0.0.0.0:8000"

            exec gunicorn core.wsgi:application \
                --bind 0.0.0.0:8000 \
                --workers ${GUNICORN_WORKERS:-4} \
                --worker-class ${GUNICORN_WORKER_CLASS:-gthread} \
                --worker-connections ${GUNICORN_WORKER_CONNECTIONS:-1000} \
                --max-requests ${GUNICORN_MAX_REQUESTS:-1000} \
                --max-requests-jitter ${GUNICORN_MAX_REQUESTS_JITTER:-50} \
                --timeout ${GUNICORN_TIMEOUT:-300} \
                --keep-alive 2 \
                --log-level ${LOG_LEVEL:-info} \
                --access-logfile - \
                --error-logfile - \
                --capture-output
            ;;

        "celery-worker")
            setup_django
            log_info " Démarrage Celery Worker..."
            log_info " Queues: ${CELERY_QUEUES:-default,document_processing,rag_processing}"

            exec celery -A core worker \
                --loglevel=${CELERY_LOG_LEVEL:-info} \
                --concurrency=${CELERY_CONCURRENCY:-4} \
                --max-tasks-per-child=${CELERY_MAX_TASKS_PER_CHILD:-1000} \
                --time-limit=${CELERY_TIME_LIMIT:-7200} \
                --soft-time-limit=${CELERY_SOFT_TIME_LIMIT:-3600} \
                --queues=${CELERY_QUEUES:-default,document_processing,rag_processing} \
                --without-gossip \
                --without-mingle \
                --without-heartbeat \
                --optimization=fair
            ;;

        "celery-beat")
            setup_django
            log_info " Démarrage Celery Beat (Scheduler)..."

            exec celery -A core beat \
                --loglevel=${CELERY_LOG_LEVEL:-info} \
                --schedule=/app/celerybeat-schedule \
                --scheduler=django_celery_beat.schedulers:DatabaseScheduler
            ;;

        "flower")
            log_info " Démarrage Flower (Monitoring Celery)..."
            log_info " Disponible sur http://0.0.0.0:5555"

            exec celery -A core flower \
                --port=5555 \
                --broker=${CELERY_BROKER_URL} \
                --url_prefix=${FLOWER_URL_PREFIX:-/flower} \
                --basic_auth=${FLOWER_BASIC_AUTH:-admin:admin} \
                --max_tasks=${FLOWER_MAX_TASKS:-10000}
            ;;

        "shell")
            setup_django
            log_info " Démarrage Django shell..."
            exec python manage.py shell
            ;;

        "bash")
            log_info " Démarrage bash shell..."
            exec /bin/bash
            ;;

        "migrate")
            log_info " Application des migrations uniquement..."
            python manage.py makemigrations --noinput
            python manage.py migrate --noinput
            log_info " Migrations appliquées"
            ;;

        "collectstatic")
            log_info " Collecte des fichiers statiques uniquement..."
            python manage.py collectstatic --noinput --clear
            log_info " Fichiers statiques collectés"
            ;;

        "test")
            setup_django
            log_info " Exécution des tests..."
            exec python manage.py test ${TEST_ARGS:-}
            ;;

        "dev")
            setup_django
            log_info " Mode développement Django..."
            log_info "Disponible sur http://0.0.0.0:8000"
            exec python manage.py runserver 0.0.0.0:8000
            ;;

        "worker-dev")
            setup_django
            log_info " Celery Worker mode développement..."
            exec celery -A core worker \
                --loglevel=debug \
                --concurrency=2 \
                --without-gossip \
                --without-mingle \
                --without-heartbeat
            ;;

        "all")
            log_info " Démarrage de tous les services (utilise les scripts shell)..."
            exec /app/start_app.sh
            ;;

        "rag-test")
            setup_django
            log_info " Test des fonctionnalités RAG..."
            python -c "
from doctrine.services.rag import rag_retriever
from doctrine.services.huggingface_rag import huggingface_rag_service

print(' Services RAG importés avec succès')

# Test simple
try:
    info = huggingface_rag_service.get_model_info()
    print(f' Modèle RAG: {info}')
except Exception as e:
    print(f'  RAG non disponible: {e}')
"
            ;;

        "check")
            setup_django
            log_info " Vérifications complètes..."

            # Vérifications Django
            python manage.py check --deploy

            # Vérifications Celery
            celery -A core inspect ping >/dev/null 2>&1 && echo " Celery OK" || echo " Celery KO"

            # Vérifications RAG
            python -c "
try:
    from doctrine.services.huggingface_rag import huggingface_rag_service
    print(' Services RAG OK')
except Exception as e:
    print(f'  Services RAG: {e}')
"
            log_info " Vérifications terminées"
            ;;

        *)
            log_info " Commande personnalisée: $*"
            exec "$@"
            ;;
    esac
}

# Gestion des signaux pour un arrêt propre
cleanup() {
    log_info " Arrêt en cours..."
    # Celery et autres processus se termineront proprement
    exit 0
}

trap cleanup SIGTERM SIGINT

# Point d'entrée principal
main "$@"