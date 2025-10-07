#!/bin/bash

# ==========================================
# Script de démarrage pour DGI-EXTRACTOR
# Avec configuration Celery complète
# ==========================================

set -e  # Arrêter le script en cas d'erreur

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="DGI-EXTRACTOR"
PROJECT_DIR="/Users/krohn/PycharmProjects/DME-AGENTS/dgi-extractor"
VENV_PATH="$PROJECT_DIR/venv"
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/pids"

# Ports par défaut
DJANGO_PORT=${DJANGO_PORT:-8000}
CELERY_FLOWER_PORT=${CELERY_FLOWER_PORT:-5555}

# Redis configuration
REDIS_HOST=${REDIS_HOST:-"37.60.239.221"}
REDIS_PORT=${REDIS_PORT:-6379}
REDIS_DB=${REDIS_DB:-1}

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Démarrage de $PROJECT_NAME${NC}"
echo -e "${BLUE}=========================================${NC}"

# Fonction pour afficher les logs
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Fonction de nettoyage
cleanup() {
    log_info "Nettoyage des processus..."

    # Arrêter les processus Celery
    if [ -f "$PID_DIR/celery_worker.pid" ]; then
        if kill -0 $(cat "$PID_DIR/celery_worker.pid") 2>/dev/null; then
            kill $(cat "$PID_DIR/celery_worker.pid")
            log_info " Celery Worker arrêté"
        fi
        rm -f "$PID_DIR/celery_worker.pid"
    fi

    if [ -f "$PID_DIR/celery_beat.pid" ]; then
        if kill -0 $(cat "$PID_DIR/celery_beat.pid") 2>/dev/null; then
            kill $(cat "$PID_DIR/celery_beat.pid")
            log_info " Celery Beat arrêté"
        fi
        rm -f "$PID_DIR/celery_beat.pid"
    fi

    if [ -f "$PID_DIR/flower.pid" ]; then
        if kill -0 $(cat "$PID_DIR/flower.pid") 2>/dev/null; then
            kill $(cat "$PID_DIR/flower.pid")
            log_info " Flower arrêté"
        fi
        rm -f "$PID_DIR/flower.pid"
    fi

    # Arrêter Django si lancé par ce script
    if [ -f "$PID_DIR/django.pid" ]; then
        if kill -0 $(cat "$PID_DIR/django.pid") 2>/dev/null; then
            kill $(cat "$PID_DIR/django.pid")
            log_info " Django arrêté"
        fi
        rm -f "$PID_DIR/django.pid"
    fi

    log_info "🏁 Nettoyage terminé"
}

# Gestion du signal SIGINT (Ctrl+C)
trap cleanup SIGINT SIGTERM

# Fonction de vérification des prérequis
check_prerequisites() {
    log_info "🔍 Vérification des prérequis..."

    # Vérifier que le répertoire du projet existe
    if [ ! -d "$PROJECT_DIR" ]; then
        log_error "Répertoire du projet introuvable: $PROJECT_DIR"
        exit 1
    fi

    cd "$PROJECT_DIR"

    # Vérifier l'environnement virtuel
    if [ ! -d "$VENV_PATH" ]; then
        log_warn "Environnement virtuel introuvable. Création..."
        python3 -m venv "$VENV_PATH"
    fi

    # Activer l'environnement virtuel
    source "$VENV_PATH/bin/activate"

    # Vérifier Python
    if ! command -v python &> /dev/null; then
        log_error "Python non trouvé dans l'environnement virtuel"
        exit 1
    fi

    # Vérifier les dépendances critiques
    python -c "import django" 2>/dev/null || {
        log_error "Django non installé. Installation des dépendances..."
        pip install -r requirements.txt
    }

    python -c "import celery" 2>/dev/null || {
        log_error "Celery non installé"
        exit 1
    }

    # Vérifier Redis
    log_info "🔗 Vérification de la connexion Redis..."
    python -c "
import redis
try:
    r = redis.Redis(host='$REDIS_HOST', port=$REDIS_PORT, db=$REDIS_DB)
    r.ping()
    print(' Redis connecté')
except Exception as e:
    print(f' Erreur Redis: {e}')
    exit(1)
" || exit 1

    log_info " Prérequis vérifiés"
}

# Créer les répertoires nécessaires
create_directories() {
    log_info " Création des répertoires..."

    mkdir -p "$LOG_DIR"
    mkdir -p "$PID_DIR"

    # Créer le répertoire pour les médias si nécessaire
    mkdir -p "$PROJECT_DIR/media"
    mkdir -p "$PROJECT_DIR/static"

    log_info " Répertoires créés"
}

# Préparer Django
setup_django() {
    log_info "🔧 Configuration Django..."

    # Variables d'environnement Django
    export DJANGO_SETTINGS_MODULE="core.settings"

    # Migrations
    log_info " Application des migrations..."
    python manage.py makemigrations --noinput
    python manage.py migrate --noinput

    # Collecte des fichiers statiques
    log_info " Collecte des fichiers statiques..."
    python manage.py collectstatic --noinput --clear

    # Vérifier la configuration
    log_info "🧪 Vérification de la configuration Django..."
    python manage.py check --deploy

    log_info " Django configuré"
}

# Démarrer Celery Worker
start_celery_worker() {
    log_info " Démarrage Celery Worker..."

    celery -A core worker \
        --loglevel=info \
        --concurrency=4 \
        --max-tasks-per-child=1000 \
        --time-limit=7200 \
        --soft-time-limit=3600 \
        --pidfile="$PID_DIR/celery_worker.pid" \
        --logfile="$LOG_DIR/celery_worker.log" \
        --detach

    # Vérifier que le worker a démarré
    sleep 3
    if [ -f "$PID_DIR/celery_worker.pid" ] && kill -0 $(cat "$PID_DIR/celery_worker.pid") 2>/dev/null; then
        log_info " Celery Worker démarré (PID: $(cat "$PID_DIR/celery_worker.pid"))"
    else
        log_error " Échec du démarrage Celery Worker"
        exit 1
    fi
}

# Démarrer Celery Beat (scheduler)
start_celery_beat() {
    log_info " Démarrage Celery Beat..."

    celery -A core beat \
        --loglevel=info \
        --pidfile="$PID_DIR/celery_beat.pid" \
        --logfile="$LOG_DIR/celery_beat.log" \
        --schedule="$PROJECT_DIR/celerybeat-schedule" \
        --detach

    # Vérifier que beat a démarré
    sleep 2
    if [ -f "$PID_DIR/celery_beat.pid" ] && kill -0 $(cat "$PID_DIR/celery_beat.pid") 2>/dev/null; then
        log_info " Celery Beat démarré (PID: $(cat "$PID_DIR/celery_beat.pid"))"
    else
        log_error " Échec du démarrage Celery Beat"
        exit 1
    fi
}

# Démarrer Flower (monitoring Celery)
start_flower() {
    log_info " Démarrage Flower (monitoring Celery)..."

    celery -A core flower \
        --port=$CELERY_FLOWER_PORT \
        --pidfile="$PID_DIR/flower.pid" \
        --logfile="$LOG_DIR/flower.log" \
        --detach

    # Vérifier que flower a démarré
    sleep 2
    if [ -f "$PID_DIR/flower.pid" ] && kill -0 $(cat "$PID_DIR/flower.pid") 2>/dev/null; then
        log_info " Flower démarré sur http://localhost:$CELERY_FLOWER_PORT (PID: $(cat "$PID_DIR/flower.pid"))"
    else
        log_warn "⚠  Flower n'a pas pu démarrer (optionnel)"
    fi
}

# Démarrer Django
start_django() {
    log_info " Démarrage du serveur Django..."

    python manage.py runserver 0.0.0.0:$DJANGO_PORT \
        > "$LOG_DIR/django.log" 2>&1 &

    DJANGO_PID=$!
    echo $DJANGO_PID > "$PID_DIR/django.pid"

    # Attendre que Django démarre
    sleep 3

    # Vérifier que Django répond
    if curl -f http://localhost:$DJANGO_PORT/admin/ &>/dev/null; then
        log_info " Django démarré sur http://localhost:$DJANGO_PORT (PID: $DJANGO_PID)"
    else
        log_warn "  Django en cours de démarrage... Vérifiez les logs si nécessaire"
    fi
}

# Afficher le statut des services
show_status() {
    echo -e "\n${BLUE}=========================================${NC}"
    echo -e "${BLUE} STATUT DES SERVICES${NC}"
    echo -e "${BLUE}=========================================${NC}"

    # Django
    if [ -f "$PID_DIR/django.pid" ] && kill -0 $(cat "$PID_DIR/django.pid") 2>/dev/null; then
        echo -e "${GREEN} Django${NC} - http://localhost:$DJANGO_PORT (PID: $(cat "$PID_DIR/django.pid"))"
    else
        echo -e "${RED} Django${NC} - Non démarré"
    fi

    # Celery Worker
    if [ -f "$PID_DIR/celery_worker.pid" ] && kill -0 $(cat "$PID_DIR/celery_worker.pid") 2>/dev/null; then
        echo -e "${GREEN} Celery Worker${NC} (PID: $(cat "$PID_DIR/celery_worker.pid"))"
    else
        echo -e "${RED} Celery Worker${NC} - Non démarré"
    fi

    # Celery Beat
    if [ -f "$PID_DIR/celery_beat.pid" ] && kill -0 $(cat "$PID_DIR/celery_beat.pid") 2>/dev/null; then
        echo -e "${GREEN} Celery Beat${NC} (PID: $(cat "$PID_DIR/celery_beat.pid"))"
    else
        echo -e "${RED} Celery Beat${NC} - Non démarré"
    fi

    # Flower
    if [ -f "$PID_DIR/flower.pid" ] && kill -0 $(cat "$PID_DIR/flower.pid") 2>/dev/null; then
        echo -e "${GREEN} Flower${NC} - http://localhost:$CELERY_FLOWER_PORT (PID: $(cat "$PID_DIR/flower.pid"))"
    else
        echo -e "${YELLOW}️  Flower${NC} - Non démarré (optionnel)"
    fi

    echo -e "\n${BLUE} FICHIERS IMPORTANTS:${NC}"
    echo -e "   Logs: $LOG_DIR/"
    echo -e "   PIDs: $PID_DIR/"
    echo -e "   Projet: $PROJECT_DIR"

    echo -e "\n${BLUE}🔗 URLS UTILES:${NC}"
    echo -e "    Application: http://localhost:$DJANGO_PORT"
    echo -e "    API Documentation: http://localhost:$DJANGO_PORT/api/docs/"
    echo -e "    Admin Django: http://localhost:$DJANGO_PORT/admin/"
    echo -e "    Flower (Celery): http://localhost:$CELERY_FLOWER_PORT"
    echo -e "    RAG Endpoints:"
    echo -e "      • POST /api/v1/processing/rag/query/ (RAG multimode)"
    echo -e "      • POST /api/v1/processing/rag/generate/ (Génération HF)"
    echo -e "      • POST /api/v1/processing/rag/chat/completions/ (Chat OpenAI-like)"
    echo -e "      • GET /api/v1/public/all-extracted-content/ (Contenu public)"

    echo -e "\n${YELLOW} COMMANDES UTILES:${NC}"
    echo -e "   Arrêter: Ctrl+C ou kill \$(cat $PID_DIR/*.pid)"
    echo -e "   Logs en temps réel: tail -f $LOG_DIR/*.log"
    echo -e "   Restart Celery: ./start_app.sh restart-celery"
}

# Fonction pour redémarrer seulement Celery
restart_celery() {
    log_info " Redémarrage des services Celery..."

    # Arrêter Celery
    if [ -f "$PID_DIR/celery_worker.pid" ]; then
        kill $(cat "$PID_DIR/celery_worker.pid") 2>/dev/null || true
        rm -f "$PID_DIR/celery_worker.pid"
    fi

    if [ -f "$PID_DIR/celery_beat.pid" ]; then
        kill $(cat "$PID_DIR/celery_beat.pid") 2>/dev/null || true
        rm -f "$PID_DIR/celery_beat.pid"
    fi

    if [ -f "$PID_DIR/flower.pid" ]; then
        kill $(cat "$PID_DIR/flower.pid") 2>/dev/null || true
        rm -f "$PID_DIR/flower.pid"
    fi

    sleep 2

    # Redémarrer
    start_celery_worker
    start_celery_beat
    start_flower

    log_info " Services Celery redémarrés"
}

# Gestion des arguments
case "${1:-start}" in
    "start")
        # Démarrage complet
        check_prerequisites
        create_directories
        setup_django
        start_celery_worker
        start_celery_beat
        start_flower
        start_django
        show_status

        echo -e "\n${GREEN} $PROJECT_NAME démarré avec succès !${NC}"
        echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter tous les services${NC}\n"

        # Garder le script en vie et surveiller les processus
        while true; do
            sleep 10

            # Vérifier que les processus critiques sont toujours en vie
            if [ -f "$PID_DIR/django.pid" ] && ! kill -0 $(cat "$PID_DIR/django.pid") 2>/dev/null; then
                log_error "Django s'est arrêté de manière inattendue"
                break
            fi

            if [ -f "$PID_DIR/celery_worker.pid" ] && ! kill -0 $(cat "$PID_DIR/celery_worker.pid") 2>/dev/null; then
                log_error "Celery Worker s'est arrêté de manière inattendue"
                break
            fi
        done
        ;;

    "stop")
        log_info " Arrêt de $PROJECT_NAME..."
        cleanup
        ;;

    "restart")
        log_info " Redémarrage de $PROJECT_NAME..."
        cleanup
        sleep 2
        exec "$0" start
        ;;

    "restart-celery")
        cd "$PROJECT_DIR"
        source "$VENV_PATH/bin/activate"
        restart_celery
        ;;

    "status")
        show_status
        ;;

    "logs")
        echo -e "${BLUE} Logs en temps réel (Ctrl+C pour quitter):${NC}"
        tail -f "$LOG_DIR"/*.log
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|restart-celery|status|logs}"
        echo ""
        echo "  start         - Démarre tous les services"
        echo "  stop          - Arrête tous les services"
        echo "  restart       - Redémarre tous les services"
        echo "  restart-celery- Redémarre seulement Celery"
        echo "  status        - Affiche le statut des services"
        echo "  logs          - Affiche les logs en temps réel"
        exit 1
        ;;
esac