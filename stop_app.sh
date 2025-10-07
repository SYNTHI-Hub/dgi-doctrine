#!/bin/bash

# ==========================================
# Script d'arrêt pour DGI-EXTRACTOR
# ==========================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_DIR="/Users/krohn/PycharmProjects/DME-AGENTS/dgi-extractor"
PID_DIR="$PROJECT_DIR/pids"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo -e "${BLUE}🛑 Arrêt de DGI-EXTRACTOR${NC}"

# Fonction pour arrêter un service
stop_service() {
    local service_name=$1
    local pid_file=$2

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            sleep 2

            # Vérifier si le processus est vraiment arrêté
            if kill -0 "$pid" 2>/dev/null; then
                log_warn "Force killing $service_name (PID: $pid)"
                kill -9 "$pid" 2>/dev/null || true
            fi

            log_info "✅ $service_name arrêté"
        else
            log_warn "$service_name n'était pas en cours d'exécution"
        fi
        rm -f "$pid_file"
    else
        log_warn "Fichier PID $service_name introuvable"
    fi
}

# Arrêter tous les services
stop_service "Django" "$PID_DIR/django.pid"
stop_service "Celery Worker" "$PID_DIR/celery_worker.pid"
stop_service "Celery Beat" "$PID_DIR/celery_beat.pid"
stop_service "Flower" "$PID_DIR/flower.pid"

# Nettoyer les fichiers temporaires
rm -f "$PROJECT_DIR/celerybeat-schedule"
rm -f "$PROJECT_DIR/celerybeat.pid"

# Arrêter tous les processus Celery restants (sécurité)
pkill -f "celery.*core" 2>/dev/null || true

log_info "🏁 Tous les services sont arrêtés"