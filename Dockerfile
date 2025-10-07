# ==========================================
# Dockerfile optimisé pour DGI-EXTRACTOR
# Avec support Celery et RAG Hugging Face
# ==========================================

FROM python:3.11-slim-bookworm AS builder

# Variables d'environnement pour l'optimisation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Installation des dépendances système pour la construction
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn uvicorn[standard]

# ==========================================
# Image finale
# ==========================================

FROM python:3.11-slim-bookworm

# Métadonnées
LABEL maintainer="DGI-EXTRACTOR Team" \
      version="1.0" \
      description="Django app with Celery and Hugging Face RAG"

# Variables d'environnement optimisées
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DJANGO_SETTINGS_MODULE=core.settings \
    CELERY_BROKER_URL=redis://redis:6379/0 \
    CELERY_RESULT_BACKEND=redis://redis:6379/0 \
    RAG_USE_GPU=False \
    RAG_USE_DUMMY_DATASET=True \
    RAG_USE_CUSTOM_RETRIEVER=True \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Création de l'utilisateur non-privilégié
RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home --shell /bin/bash appuser

# Installation des dépendances système runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    netcat-openbsd \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances Python depuis builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Création des répertoires nécessaires
RUN mkdir -p \
    /app/logs \
    /app/pids \
    /app/media \
    /app/static \
    /app/staticfiles \
    /vol/web/media \
    /vol/web/static \
    && chown -R appuser:appgroup \
       /app/logs \
       /app/pids \
       /app/media \
       /app/static \
       /app/staticfiles \
       /vol/web \
    && chmod -R 755 \
       /app/logs \
       /app/pids \
       /app/media \
       /app/static \
       /app/staticfiles \
       /vol/web

# Copie du code source
COPY --chown=appuser:appgroup . /app/

# Rendre les scripts exécutables
RUN chmod +x /app/start_app.sh \
    && chmod +x /app/stop_app.sh \
    && chmod +x /app/dev_server.sh

# Exposition des ports
EXPOSE 8000 5555

# Configuration du healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/admin/ || exit 1

# Basculer vers l'utilisateur non-privilégié
USER appuser

# Point d'entrée et commande par défaut
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["web"]
