FROM python:3.11-slim-bookworm as builder

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app


FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV DJANGO_SETTINGS_MODULE core.settings.production

WORKDIR /app

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

RUN mkdir -p /vol/web/media /vol/web/static \
    && chown -R appuser:appgroup /vol/web/media /vol/web/static \
    && chmod -R 755 /vol/web/media /vol/web/static

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY . /app


COPY ./docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

USER appuser

ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["gunicorn", "core.wsgi:application", "--bind", "0.0.0.0:8000"]