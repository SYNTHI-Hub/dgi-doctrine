from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os
import hashlib
import uuid

class DocumentStorage:
    '''
    Gestionnaire personnalisé pour le stockage des documents
    '''

    @staticmethod
    def generate_file_path(instance, filename):
        '''
        Génère un chemin de fichier unique basé sur le hash et l'UUID
        '''

        ext = os.path.splitext(filename)[1].lower()

        unique_filename = f"{uuid.uuid4().hex}{ext}"

        from datetime import datetime
        now = datetime.now()

        return f"documents/{now.year}/{now.month:02d}/{now.day:02d}/{unique_filename}"

    @staticmethod
    def calculate_file_hash(file_obj):
        '''
        Calcule le hash SHA-256 d'un fichier
        '''
        hash_sha256 = hashlib.sha256()

        file_obj.seek(0)

        for chunk in iter(lambda: file_obj.read(4096), b""):
            hash_sha256.update(chunk)

        file_obj.seek(0)

        return hash_sha256.hexdigest()

    @staticmethod
    def check_duplicate(file_hash):
        '''
        Vérifie si un fichier avec le même hash existe déjà
        '''
        from .models import Document
        return Document.objects.filter(file_checksum=file_hash).first()

    @staticmethod
    def save_document_file(file_obj, filename, user):
        '''
        Sauvegarde un fichier document avec vérifications
        '''
        file_hash = DocumentStorage.calculate_file_hash(file_obj)

        existing_doc = DocumentStorage.check_duplicate(file_hash)
        if existing_doc:
            return existing_doc, True

        file_size_mb = file_obj.size / (1024 * 1024)
        if user.is_quota_exceeded(file_size_mb):
            raise ValueError("Quota de stockage dépassé")

        file_path = DocumentStorage.generate_file_path(None, filename)

        saved_path = default_storage.save(file_path, ContentFile(file_obj.read()))

        return saved_path, False