#!/usr/bin/env python3
"""
Script de test pour le système de fallback RAG vers Gemini
"""

import os
import sys
import django

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append('/Users/krohn/PycharmProjects/DME-AGENTS/dgi-extractor')

# Configuration Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from doctrine.services.rag import rag_retriever, RAGMode
from doctrine.models import Document, DocumentContent
from django.utils import timezone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_fallback():
    """Test du système RAG avec fallback Gemini"""

    print("=== Test du système RAG avec fallback Gemini ===")

    # Test 1: Vérifier la disponibilité des documents
    print("\n1. Vérification des documents dans la base de données...")
    doc_count = Document.objects.filter(
        is_deleted=False,
        status=Document.StatusChoices.PROCESSED,
        is_searchable=True
    ).count()

    content_count = DocumentContent.objects.filter(
        document__is_deleted=False,
        document__status=Document.StatusChoices.PROCESSED,
        document__is_searchable=True
    ).count()

    print(f"Documents disponibles: {doc_count}")
    print(f"Contenus de documents: {content_count}")

    if content_count == 0:
        print("⚠️  Aucun contenu de document trouvé. Créons un exemple...")
        create_test_document()

    # Test 2: Test du mode similarité seulement
    print("\n2. Test du mode similarité...")
    query = "politique de sécurité"
    result_similarity = rag_retriever.retrieve(
        query=query,
        k=3,
        mode=RAGMode.SIMILARITY_ONLY
    )
    print(f"Résultats similarité: {result_similarity['count']} documents trouvés")

    # Test 3: Test du mode Hugging Face (qui devrait déclencher le fallback)
    print("\n3. Test du mode Hugging Face (avec fallback attendu vers Gemini)...")
    result_hf = rag_retriever.retrieve(
        query=query,
        k=3,
        mode=RAGMode.HUGGINGFACE_RAG
    )

    print(f"Mode utilisé: {result_hf.get('mode', 'unknown')}")
    print(f"Résultats: {result_hf['count']} documents")

    if 'generated_response' in result_hf:
        response = result_hf['generated_response']
        print(f"Réponse générée ({len(response)} caractères):")
        print(f"'{response[:200]}{'...' if len(response) > 200 else ''}'")

    if 'error' in result_hf:
        print(f"Erreur: {result_hf['error']}")

    # Test 4: Test du mode hybride
    print("\n4. Test du mode hybride...")
    result_hybrid = rag_retriever.retrieve(
        query=query,
        k=3,
        mode=RAGMode.HYBRID
    )

    print(f"Mode utilisé: {result_hybrid.get('mode', 'unknown')}")
    print(f"Résultats: {result_hybrid['count']} documents")

    if 'generated_response' in result_hybrid:
        response = result_hybrid['generated_response']
        if response:
            print(f"Réponse générée ({len(response)} caractères):")
            print(f"'{response[:200]}{'...' if len(response) > 200 else ''}'")

    # Test 5: Test direct du fallback Gemini
    print("\n5. Test direct du fallback Gemini...")
    try:
        started_at = timezone.now()
        result_gemini = rag_retriever._fallback_to_gemini(
            query=query,
            k=3,
            scope='all',
            document=None,
            started_at=started_at
        )

        print(f"Mode utilisé: {result_gemini.get('mode', 'unknown')}")
        print(f"Résultats: {result_gemini['count']} documents")
        print(f"Temps de traitement: {result_gemini['took_ms']}ms")

        if 'generated_response' in result_gemini:
            response = result_gemini['generated_response']
            print(f"Réponse Gemini ({len(response)} caractères):")
            print(f"'{response[:300]}{'...' if len(response) > 300 else ''}'")

        if 'error' in result_gemini:
            print(f"Erreur Gemini: {result_gemini['error']}")

    except Exception as e:
        print(f"Erreur lors du test direct Gemini: {str(e)}")

    print("\n=== Fin des tests ===")

def create_test_document():
    """Crée un document de test si aucun n'existe"""
    from doctrine.models import Theme, DocumentCategory, User

    try:
        # Obtenir ou créer un utilisateur de test
        user, created = User.objects.get_or_create(
            email='test@example.com',
            defaults={
                'first_name': 'Test',
                'last_name': 'User',
                'is_staff': True
            }
        )

        # Obtenir ou créer un thème
        theme, created = Theme.objects.get_or_create(
            name='Test Theme',
            defaults={'description': 'Thème de test'}
        )

        # Obtenir ou créer une catégorie
        category, created = DocumentCategory.objects.get_or_create(
            name='Test Category',
            defaults={'description': 'Catégorie de test'}
        )

        # Créer un document de test
        document = Document.objects.create(
            title='Document de test sur la sécurité',
            description='Document de test contenant des informations sur les politiques de sécurité',
            original_filename='test_security.pdf',
            file_size=1000,
            file_checksum='test123',
            theme=theme,
            category=category,
            uploaded_by=user,
            status=Document.StatusChoices.PROCESSED,
            is_searchable=True
        )

        # Créer le contenu du document
        content_text = """
        Politique de sécurité informatique

        1. Introduction
        Cette politique de sécurité définit les règles et procédures à suivre pour assurer la sécurité des systèmes d'information de l'organisation.

        2. Objectifs
        - Protéger les données sensibles
        - Prévenir les accès non autorisés
        - Assurer la continuité des services

        3. Responsabilités
        Chaque utilisateur est responsable de la sécurité de ses accès et doit :
        - Utiliser des mots de passe forts
        - Ne pas partager ses identifiants
        - Signaler tout incident de sécurité

        4. Mesures techniques
        - Chiffrement des données
        - Sauvegarde régulière
        - Mise à jour des systèmes
        """

        DocumentContent.objects.create(
            document=document,
            raw_content=content_text,
            clean_content=content_text,
            processing_status=DocumentContent.ProcessingStatus.COMPLETED,
            word_count=len(content_text.split())
        )

        print("✅ Document de test créé avec succès")

    except Exception as e:
        print(f"❌ Erreur lors de la création du document de test: {str(e)}")

if __name__ == '__main__':
    test_rag_fallback()