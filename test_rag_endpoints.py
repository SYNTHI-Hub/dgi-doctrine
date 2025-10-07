#!/usr/bin/env python3
"""
Script de test pour les endpoints RAG avec la nouvelle logique Gemini->Hugging Face
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"  # Ajustez selon votre configuration
API_BASE = f"{BASE_URL}/api/v1/processing/rag"

# Headers pour les tests
HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

def test_endpoint(url: str, method: str = 'POST', data: Dict = None, params: Dict = None) -> Dict[str, Any]:
    """Test générique d'un endpoint"""

    print(f"\n🧪 Test: {method} {url}")
    if data:
        print(f"📤 Données: {json.dumps(data, indent=2)}")
    if params:
        print(f"📤 Paramètres: {params}")

    try:
        start_time = time.time()

        if method.upper() == 'POST':
            response = requests.post(url, json=data, headers=HEADERS, timeout=30)
        else:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)

        end_time = time.time()
        duration = (end_time - start_time) * 1000

        print(f"⏱️  Temps de réponse: {duration:.0f}ms")
        print(f"📊 Code de statut: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # Analyser la réponse pour voir quel service a été utilisé
            mode = result.get('mode', 'unknown')
            generation_method = result.get('generation_metadata', {}).get('generation_method', 'unknown')

            print(f"✅ Succès - Mode: {mode}")
            print(f"🤖 Méthode de génération: {generation_method}")

            if 'generated_response' in result:
                response_text = result['generated_response']
                if response_text:
                    print(f"💬 Réponse générée ({len(response_text)} caractères):")
                    print(f"   '{response_text[:150]}{'...' if len(response_text) > 150 else ''}'")
                else:
                    print("💬 Aucune réponse générée")

            if 'results' in result:
                print(f"📄 Documents trouvés: {result.get('count', 0)}")

            if 'error' in result:
                print(f"⚠️  Erreur rapportée: {result['error']}")

            return result
        else:
            print(f"❌ Échec: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"📋 Détails: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"📋 Réponse brute: {response.text}")
            return {}

    except requests.exceptions.Timeout:
        print("⏰ Timeout - Le serveur n'a pas répondu dans les temps")
        return {}
    except requests.exceptions.ConnectionError:
        print("🔌 Erreur de connexion - Vérifiez que le serveur Django est démarré")
        return {}
    except Exception as e:
        print(f"💥 Erreur inattendue: {str(e)}")
        return {}

def test_rag_query_endpoint():
    """Test de l'endpoint de requête RAG principal"""

    print("\n" + "="*60)
    print("🎯 TEST 1: Endpoint RAG Query (POST)")
    print("="*60)

    # Test avec différents modes
    test_cases = [
        {
            'name': 'Mode Similarité Seulement',
            'data': {
                'query': 'politique de sécurité informatique',
                'mode': 'similarity_only',
                'k': 3
            }
        },
        {
            'name': 'Mode RAG avec Génération (Gemini prioritaire)',
            'data': {
                'query': 'Quelles sont les principales exigences de sécurité ?',
                'mode': 'huggingface_rag',  # Nom conservé mais utilise maintenant Gemini en priorité
                'k': 5
            }
        },
        {
            'name': 'Mode Hybride (Similarité + Génération)',
            'data': {
                'query': 'procédures de sauvegarde et sécurité',
                'mode': 'hybrid',
                'k': 3
            }
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Test 1.{i}: {case['name']}")
        result = test_endpoint(f"{API_BASE}/query/", 'POST', case['data'])

        # Pause entre les tests
        if i < len(test_cases):
            time.sleep(2)

def test_rag_generate_endpoint():
    """Test de l'endpoint de génération RAG dédié"""

    print("\n" + "="*60)
    print("🎯 TEST 2: Endpoint RAG Generate (POST)")
    print("="*60)

    test_data = {
        'query': 'Comment mettre en place une politique de sauvegarde efficace ?',
        'max_new_tokens': 150,
        'use_custom_context': True
    }

    result = test_endpoint(f"{API_BASE}/generate/", 'POST', test_data)

def test_chat_completion_endpoint():
    """Test de l'endpoint de chat completion"""

    print("\n" + "="*60)
    print("🎯 TEST 3: Endpoint Chat Completions (POST)")
    print("="*60)

    test_data = {
        'messages': [
            {
                'role': 'user',
                'content': 'Peux-tu m\'expliquer les meilleures pratiques en matière de sécurité informatique ?'
            }
        ],
        'max_tokens': 200,
        'temperature': 0.7
    }

    result = test_endpoint(f"{API_BASE}/chat/completions/", 'POST', test_data)

def test_rag_query_get():
    """Test de l'endpoint RAG avec GET (rétrocompatibilité)"""

    print("\n" + "="*60)
    print("🎯 TEST 4: Endpoint RAG Query (GET)")
    print("="*60)

    params = {
        'q': 'politique sécurité',
        'mode': 'huggingface_rag',  # Utilise maintenant Gemini en priorité
        'k': 3
    }

    result = test_endpoint(f"{API_BASE}/query/", 'GET', params=params)

def test_model_info_endpoint():
    """Test de l'endpoint d'information sur les modèles"""

    print("\n" + "="*60)
    print("🎯 TEST 5: Endpoint Model Info (GET)")
    print("="*60)

    result = test_endpoint(f"{API_BASE}/models/info/", 'GET')

def check_server_status():
    """Vérifie si le serveur Django est accessible"""

    print("🔍 Vérification du statut du serveur...")

    try:
        response = requests.get(f"{BASE_URL}/admin/", timeout=5)
        if response.status_code in [200, 302]:  # 302 = redirection vers login
            print("✅ Serveur Django accessible")
            return True
        else:
            print(f"⚠️  Serveur répond mais statut inattendu: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Serveur Django non accessible")
        print(f"💡 Assurez-vous que le serveur est démarré sur {BASE_URL}")
        print("   Commande: python manage.py runserver")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {str(e)}")
        return False

def show_test_summary():
    """Affiche un résumé des tests et de la nouvelle logique"""

    print("\n" + "="*80)
    print("📋 RÉSUMÉ DES TESTS RAG - NOUVELLE LOGIQUE GEMINI -> HUGGING FACE")
    print("="*80)

    print("""
🎯 ENDPOINTS TESTÉS :

1. POST /api/v1/processing/rag/query/
   • Mode similarity_only: Recherche par similarité uniquement
   • Mode huggingface_rag: Gemini en priorité → Hugging Face en fallback
   • Mode hybrid: Similarité + (Gemini → Hugging Face)

2. POST /api/v1/processing/rag/generate/
   • Génération directe avec la nouvelle logique de priorité

3. POST /api/v1/processing/rag/chat/completions/
   • Chat style OpenAI avec backend RAG

4. GET /api/v1/processing/rag/query/
   • Version GET pour rétrocompatibilité

5. GET /api/v1/processing/rag/models/info/
   • Informations sur l'état des modèles

🔄 NOUVELLE LOGIQUE DE FALLBACK :

1. 🥇 GEMINI (Priorité)
   ├── ✅ Disponible + Clé API → Utilise Gemini
   └── ❌ Indisponible/Erreur → Fallback Hugging Face

2. 🥈 HUGGING FACE (Fallback)
   ├── ✅ Service disponible → Utilise Hugging Face
   └── ❌ Indisponible → Retourne erreur

📊 ATTENDU DANS LES RÉPONSES :
• mode: 'gemini_primary' ou 'huggingface_fallback'
• generation_method: 'gemini' ou 'huggingface_fallback'
• generated_response: La réponse générée
• results: Documents utilisés comme contexte
    """)

def main():
    """Fonction principale de test"""

    print("🚀 TESTS DES ENDPOINTS RAG - GEMINI EN PRIORITÉ")
    print("="*60)

    # Vérifier le serveur
    if not check_server_status():
        return

    # Exécuter tous les tests
    test_rag_query_endpoint()
    test_rag_generate_endpoint()
    test_chat_completion_endpoint()
    test_rag_query_get()
    test_model_info_endpoint()

    # Afficher le résumé
    show_test_summary()

    print("\n✅ Tests terminés")
    print("\n💡 Pour analyser les logs détaillés, vérifiez la console Django")

if __name__ == '__main__':
    main()