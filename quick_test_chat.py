#!/usr/bin/env python3
"""
Test rapide de l'endpoint de chat avec la nouvelle logique Gemini->Hugging Face
"""

import requests
import json

def test_chat_endpoint():
    """Test rapide du chat endpoint"""

    url = "http://localhost:8000/api/v1/processing/rag/query/"

    # Test simple
    data = {
        "query": "Quelles sont les principales mesures de sécurité informatique ?",
        "mode": "huggingface_rag",  # Utilise maintenant Gemini en priorité
        "k": 3
    }

    headers = {
        'Content-Type': 'application/json'
    }

    print("🧪 Test rapide - Endpoint RAG Query")
    print(f"📤 Requête: {data['query']}")
    print(f"🔧 Mode: {data['mode']} (Gemini prioritaire)")

    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)

        print(f"📊 Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # Analyser la réponse
            mode = result.get('mode', 'unknown')
            method = result.get('generation_metadata', {}).get('generation_method', 'unknown')
            response_text = result.get('generated_response', '')
            count = result.get('count', 0)
            error = result.get('error')

            print(f"✅ Mode utilisé: {mode}")
            print(f"🤖 Méthode de génération: {method}")
            print(f"📄 Documents trouvés: {count}")

            if error:
                print(f"⚠️  Erreur: {error}")

            if response_text:
                print(f"💬 Réponse ({len(response_text)} caractères):")
                print(f"   {response_text}")
            else:
                print("💬 Aucune réponse générée")

            # Vérifier que la nouvelle logique fonctionne
            if mode == 'gemini_primary':
                print("✅ SUCCESS: Gemini utilisé en priorité !")
            elif mode == 'huggingface_fallback':
                print("✅ SUCCESS: Fallback vers Hugging Face activé")
            else:
                print(f"❓ Mode inattendu: {mode}")

        else:
            print(f"❌ Erreur: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Détails: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Réponse: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter au serveur")
        print("💡 Assurez-vous que Django est démarré: python manage.py runserver")
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")

if __name__ == '__main__':
    test_chat_endpoint()