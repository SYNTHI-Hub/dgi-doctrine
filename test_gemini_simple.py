#!/usr/bin/env python3
"""
Test simple de Gemini pour valider l'implémentation
"""

import os

try:
    import google.generativeai as genai
    print("✅ Google Generative AI disponible")
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"❌ Google Generative AI non disponible: {e}")
    GEMINI_AVAILABLE = False

def test_gemini_simple():
    """Test simple de Gemini"""

    if not GEMINI_AVAILABLE:
        print("❌ Gemini non disponible")
        return False

    # Récupérer la clé API
    api_key = os.getenv('GOOGLE_AI_API_KEY', 'AIzaSyDozWBNuhVS_zHE6ZeQ-NgVODQCwmsOCbM')

    if not api_key or api_key == '':
        print("❌ Clé API Google manquante")
        return False

    try:
        # Configuration de Gemini
        genai.configure(api_key=api_key)

        # Test simple
        model = genai.GenerativeModel('gemini-pro')

        # Test de génération simple
        context = """
        Document 1: Politique de sécurité informatique
        Cette politique définit les règles de sécurité pour protéger les données et systèmes.

        Document 2: Procédures de sauvegarde
        Les sauvegardes doivent être effectuées quotidiennement selon les procédures établies.
        """

        query = "Quelles sont les exigences de sécurité ?"

        prompt = f"""En tant qu'assistant AI, réponds à la question suivante en te basant sur le contexte fourni.

Contexte:
{context}

Question: {query}

Instructions:
- Réponds en français
- Base ta réponse uniquement sur les informations du contexte
- Sois précis et concis

Réponse:"""

        print("🧪 Test de génération Gemini...")
        response = model.generate_content(prompt)

        if response.text:
            print(f"✅ Réponse générée ({len(response.text)} caractères):")
            print(f"'{response.text[:200]}{'...' if len(response.text) > 200 else ''}'")
            return True
        else:
            print("❌ Aucune réponse générée")
            return False

    except Exception as e:
        print(f"❌ Erreur lors du test Gemini: {str(e)}")
        return False

def test_text_search_simulation():
    """Simule la recherche textuelle dans document_content"""

    print("\n🧪 Test de simulation de recherche textuelle...")

    # Simuler des contenus de documents
    mock_documents = [
        {
            'id': 1,
            'clean_content': 'Politique de sécurité informatique. Cette politique définit les règles pour protéger les données sensibles et assurer la sécurité des systèmes.',
            'document': {
                'id': 1,
                'title': 'Politique de sécurité',
                'original_filename': 'security_policy.pdf'
            }
        },
        {
            'id': 2,
            'clean_content': 'Procédures de sauvegarde. Les sauvegardes doivent être effectuées quotidiennement pour assurer la continuité des services.',
            'document': {
                'id': 2,
                'title': 'Procédures de sauvegarde',
                'original_filename': 'backup_procedures.pdf'
            }
        }
    ]

    query = "politique de sécurité"
    query_terms = query.lower().split()

    # Simuler la recherche
    results = []
    for doc in mock_documents:
        content_lower = doc['clean_content'].lower()

        # Vérifier si un terme de recherche est présent
        found = any(term in content_lower for term in query_terms)

        if found:
            # Extraire un extrait pertinent
            excerpt = extract_relevant_excerpt(doc['clean_content'], query)
            score = calculate_text_relevance(excerpt, query)

            result = {
                'text': excerpt,
                'score': score,
                'source': 'document_content',
                'document': doc['document']
            }
            results.append(result)

    print(f"✅ Recherche simulée: {len(results)} résultats trouvés")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.3f} - {result['document']['title']}")
        print(f"     Extrait: '{result['text'][:100]}....'")

    return results

def extract_relevant_excerpt(text, query, max_length=200):
    """Extrait un extrait pertinent du texte"""
    if not text:
        return ""

    query_terms = [term.lower() for term in query.split()]
    text_lower = text.lower()

    # Trouver la première occurrence d'un terme
    best_position = 0
    for term in query_terms:
        pos = text_lower.find(term)
        if pos != -1:
            best_position = max(0, pos - 50)
            break

    # Extraire l'extrait
    excerpt = text[best_position:best_position + max_length]

    if best_position > 0:
        excerpt = "..." + excerpt
    if len(text) > best_position + max_length:
        excerpt = excerpt + "..."

    return excerpt.strip()

def calculate_text_relevance(text, query):
    """Calcule un score de pertinence simple"""
    query_terms = [term.lower() for term in query.split()]
    text_lower = text.lower()

    score = 0.0
    text_words = text_lower.split()

    for term in query_terms:
        count = text_lower.count(term)
        score += count / len(text_words) if text_words else 0

    return min(1.0, score * 10)

if __name__ == '__main__':
    print("=== Test du système RAG avec Gemini en priorité et Hugging Face en fallback ===")

    # Test 1: Disponibilité de Gemini
    print("\n1. Test de disponibilité de Gemini...")
    gemini_ok = test_gemini_simple()

    # Test 2: Simulation de recherche textuelle
    print("\n2. Test de recherche textuelle...")
    search_results = test_text_search_simulation()

    # Test 3: Combinaison des deux (si Gemini disponible)
    if gemini_ok and search_results:
        print("\n3. Test de combinaison recherche + génération...")

        # Construire le contexte
        context_parts = []
        for i, result in enumerate(search_results[:2]):
            doc_title = result['document']['title']
            text = result['text']
            context_parts.append(f"## Document {i+1}: {doc_title}\n{text}")

        context = "\n\n".join(context_parts)
        query = "Quelles sont les principales exigences de sécurité ?"

        # Test avec Gemini
        try:
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""En tant qu'assistant AI, réponds à la question suivante en te basant sur le contexte fourni.

Contexte:
{context}

Question: {query}

Instructions:
- Réponds en français
- Base ta réponse uniquement sur les informations du contexte
- Sois précis et concis

Réponse:"""

            response = model.generate_content(prompt)

            if response.text:
                print(f"✅ Réponse combinée générée:")
                print(f"'{response.text}'")
            else:
                print("❌ Aucune réponse combinée générée")

        except Exception as e:
            print(f"❌ Erreur test combiné: {str(e)}")

    print("\n=== Fin des tests ===")