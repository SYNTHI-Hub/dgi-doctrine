#!/usr/bin/env python3
"""
Test de validation de la nouvelle logique de priorité :
Gemini en premier, Hugging Face en fallback
"""

import os
import sys

def test_priority_logic():
    """Test de la logique de priorité sans Django"""

    print("=== Test de la logique de priorité RAG ===")
    print("Gemini -> Hugging Face -> Erreur")

    # Simuler les conditions
    test_cases = [
        {
            'name': 'Gemini disponible avec clé API',
            'gemini_available': True,
            'api_key': 'test_key',
            'gemini_works': True,
            'expected': 'gemini_primary'
        },
        {
            'name': 'Gemini disponible mais erreur de génération',
            'gemini_available': True,
            'api_key': 'test_key',
            'gemini_works': False,
            'expected': 'huggingface_fallback'
        },
        {
            'name': 'Gemini disponible mais pas de clé API',
            'gemini_available': True,
            'api_key': None,
            'gemini_works': True,
            'expected': 'huggingface_fallback'
        },
        {
            'name': 'Gemini non disponible',
            'gemini_available': False,
            'api_key': 'test_key',
            'gemini_works': True,
            'expected': 'huggingface_fallback'
        }
    ]

    print("\n📋 Scénarios de test :")
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   📍 Conditions :")
        print(f"      - Gemini disponible: {case['gemini_available']}")
        print(f"      - Clé API: {'✅' if case['api_key'] else '❌'}")
        print(f"      - Gemini fonctionne: {case['gemini_works']}")

        # Simuler la logique de décision
        result = simulate_rag_decision(
            gemini_available=case['gemini_available'],
            api_key=case['api_key'],
            gemini_works=case['gemini_works']
        )

        success = result == case['expected']
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   🎯 Attendu: {case['expected']}")
        print(f"   🔄 Résultat: {result}")
        print(f"   📊 Status: {status}")

def simulate_rag_decision(gemini_available, api_key, gemini_works):
    """
    Simule la logique de décision RAG selon la nouvelle priorité
    """

    # Étape 1: Vérifier si Gemini est disponible
    if gemini_available:
        # Étape 2: Vérifier la clé API
        if api_key:
            # Étape 3: Tenter d'utiliser Gemini
            if gemini_works:
                return 'gemini_primary'
            else:
                # Erreur Gemini -> fallback Hugging Face
                return 'huggingface_fallback'
        else:
            # Pas de clé API -> fallback Hugging Face
            return 'huggingface_fallback'
    else:
        # Gemini non disponible -> fallback Hugging Face
        return 'huggingface_fallback'

def test_mode_behaviors():
    """Test du comportement des différents modes"""

    print("\n\n=== Test des modes RAG ===")

    modes_info = {
        'SIMILARITY_ONLY': {
            'description': 'Recherche par similarité seulement',
            'uses_generation': False,
            'fallback_logic': 'Aucun - mode indépendant'
        },
        'HUGGINGFACE_RAG': {
            'description': 'Génération RAG (maintenant avec Gemini en priorité)',
            'uses_generation': True,
            'fallback_logic': 'Gemini -> Hugging Face -> Erreur'
        },
        'HYBRID': {
            'description': 'Similarité + Génération',
            'uses_generation': True,
            'fallback_logic': 'Similarité + (Gemini -> Hugging Face)'
        }
    }

    for mode, info in modes_info.items():
        print(f"\n🔧 Mode: {mode}")
        print(f"   📝 Description: {info['description']}")
        print(f"   🤖 Utilise génération: {info['uses_generation']}")
        print(f"   🔄 Logique fallback: {info['fallback_logic']}")

def show_new_architecture():
    """Affiche la nouvelle architecture"""

    print("\n\n=== Nouvelle Architecture RAG ===")

    print("""
🏗️  ARCHITECTURE RÉVISÉE :

┌─────────────────────────┐
│   Requête utilisateur   │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│    Mode sélectionné     │
├─────────┬───────────────┤
│ SIMILARITY_ONLY │ -> Recherche embeddings uniquement
│ HUGGINGFACE_RAG │ -> Gemini (1er) -> HF (fallback)
│ HYBRID          │ -> Similarité + Génération priorité Gemini
└─────────┬───────────────┘
          │
          ▼ (si génération nécessaire)
┌─────────────────────────┐
│  🥇 PRIORITÉ : GEMINI   │
├─────────────────────────┤
│ 1. Vérifier disponibilité Gemini
│ 2. Vérifier clé API Google
│ 3. Rechercher documents (document_content)
│ 4. Générer réponse avec Gemini
└─────────┬───────────────┘
          │ (si erreur)
          ▼
┌─────────────────────────┐
│ 🥈 FALLBACK : HUGGING   │
├─────────────────────────┤
│ 1. Charger service HF
│ 2. Utiliser retriever HF
│ 3. Générer avec modèle HF
└─────────┬───────────────┘
          │ (si erreur)
          ▼
┌─────────────────────────┐
│      ❌ ERREUR          │
└─────────────────────────┘

🔑 AVANTAGES :
• Gemini plus rapide et accessible
• Meilleure utilisation base de données locale
• Fallback robuste vers HF si nécessaire
• Logs détaillés du chemin pris
    """)

if __name__ == '__main__':
    test_priority_logic()
    test_mode_behaviors()
    show_new_architecture()

    print("\n✅ Tests de validation terminés")