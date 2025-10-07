"""
Service RAG direct utilisant Gemini avec les données de la base de données
Priorité : Gemini + données BD (Topic, Section, Document.content)
"""

import logging
import os
from typing import Dict, Any, List, Optional
from django.utils import timezone
from django.conf import settings
from django.db.models import Q

logger = logging.getLogger(__name__)

# Import optionnel pour Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

from doctrine.models import Document, Topic, Section


class GeminiDatabaseRAGService:
    """
    Service RAG utilisant directement Gemini avec les données de la base de données
    Recherche dans : Topic, Section, Document (colonne content)
    """

    def __init__(self):
        self.api_key = None
        self.model = None
        self._initialized = False

    def ensure_ready(self):
        """Initialise le service Gemini si nécessaire"""
        if not GEMINI_AVAILABLE:
            logger.error("google.generativeai non disponible. Installez: pip install google-generativeai")
            return False

        if not self._initialized:
            try:
                self._initialize_gemini()
                return True
            except Exception as e:
                logger.error(f"Erreur initialisation Gemini: {e}")
                return False
        return True

    def _initialize_gemini(self):
        """Initialise Gemini"""
        self.api_key = getattr(settings, 'GOOGLE_AI_API_KEY', os.getenv('GOOGLE_AI_API_KEY'))
        if not self.api_key:
            raise Exception("GOOGLE_AI_API_KEY manquante")

        genai.configure(api_key=self.api_key)
        model_name = getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash')
        self.model = genai.GenerativeModel(model_name)
        self._initialized = True

        logger.info(f"Service Gemini + Database RAG initialise avec {model_name}")

    def query(self, question: str, k: int = 5, document_id: str = None, scope: str = 'all', **kwargs) -> Dict[str, Any]:
        """
        Interroge le système RAG avec Gemini et la base de données

        Args:
            question: La question à poser
            k: Nombre de documents à récupérer
            document_id: ID d'un document spécifique (optionnel)
            scope: Portée de recherche ('all', 'document')
            **kwargs: Arguments supplémentaires

        Returns:
            Dictionnaire avec la réponse et les métadonnées
        """
        start_time = timezone.now()

        if not self.ensure_ready():
            return self._error_response(question, start_time, "Service Gemini non disponible")

        try:
            # Étape 1: Rechercher dans la base de données
            context_data = self._search_database(question, k, document_id, scope)

            # Étape 2: Construire le contexte pour Gemini
            context_text = self._build_context_for_gemini(context_data)

            # Étape 3: Générer la réponse avec Gemini
            answer = self._generate_with_gemini(question, context_text)

            end_time = timezone.now()
            took_ms = int((end_time - start_time).total_seconds() * 1000)

            return {
                "query": question,
                "answer": answer,
                "context_documents": context_data,
                "count": len(context_data),
                "took_ms": took_ms,
                "mode": "gemini_database_rag",
                "scope": scope,
                "generation_metadata": {
                    "generation_method": "gemini_direct_with_database",
                    "model": getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash'),
                    "knowledge_source": "django_database_direct",
                    "tables_used": ["Document", "Topic", "Section"],
                    "filtered_by_document": document_id is not None,
                    "error": None
                }
            }

        except Exception as e:
            logger.exception(f"Erreur lors de la requête Gemini Database RAG: {e}")
            return self._error_response(question, start_time, str(e))

    def _search_database(self, query: str, k: int, document_id: str = None, scope: str = 'all') -> List[Dict[str, Any]]:
        """
        Recherche directe dans les tables de la base de données
        Priorité : Topic, Section, Document.content
        """
        logger.info(f"Recherche dans la BD pour: '{query}' (k={k}, scope={scope})")

        results = []
        query_terms = query.lower().split()

        # Construire les filtres de base
        base_filter = Q()
        if document_id and scope == 'document':
            base_filter = Q(document_id=document_id)
        else:
            base_filter = Q(document__is_deleted=False, document__status='PROCESSED', document__is_searchable=True)

        # 1. Rechercher dans les Topics
        try:
            topic_filter = base_filter
            for term in query_terms:
                topic_filter &= (Q(title__icontains=term) | Q(content__icontains=term))

            topics = Topic.objects.filter(topic_filter).select_related('document')[:k]

            for topic in topics:
                if len(results) >= k:
                    break

                # Extraire un extrait pertinent
                excerpt = self._extract_relevant_excerpt(topic.content or topic.title, query, 800)
                if excerpt:
                    results.append({
                        'text': excerpt,
                        'score': self._calculate_relevance_score(excerpt, query),
                        'source': 'topic',
                        'source_id': topic.id,
                        'source_title': topic.title,
                        'document': {
                            'id': topic.document.id,
                            'title': topic.document.title,
                            'file_name': topic.document.original_filename,
                            'upload_date': topic.document.upload_date.isoformat() if topic.document.upload_date else None,
                        },
                        'metadata': {
                            'table': 'Topic',
                            'topic_id': topic.id,
                            'topic_title': topic.title,
                            'order_index': getattr(topic, 'order_index', 0)
                        }
                    })
        except Exception as e:
            logger.warning(f"Erreur recherche Topics: {e}")

        # 2. Rechercher dans les Sections (si pas assez de résultats)
        if len(results) < k:
            try:
                section_filter = base_filter
                for term in query_terms:
                    section_filter &= (Q(title__icontains=term) | Q(content__icontains=term))

                sections = Section.objects.filter(section_filter).select_related('topic__document')[:k-len(results)]

                for section in sections:
                    if len(results) >= k:
                        break

                    excerpt = self._extract_relevant_excerpt(section.content or section.title, query, 800)
                    if excerpt:
                        results.append({
                            'text': excerpt,
                            'score': self._calculate_relevance_score(excerpt, query),
                            'source': 'section',
                            'source_id': section.id,
                            'source_title': section.title,
                            'document': {
                                'id': section.topic.document.id,
                                'title': section.topic.document.title,
                                'file_name': section.topic.document.original_filename,
                                'upload_date': section.topic.document.upload_date.isoformat() if section.topic.document.upload_date else None,
                            },
                            'metadata': {
                                'table': 'Section',
                                'section_id': section.id,
                                'section_title': section.title,
                                'topic_id': section.topic.id,
                                'topic_title': section.topic.title,
                                'order_index': getattr(section, 'order_index', 0)
                            }
                        })
            except Exception as e:
                logger.warning(f"Erreur recherche Sections: {e}")

        # 3. Rechercher dans Document.content (si pas assez de résultats)
        if len(results) < k:
            try:
                doc_filter = Q(is_deleted=False, status='PROCESSED', is_searchable=True)
                if document_id and scope == 'document':
                    doc_filter &= Q(id=document_id)

                for term in query_terms:
                    doc_filter &= Q(content__icontains=term)

                documents = Document.objects.filter(doc_filter)[:k-len(results)]

                for doc in documents:
                    if len(results) >= k:
                        break

                    if doc.content:
                        excerpt = self._extract_relevant_excerpt(doc.content, query, 800)
                        if excerpt:
                            results.append({
                                'text': excerpt,
                                'score': self._calculate_relevance_score(excerpt, query),
                                'source': 'document_content',
                                'source_id': doc.id,
                                'source_title': doc.title,
                                'document': {
                                    'id': doc.id,
                                    'title': doc.title,
                                    'file_name': doc.original_filename,
                                    'upload_date': doc.upload_date.isoformat() if doc.upload_date else None,
                                },
                                'metadata': {
                                    'table': 'Document',
                                    'document_id': doc.id,
                                    'content_length': len(doc.content) if doc.content else 0,
                                    'file_size': doc.file_size
                                }
                            })
            except Exception as e:
                logger.warning(f"Erreur recherche Documents: {e}")

        # Trier par score de pertinence
        results.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"Trouvé {len(results)} résultats dans la base de données")
        return results[:k]

    def _extract_relevant_excerpt(self, text: str, query: str, max_length: int = 800) -> str:
        """Extrait un extrait pertinent du texte basé sur la requête"""
        if not text:
            return ""

        text = str(text)
        query_terms = [term.lower() for term in query.split()]
        text_lower = text.lower()

        # Trouver la meilleure position
        best_position = 0
        max_matches = 0

        # Chercher la zone avec le plus de termes de recherche
        for i in range(0, len(text), 100):
            window = text_lower[i:i+max_length]
            matches = sum(1 for term in query_terms if term in window)
            if matches > max_matches:
                max_matches = matches
                best_position = i

        # Extraire l'extrait
        excerpt = text[best_position:best_position + max_length]

        # Nettoyer les débuts/fins
        if best_position > 0:
            excerpt = "..." + excerpt
        if len(text) > best_position + max_length:
            excerpt = excerpt + "..."

        return excerpt.strip()

    def _calculate_relevance_score(self, text: str, query: str) -> float:
        """Calcule un score de pertinence simple"""
        if not text:
            return 0.0

        query_terms = [term.lower() for term in query.split()]
        text_lower = text.lower()
        text_words = text_lower.split()

        score = 0.0
        for term in query_terms:
            # Score pour correspondance exacte
            exact_matches = text_lower.count(term)
            score += exact_matches * 2

            # Score pour correspondance partielle
            partial_matches = sum(1 for word in text_words if term in word)
            score += partial_matches * 0.5

        # Normaliser par la longueur du texte
        if text_words:
            score = score / len(text_words)

        return min(1.0, score * 10)  # Limiter à 1.0

    def _build_context_for_gemini(self, context_data: List[Dict[str, Any]]) -> str:
        """Construit le contexte textuel pour Gemini"""
        if not context_data:
            return ""

        context_parts = []
        for i, item in enumerate(context_data[:5]):  # Limiter à 5 résultats
            text = item.get('text', '')
            source = item.get('source', 'unknown')
            source_title = item.get('source_title', 'Sans titre')
            doc_title = item.get('document', {}).get('title', 'Document inconnu')

            if text:
                context_parts.append(
                    f"## Résultat {i+1} ({source})\n"
                    f"**Source:** {source_title}\n"
                    f"**Document:** {doc_title}\n"
                    f"**Contenu:** {text}\n"
                )

        return "\n".join(context_parts)

    def _generate_with_gemini(self, query: str, context: str) -> str:
        """Génère une réponse avec Gemini"""
        if not context:
            return "Je n'ai pas trouvé d'informations pertinentes dans la base de connaissances pour répondre à votre question."

        prompt = f"""En tant qu'assistant AI spécialisé dans l'analyse de documents, réponds à la question suivante en te basant uniquement sur le contexte fourni.

CONTEXTE:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Réponds en français
- Base ta réponse uniquement sur les informations du contexte fourni
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois précis et concis
- Cite les sources pertinentes (topics, sections, documents) quand c'est approprié
- Utilise maximum 3-4 phrases pour ta réponse principale

RÉPONSE:"""

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                return response.text
            else:
                return "Désolé, je n'ai pas pu générer une réponse pour cette question."
        except Exception as e:
            logger.error(f"Erreur génération Gemini: {e}")
            return f"Erreur lors de la génération de la réponse: {str(e)}"

    def _error_response(self, query: str, start_time, error_msg: str) -> Dict[str, Any]:
        """Retourne une réponse d'erreur formatée"""
        end_time = timezone.now()
        took_ms = int((end_time - start_time).total_seconds() * 1000)

        return {
            "query": query,
            "answer": f"Service temporairement indisponible: {error_msg}",
            "context_documents": [],
            "count": 0,
            "took_ms": took_ms,
            "mode": "gemini_database_rag_error",
            "generation_metadata": {
                "generation_method": "error",
                "error": error_msg
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du service"""
        # Statistiques de la base de données
        try:
            total_docs = Document.objects.filter(is_deleted=False, status='PROCESSED', is_searchable=True).count()
            total_topics = Topic.objects.filter(document__is_deleted=False, document__status='PROCESSED').count()
            total_sections = Section.objects.filter(topic__document__is_deleted=False, topic__document__status='PROCESSED').count()
            docs_with_content = Document.objects.filter(is_deleted=False, status='PROCESSED', is_searchable=True, content__isnull=False).exclude(content='').count()
        except Exception as e:
            logger.error(f"Erreur calcul stats: {e}")
            total_docs = total_topics = total_sections = docs_with_content = 0

        return {
            "initialized": self._initialized,
            "gemini_available": GEMINI_AVAILABLE,
            "api_key_configured": bool(self.api_key),
            "database_stats": {
                "total_documents": total_docs,
                "total_topics": total_topics,
                "total_sections": total_sections,
                "documents_with_content": docs_with_content
            }
        }


# Instance singleton
gemini_database_rag_service = GeminiDatabaseRAGService()