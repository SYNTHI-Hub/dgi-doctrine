"""
Service RAG optimisé pour tous les PDFs du dossier media
Lit tous les PDFs du dossier media, utilise recherche textuelle simple et API Gemini directe.
Priorité : Chargement PDFs → Splitting → Recherche textuelle → Génération avec API Gemini.
"""

import logging
import os
import requests
import glob
from typing import Dict, Any, List, Optional
from django.utils import timezone
from django.conf import settings

logger = logging.getLogger(__name__)

# Imports minimaux pour PDF uniquement
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    PDF_LOADERS_AVAILABLE = True
except ImportError:
    PyPDFLoader = None
    RecursiveCharacterTextSplitter = None
    PDF_LOADERS_AVAILABLE = False


class PDFRAGService:
    """
    Service RAG optimisé pour tous les PDFs du dossier media.
    Charge tous les PDFs, utilise recherche textuelle simple et API Gemini directe.
    """

    def __init__(self, media_folder: str = None):
        self.media_folder = media_folder or getattr(settings, 'MEDIA_ROOT', 'media')
        self.gemini_api_key = None
        self.text_splitter = None
        self.pdf_chunks = []  # Cache des chunks de tous les PDFs
        self._initialized = False

    def ensure_ready(self):
        """Initialise le service si nécessaire"""
        if not PDF_LOADERS_AVAILABLE:
            logger.error("PyPDFLoader non disponible. Installez: pip install langchain-community")
            return False

        if not self._initialized:
            try:
                logger.info(f"Initialisation du service PDF RAG avec media_folder: {self.media_folder}")
                self._initialize_gemini_api()
                self._initialize_text_splitter()
                self._load_all_pdfs()
                self._initialized = True
                logger.info(f"Service PDF RAG initialisé avec {len(self.pdf_chunks)} chunks depuis {self.media_folder}")
                return True
            except Exception as e:
                logger.error(f"Erreur initialisation PDF RAG: {e}")
                return False
        return True

    def _initialize_gemini_api(self):
        """Initialise l'API Gemini directement"""
        self.gemini_api_key = getattr(settings, 'GOOGLE_AI_API_KEY', os.getenv('GOOGLE_AI_API_KEY'))
        if not self.gemini_api_key:
            raise Exception("GOOGLE_AI_API_KEY manquante")
        logger.info("API Gemini configurée")

    def _initialize_text_splitter(self):
        """Initialise le text splitter"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        logger.info("Text splitter initialisé")

    def _load_all_pdfs(self):
        """Charge tous les PDFs du dossier media et les divise en chunks"""
        logger.info(f"Recherche de PDFs dans: {self.media_folder}")
        logger.info(f"Chemin absolu: {os.path.abspath(self.media_folder)}")

        if not os.path.exists(self.media_folder):
            raise Exception(f"Dossier media introuvable: {self.media_folder}")

        logger.info(f"Chargement des PDFs depuis: {self.media_folder}")

        self.pdf_chunks = []
        pdf_files = glob.glob(os.path.join(self.media_folder, "**/*.pdf"), recursive=True)

        logger.info(f"Pattern de recherche: {os.path.join(self.media_folder, '**/*.pdf')}")
        logger.info(f"Fichiers trouvés: {pdf_files}")

        if not pdf_files:
            logger.warning(f"Aucun fichier PDF trouvé dans {self.media_folder}")
            return

        logger.info(f"Trouvé {len(pdf_files)} fichiers PDF")

        for pdf_file in pdf_files:
            try:
                logger.info(f"Chargement de: {pdf_file}")
                loader = PyPDFLoader(pdf_file)
                pages = []
                for page in loader.lazy_load():
                    pages.append(page)

                logger.info(f"{len(pages)} pages chargées depuis {os.path.basename(pdf_file)}")

                if pages:
                    # Diviser les pages en chunks
                    all_splits = self.text_splitter.split_documents(pages)

                    for split in all_splits:
                        split.metadata['file_path'] = pdf_file
                        split.metadata['file_name'] = os.path.basename(pdf_file)
                        self.pdf_chunks.append(split)

                    logger.info(f"Chargé {len(all_splits)} chunks depuis {os.path.basename(pdf_file)}")

            except Exception as e:
                logger.error(f"Erreur lors du chargement de {pdf_file}: {e}")
                continue

        logger.info(f"Total: {len(self.pdf_chunks)} chunks chargés depuis {len(pdf_files)} fichiers PDF")


    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Interroge le système RAG sur tous les PDFs

        Args:
            question: La question à poser
            k: Nombre de chunks à récupérer

        Returns:
            Dictionnaire avec la réponse et les métadonnées
        """
        start_time = timezone.now()

        if not self.ensure_ready():
            return self._error_response(question, start_time, "Service PDF RAG non disponible")

        try:
            # Étape 1: Recherche textuelle dans les chunks
            context_data = self._search_pdf_chunks(question, k)

            # Étape 2: Construire le contexte pour Gemini
            context_text = self._build_context_for_gemini(context_data)

            # Étape 3: Générer la réponse avec API Gemini directe
            answer = self._generate_with_gemini_api(question, context_text)

            end_time = timezone.now()
            took_ms = int((end_time - start_time).total_seconds() * 1000)

            return {
                "query": question,
                "answer": answer,
                "context_documents": context_data,
                "count": len(context_data),
                "took_ms": took_ms,
                "mode": "pdf_gemini_rag_optimized",
                "generation_metadata": {
                    "generation_method": "gemini_api_direct_with_pdf_chunks",
                    "knowledge_source": "pdf_media_folder_chunks",
                    "search_method": "textual_search",
                    "error": None
                }
            }

        except Exception as e:
            logger.exception(f"Erreur lors de la requête PDF RAG: {e}")
            return self._error_response(question, start_time, str(e))

    def _search_pdf_chunks(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Recherche dans les chunks de PDFs avec recherche textuelle simple
        """
        logger.info(f"Recherche textuelle dans {len(self.pdf_chunks)} chunks pour: '{query}'")

        if not self.pdf_chunks:
            logger.warning("Aucun chunk PDF disponible")
            return []

        results = []
        query_terms = [term.lower() for term in query.split() if len(term) > 2]

        for chunk in self.pdf_chunks:
            content = chunk.page_content
            if content and len(content.strip()) > 10:
                score = self._calculate_relevance_score(content, query_terms)
                if score > 0:
                    results.append({
                        'text': content,
                        'score': score,
                        'source': 'pdf_chunk',
                        'source_id': f"{chunk.metadata.get('file_name', 'unknown')}_page_{chunk.metadata.get('page', 0)}",
                        'source_title': chunk.metadata.get('file_name', 'PDF inconnu'),
                        'document': {
                            'file_name': chunk.metadata.get('file_name', 'N/A'),
                            'file_path': chunk.metadata.get('file_path', 'N/A'),
                            'page': chunk.metadata.get('page', 0),
                            'start_index': chunk.metadata.get('start_index', 0)
                        },
                        'metadata': {
                            'source_type': 'pdf_chunk',
                            'file_name': chunk.metadata.get('file_name', 'N/A'),
                            'page_number': chunk.metadata.get('page', 0),
                            'chunk_length': len(content)
                        }
                    })

        # Trier par score de pertinence
        results.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"Trouvé {len(results)} résultats pertinents dans les PDFs")
        return results[:k]

    def _calculate_relevance_score(self, text: str, query_terms: List[str]) -> float:
        """Calcule un score de pertinence simple basé sur la fréquence des termes"""
        if not text or not query_terms:
            return 0.0

        text_lower = text.lower()
        text_words = text_lower.split()

        if not text_words:
            return 0.0

        score = 0.0
        for term in query_terms:
            # Score pour correspondance exacte du terme
            exact_count = text_lower.count(term)
            score += exact_count * 2.0

            # Score pour correspondance partielle
            partial_count = sum(1 for word in text_words if term in word and term != word)
            score += partial_count * 0.5

        # Normaliser par la longueur du texte
        normalized_score = score / len(text_words)

        # Bonus si le terme apparaît tôt dans le texte
        first_paragraph = ' '.join(text_words[:50])
        early_matches = sum(1 for term in query_terms if term in first_paragraph.lower())
        bonus = early_matches * 0.1

        return min(1.0, normalized_score + bonus)

    def _generate_with_gemini_api(self, query: str, context: str) -> str:
        """Génère une réponse avec l'API Gemini directe"""
        if not context:
            return "Je n'ai pas trouvé d'informations pertinentes dans les documents PDF pour répondre à votre question."

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

        prompt = f"""En tant qu'assistant AI spécialisé dans l'analyse de documents PDF, réponds à la question suivante en te basant uniquement sur le contexte fourni.

CONTEXTE DISPONIBLE:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Réponds en français
- Base ta réponse uniquement sur les informations du contexte fourni
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois précis et concis
- Cite les fichiers et pages pertinentes quand c'est approprié
- Utilise maximum 3-4 phrases pour ta réponse principale

RÉPONSE:"""

        headers = {
            'Content-Type': 'application/json',
        }

        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }

        try:
            response = requests.post(
                f"{url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0].get('content', {})
                    parts = content.get('parts', [])
                    if parts and 'text' in parts[0]:
                        return parts[0]['text']

                return "Désolé, je n'ai pas pu générer une réponse valide."
            else:
                logger.error(f"Erreur API Gemini: {response.status_code} - {response.text}")
                return f"Erreur lors de la génération de la réponse (Code: {response.status_code})"

        except requests.exceptions.Timeout:
            logger.error("Timeout lors de l'appel à l'API Gemini")
            return "Délai d'attente dépassé lors de la génération de la réponse."
        except Exception as e:
            logger.error(f"Erreur lors de l'appel API Gemini: {e}")
            return f"Erreur technique lors de la génération: {str(e)}"

    def _build_context_for_gemini(self, context_data: List[Dict[str, Any]]) -> str:
        """Construit le contexte textuel pour Gemini"""
        if not context_data:
            return ""

        context_parts = []
        for i, item in enumerate(context_data[:5]):  # Limiter à 5 résultats
            text = item.get('text', '')
            source = item.get('source', 'unknown')
            source_title = item.get('source_title', 'Sans titre')
            doc_info = item.get('document', {})
            file_name = doc_info.get('file_name', 'Document inconnu')
            page = doc_info.get('page', 0)
            score = item.get('score', 0)

            if text:
                context_parts.append(
                    f"## Source {i+1} ({source}) - Score: {score:.2f}\n"
                    f"**Fichier:** {file_name}\n"
                    f"**Page:** {page}\n"
                    f"**Contenu:** {text}\n"
                )

        return "\n".join(context_parts)

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
            "mode": "pdf_rag_error",
            "generation_metadata": {
                "generation_method": "error",
                "error": error_msg
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du service"""
        try:
            if self.media_folder and os.path.exists(self.media_folder):
                pdf_files = glob.glob(os.path.join(self.media_folder, "**/*.pdf"), recursive=True)
                total_pdf_files = len(pdf_files)
                total_chunks = len(self.pdf_chunks)
            else:
                total_pdf_files = 0
                total_chunks = 0
        except Exception as e:
            logger.error(f"Erreur calcul stats PDFs: {e}")
            total_pdf_files = 0
            total_chunks = 0

        return {
            "initialized": self._initialized,
            "pdf_loaders_available": PDF_LOADERS_AVAILABLE,
            "gemini_api_configured": bool(self.gemini_api_key),
            "search_method": "textual_search_in_pdf_chunks",
            "generation_method": "gemini_api_rest",
            "media_folder": self.media_folder,
            "pdf_stats": {
                "total_pdf_files": total_pdf_files,
                "total_chunks_loaded": total_chunks,
                "chunks_cache_ready": len(self.pdf_chunks) > 0
            }
        }


# Instance singleton pour utilisation directe
rag_service = PDFRAGService()