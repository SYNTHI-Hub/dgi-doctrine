import logging
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import os

from django.utils import timezone
from django.db.models import QuerySet
from django.conf import settings

from doctrine.models import Document, DocumentContent, Section, Paragraph

logger = logging.getLogger(__name__)


class RAGMode(Enum):
    """Modes de fonctionnement du service RAG"""
    SIMILARITY_ONLY = "similarity_only"  # Similarité simple avec embeddings
    HUGGINGFACE_RAG = "huggingface_rag"  # RAG complet avec génération (maintenant LangChain+Gemini)
    HYBRID = "hybrid"
    LANGCHAIN_RAG = "langchain_rag"  # RAG avec LangChain + Gemini

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

# Import optionnel pour éviter les erreurs de compatibilité
SentenceTransformer = None

# Import optionnel pour Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False


class RAGDependenciesMissing(Exception):
    pass


class RAGRetriever:
    """
    Minimal in-app retriever using sentence-transformers and NumPy for cosine similarity.
    No DB schema change; computes on the fly from stored structures.
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._model = None

    def ensure_ready(self):
        if np is None or SentenceTransformer is None:
            raise RAGDependenciesMissing(
                "RAG dependencies missing. Please install: pip install sentence-transformers numpy"
            )
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def _chunk_document(self, document: Document) -> List[Dict[str, Any]]:
        """
        Build chunks from Paragraphs if available, otherwise from clean_content using
        a simple sliding window over lines.
        Returns list of dicts with text and metadata references.
        """
        chunks: List[Dict[str, Any]] = []

        # Prefer paragraphs if present
        paragraphs_qs: QuerySet[Paragraph] = Paragraph.objects.filter(
            section__topic__document=document
        ).select_related('section').order_by('section__order_index', 'order_index')

        if paragraphs_qs.exists():
            for p in paragraphs_qs:
                text = (p.content or '').strip()
                if not text:
                    continue
                chunks.append({
                    'text': text,
                    'document': document,
                    'section': p.section,
                    'paragraph': p,
                    'source': 'paragraph'
                })
            if chunks:
                return chunks

        # Fallback to clean_content
        if hasattr(document, 'content') and document.content and document.content.clean_content:
            clean = document.content.clean_content
            lines = [ln.strip() for ln in clean.split('\n') if ln.strip()]
            if not lines:
                return chunks
            window_size = 5  # lines per chunk
            step = 4
            for i in range(0, len(lines), step):
                window = lines[i:i + window_size]
                if not window:
                    continue
                text = ' '.join(window).strip()
                if text:
                    chunks.append({
                        'text': text,
                        'document': document,
                        'section': None,
                        'paragraph': None,
                        'source': 'content_window'
                    })
        return chunks

    def _encode(self, texts: List[str]) -> 'np.ndarray':  # type: ignore[name-defined]
        self.ensure_ready()
        # SentenceTransformer returns list; cast to numpy array
        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embeddings, dtype='float32')

    def _cosine_sim(self, a: 'np.ndarray', b: 'np.ndarray') -> 'np.ndarray':  # type: ignore[name-defined]
        # a: (1, d) or (n, d), b: (m, d) -> returns (n, m)
        return a @ b.T

    def retrieve(self, query: str, *, k: int = 5, scope: str = 'all',
                 document: Optional[Document] = None, mode: RAGMode = RAGMode.SIMILARITY_ONLY) -> Dict[str, Any]:
        """
        Retrieve top-k relevant chunks for a query with multiple modes.

        Args:
            query: The search query
            k: Number of results to return
            scope: 'all' or 'document'
            document: Specific document to search in (if scope='document')
            mode: RAG mode to use

        Returns:
            Dictionary suitable for JSON response
        """
        started_at = timezone.now()
        query = (query or '').strip()
        if not query:
            return {
                'query': query,
                'results': [],
                'count': 0,
                'took_ms': 0,
                'mode': mode.value,
                'error': 'Empty query'
            }

        if scope not in ('all', 'document'):
            scope = 'all'

        # Route vers la méthode appropriée selon le mode
        if mode == RAGMode.HUGGINGFACE_RAG or mode == RAGMode.LANGCHAIN_RAG:
            return self._retrieve_with_generation(query, k, scope, document, started_at)
        elif mode == RAGMode.HYBRID:
            return self._retrieve_hybrid(query, k, scope, document, started_at)
        else:
            return self._retrieve_similarity_only(query, k, scope, document, started_at)

    def _retrieve_similarity_only(self, query: str, k: int, scope: str,
                                  document: Optional[Document], started_at) -> Dict[str, Any]:
        """Mode similarité uniquement (comportement original)"""
        # Collect candidate documents
        if scope == 'document' and document is not None:
            documents = [document]
        else:
            documents = list(
                Document.objects.filter(
                    is_deleted=False,
                    status=Document.StatusChoices.PROCESSED,
                    is_searchable=True
                ).select_related('content').order_by('-updated_at')[:500]  # safety cap
            )

        # Build chunks
        all_chunks: List[Dict[str, Any]] = []
        for doc in documents:
            all_chunks.extend(self._chunk_document(doc))

        if not all_chunks:
            return {
                'query': query,
                'results': [],
                'count': 0,
                'took_ms': int((timezone.now() - started_at).total_seconds() * 1000),
                'mode': RAGMode.SIMILARITY_ONLY.value
            }

        # Embed
        try:
            q_emb = self._encode([query])  # shape (1, d)
            doc_embs = self._encode([c['text'] for c in all_chunks])  # (n, d)
        except RAGDependenciesMissing as dep_err:
            return {
                'query': query,
                'results': [],
                'count': 0,
                'took_ms': int((timezone.now() - started_at).total_seconds() * 1000),
                'mode': RAGMode.SIMILARITY_ONLY.value,
                'error': str(dep_err)
            }
        except Exception as e:
            logger.exception('RAG embedding error: %s', e)
            return {
                'query': query,
                'results': [],
                'count': 0,
                'took_ms': int((timezone.now() - started_at).total_seconds() * 1000),
                'mode': RAGMode.SIMILARITY_ONLY.value,
                'error': 'Embedding error'
            }

        # Similarity and top-k
        sims = self._cosine_sim(q_emb, doc_embs)[0]  # (n,)
        top_idx = sims.argsort()[::-1][:max(1, int(k))]

        results = []
        for idx in top_idx:
            c = all_chunks[int(idx)]
            score = float(sims[int(idx)])
            doc = c['document']
            section = c.get('section')
            paragraph = c.get('paragraph')

            result_item: Dict[str, Any] = {
                'text': c['text'],
                'score': round(score, 6),
                'source': c.get('source', 'paragraph'),
                'document': {
                    'id': doc.id,
                    'title': doc.title,
                    'file_name': doc.file_name,
                    'upload_date': doc.upload_date.isoformat() if doc.upload_date else None,
                    'file_size': doc.file_size,
                    'page_count': doc.page_count
                },
            }

            if section is not None:
                result_item['section'] = {
                    'id': section.id,
                    'title': section.title,
                    'content': section.content,
                    'order': section.order
                }
            if paragraph is not None:
                result_item['paragraph'] = {
                    'id': paragraph.id,
                    'content': paragraph.content,
                    'order': paragraph.order
                }

            results.append(result_item)

        took_ms = int((timezone.now() - started_at).total_seconds() * 1000)
        return {
            'query': query,
            'scope': scope,
            'k': k,
            'count': len(results),
            'took_ms': took_ms,
            'mode': RAGMode.SIMILARITY_ONLY.value,
            'results': results
        }

    def _retrieve_with_generation(self, query: str, k: int, scope: str,
                                  document: Optional[Document], started_at) -> Dict[str, Any]:
        """Mode RAG avec génération LangChain+Gemini en priorité et fallback vers Hugging Face"""
        # Essayer d'abord LangChain + Gemini
        logger.info('Tentative de génération avec LangChain + Gemini en priorité')

        try:
            from .langchain_rag import langchain_rag_service

            # Utiliser le service LangChain RAG
            result = langchain_rag_service.query(query, k=k)

            # Adapter le format de retour pour la compatibilité
            took_ms = int((timezone.now() - started_at).total_seconds() * 1000)

            return {
                'query': query,
                'scope': scope,
                'k': k,
                'count': result.get('count', 0),
                'took_ms': took_ms,
                'mode': 'langchain_gemini',
                'results': self._format_langchain_results(result.get('context_documents', [])),
                'generated_response': result.get('answer', ''),
                'generation_metadata': {
                    'generation_took_ms': result.get('took_ms', 0),
                    'generation_method': 'langchain_gemini',
                    'error': None
                }
            }

        except ImportError:
            logger.warning('Service LangChain RAG non disponible, tentative avec Gemini direct')
            return self._try_direct_gemini_fallback(query, k, scope, document, started_at)
        except Exception as e:
            logger.warning('Erreur LangChain RAG: %s, tentative avec Gemini direct', e)
            return self._try_direct_gemini_fallback(query, k, scope, document, started_at)

    def _format_langchain_results(self, context_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formate les résultats LangChain pour la compatibilité avec l'API existante
        """
        results = []
        for doc_data in context_documents:
            content = doc_data.get('content', '')
            metadata = doc_data.get('metadata', {})

            result_item = {
                'text': content,
                'score': doc_data.get('score', 1.0),
                'source': 'langchain_db',
                'document': {
                    'id': metadata.get('document_id', ''),
                    'title': metadata.get('document_title', ''),
                    'file_name': metadata.get('document_filename', ''),
                    'upload_date': metadata.get('upload_date'),
                    'file_size': metadata.get('file_size', 0),
                    'page_count': 0  # Non disponible dans DocumentContent
                },
                'content_metadata': {
                    'word_count': metadata.get('word_count', 0),
                    'extraction_method': metadata.get('extraction_method', ''),
                    'processing_status': metadata.get('processing_status', ''),
                    'category': metadata.get('document_category'),
                    'theme': metadata.get('document_theme'),
                    'is_chunk': metadata.get('is_chunk', False),
                    'chunk_index': metadata.get('chunk_index'),
                    'source': metadata.get('source', 'langchain')
                }
            }
            results.append(result_item)

        return results

    def _try_direct_gemini_fallback(self, query: str, k: int, scope: str,
                                   document: Optional[Document], started_at) -> Dict[str, Any]:
        """Fallback vers Gemini direct puis Hugging Face"""
        if GEMINI_AVAILABLE:
            api_key = getattr(settings, 'GOOGLE_AI_API_KEY', os.getenv('GOOGLE_AI_API_KEY'))
            if api_key:
                try:
                    return self._generate_with_gemini_primary(query, k, scope, document, started_at)
                except Exception as e:
                    logger.warning('Erreur Gemini direct: %s, fallback vers Hugging Face', e)
                    return self._fallback_to_huggingface(query, k, scope, document, started_at)
            else:
                logger.warning('Clé API Google manquante, fallback vers Hugging Face')
                return self._fallback_to_huggingface(query, k, scope, document, started_at)
        else:
            logger.warning('Gemini non disponible, fallback vers Hugging Face')
            return self._fallback_to_huggingface(query, k, scope, document, started_at)

    def _generate_with_gemini_primary(self, query: str, k: int, scope: str,
                                     document: Optional[Document], started_at) -> Dict[str, Any]:
        """
        Génération primaire avec Gemini en utilisant la base de données document_content
        """
        logger.info('Génération primaire avec Gemini pour la requête: %s', query)

        if not GEMINI_AVAILABLE:
            logger.error('Gemini non disponible. Installation requise: pip install google-generativeai')
            return {
                'query': query,
                'results': [],
                'count': 0,
                'took_ms': int((timezone.now() - started_at).total_seconds() * 1000),
                'mode': 'gemini_primary',
                'error': 'Gemini non disponible - dépendances manquantes'
            }

        # Configuration de Gemini
        api_key = getattr(settings, 'GOOGLE_AI_API_KEY', os.getenv('GOOGLE_AI_API_KEY'))
        if not api_key:
            logger.error('Clé API Google manquante pour Gemini')
            return {
                'query': query,
                'results': [],
                'count': 0,
                'took_ms': int((timezone.now() - started_at).total_seconds() * 1000),
                'mode': 'gemini_primary',
                'error': 'Clé API Google manquante'
            }

        try:
            genai.configure(api_key=api_key)

            # Étape 1: Récupérer les documents pertinents depuis document_content
            context_documents = self._search_document_content(query, k, scope, document)

            # Étape 2: Construire le contexte pour Gemini
            context_text = self._build_context_for_gemini(context_documents)

            # Étape 3: Générer la réponse avec Gemini
            response_text = self._generate_with_gemini(query, context_text)

            took_ms = int((timezone.now() - started_at).total_seconds() * 1000)

            return {
                'query': query,
                'scope': scope,
                'k': k,
                'count': len(context_documents),
                'took_ms': took_ms,
                'mode': 'gemini_primary',
                'results': context_documents,
                'generated_response': response_text,
                'generation_metadata': {
                    'generation_took_ms': took_ms,
                    'generation_method': 'gemini',
                    'error': None
                }
            }

        except Exception as e:
            logger.exception('Erreur lors de la génération Gemini: %s', e)
            return {
                'query': query,
                'results': [],
                'count': 0,
                'took_ms': int((timezone.now() - started_at).total_seconds() * 1000),
                'mode': 'gemini_primary',
                'error': f'Erreur Gemini: {str(e)}'
            }

    def _fallback_to_huggingface(self, query: str, k: int, scope: str,
                                document: Optional[Document], started_at) -> Dict[str, Any]:
        """
        Fallback vers Hugging Face quand Gemini n'est pas disponible
        """
        logger.info('Activation du fallback Hugging Face pour la requête: %s', query)

        try:
            from .huggingface_rag import huggingface_rag_service

            # Obtenir la réponse générée avec contexte
            response_data = huggingface_rag_service.generate_response(
                query,
                use_custom_context=True
            )

            # Adapter le format de retour pour la compatibilité
            took_ms = int((timezone.now() - started_at).total_seconds() * 1000)

            return {
                'query': query,
                'scope': scope,
                'k': k,
                'count': len(response_data.get('context_documents', [])),
                'took_ms': took_ms,
                'mode': 'huggingface_fallback',
                'results': response_data.get('context_documents', []),
                'generated_response': response_data.get('response', ''),
                'generation_metadata': {
                    'generation_took_ms': response_data.get('took_ms', 0),
                    'generation_method': response_data.get('generation_method', 'huggingface_fallback'),
                    'error': response_data.get('error')
                }
            }

        except ImportError:
            logger.error('Service Hugging Face RAG non disponible')
            return {
                'query': query,
                'results': [],
                'count': 0,
                'took_ms': int((timezone.now() - started_at).total_seconds() * 1000),
                'mode': 'huggingface_fallback',
                'error': 'Service Hugging Face RAG non disponible'
            }
        except Exception as e:
            logger.exception('Erreur RAG Hugging Face: %s', e)
            return {
                'query': query,
                'results': [],
                'count': 0,
                'took_ms': int((timezone.now() - started_at).total_seconds() * 1000),
                'mode': 'huggingface_fallback',
                'error': f'Erreur génération Hugging Face: {str(e)}'
            }

    def _search_document_content(self, query: str, k: int, scope: str,
                                document: Optional[Document]) -> List[Dict[str, Any]]:
        """
        Recherche dans la table document_content en utilisant une recherche textuelle simple
        """
        from django.db.models import Q

        # Construire la requête de base
        if scope == 'document' and document is not None:
            content_qs = DocumentContent.objects.filter(document=document)
        else:
            content_qs = DocumentContent.objects.filter(
                document__is_deleted=False,
                document__status=Document.StatusChoices.PROCESSED,
                document__is_searchable=True
            )

        # Recherche textuelle dans clean_content
        query_terms = query.lower().split()
        q_objects = Q()

        for term in query_terms:
            q_objects |= Q(clean_content__icontains=term)

        content_results = content_qs.filter(q_objects).select_related('document')[:k*2]  # Prendre plus pour filtrer

        # Convertir en format compatible
        results = []
        for content in content_results:
            if len(results) >= k:
                break

            # Extraire un extrait pertinent
            excerpt = self._extract_relevant_excerpt(content.clean_content, query, max_length=500)
            if not excerpt:
                continue

            result_item = {
                'text': excerpt,
                'score': self._calculate_text_relevance(excerpt, query),
                'source': 'document_content',
                'document': {
                    'id': content.document.id,
                    'title': content.document.title,
                    'file_name': content.document.original_filename,
                    'upload_date': content.document.upload_date.isoformat() if content.document.upload_date else None,
                    'file_size': content.document.file_size,
                    'page_count': getattr(content.document, 'page_count', 0)
                },
                'content_metadata': {
                    'word_count': content.word_count,
                    'extraction_method': content.extraction_method,
                    'processing_status': content.processing_status
                }
            }
            results.append(result_item)

        return results

    def _extract_relevant_excerpt(self, text: str, query: str, max_length: int = 500) -> str:
        """
        Extrait un extrait pertinent du texte basé sur la requête
        """
        if not text:
            return ""

        query_terms = [term.lower() for term in query.split()]
        text_lower = text.lower()

        # Trouver la première occurrence d'un terme de recherche
        best_position = 0
        for term in query_terms:
            pos = text_lower.find(term)
            if pos != -1:
                best_position = max(0, pos - 100)  # Commencer un peu avant
                break

        # Extraire l'extrait
        excerpt = text[best_position:best_position + max_length]

        # Nettoyer les début/fin de phrase
        if best_position > 0:
            excerpt = "..." + excerpt
        if len(text) > best_position + max_length:
            excerpt = excerpt + "..."

        return excerpt.strip()

    def _calculate_text_relevance(self, text: str, query: str) -> float:
        """
        Calcule un score de pertinence simple basé sur la fréquence des termes
        """
        query_terms = [term.lower() for term in query.split()]
        text_lower = text.lower()

        score = 0.0
        text_words = text_lower.split()

        for term in query_terms:
            count = text_lower.count(term)
            score += count / len(text_words) if text_words else 0

        return min(1.0, score * 10)  # Normaliser entre 0 et 1

    def _build_context_for_gemini(self, context_documents: List[Dict[str, Any]]) -> str:
        """
        Construit le contexte textuel pour Gemini
        """
        if not context_documents:
            return ""

        context_parts = []
        for i, doc_data in enumerate(context_documents[:5]):  # Limiter à 5 documents
            text = doc_data.get('text', '')
            doc_title = doc_data.get('document', {}).get('title', 'Document inconnu')

            if text:
                context_parts.append(f"## Document {i+1}: {doc_title}\n{text}")

        return "\n\n".join(context_parts)

    def _generate_with_gemini(self, query: str, context: str) -> str:
        """
        Génère une réponse en utilisant Gemini
        """
        model = genai.GenerativeModel('gemini-pro')

        # Construire le prompt
        prompt = f"""En tant qu'assistant AI spécialisé dans l'analyse de documents, réponds à la question suivante en te basant uniquement sur le contexte fourni.

Contexte:
{context}

Question: {query}

Instructions:
- Réponds en français
- Base ta réponse uniquement sur les informations du contexte
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois précis et concis
- Cite les documents pertinents quand c'est approprié

Réponse:"""

        try:
            response = model.generate_content(prompt)
            return response.text if response.text else "Aucune réponse générée"
        except Exception as e:
            logger.error('Erreur génération Gemini: %s', e)
            return f"Erreur lors de la génération de la réponse: {str(e)}"

    def _retrieve_hybrid(self, query: str, k: int, scope: str,
                         document: Optional[Document], started_at) -> Dict[str, Any]:
        """Mode hybride : similarité + génération"""
        # D'abord, récupérer avec similarité
        similarity_results = self._retrieve_similarity_only(query, k, scope, document, started_at)

        # Ensuite, générer une réponse si on a des résultats
        if similarity_results.get('count', 0) > 0:
            # Essayer d'abord LangChain + Gemini
            try:
                from .langchain_rag import langchain_rag_service

                # Utiliser les résultats de similarité comme contexte pour LangChain
                context_texts = [r.get('text', '') for r in similarity_results.get('results', [])]
                context = "\n\n".join(context_texts[:3])

                # Générer directement avec Gemini via LangChain
                langchain_rag_service.ensure_ready()
                messages = langchain_rag_service.prompt.invoke({
                    "question": query,
                    "context": context
                })
                response = langchain_rag_service.llm.invoke(messages)

                took_ms = int((timezone.now() - started_at).total_seconds() * 1000)

                return {
                    **similarity_results,
                    'mode': RAGMode.HYBRID.value,
                    'took_ms': took_ms,
                    'generated_response': response.content,
                    'generation_metadata': {
                        'generation_method': 'langchain_gemini',
                        'generation_took_ms': took_ms,
                        'hybrid_approach': 'similarity + langchain_generation',
                        'error': None
                    }
                }

            except ImportError:
                logger.warning('LangChain non disponible, tentative avec Gemini direct en mode hybride')
                # Fallback vers Gemini direct
                if GEMINI_AVAILABLE:
                    api_key = getattr(settings, 'GOOGLE_AI_API_KEY', os.getenv('GOOGLE_AI_API_KEY'))
                    if api_key:
                        genai.configure(api_key=api_key)
                        context_texts = [r.get('text', '') for r in similarity_results.get('results', [])]
                        context = "\n\n".join(context_texts[:3])
                        gemini_response = self._generate_with_gemini(query, context)

                        took_ms = int((timezone.now() - started_at).total_seconds() * 1000)

                        return {
                            **similarity_results,
                            'mode': RAGMode.HYBRID.value,
                            'took_ms': took_ms,
                            'generated_response': gemini_response,
                            'generation_metadata': {
                                'generation_method': 'gemini_direct',
                                'generation_took_ms': took_ms,
                                'error': None
                            }
                        }
                    else:
                        logger.warning('Clé API Google manquante, fallback vers Hugging Face en mode hybride')
                        raise Exception("Clé API manquante")
                else:
                    logger.warning('Gemini non disponible, fallback vers Hugging Face en mode hybride')
                    raise Exception("Gemini non disponible")

            except Exception as gemini_error:
                logger.warning(f'Erreur génération Gemini en mode hybride: {str(gemini_error)}, tentative avec Hugging Face')
                try:
                    # Fallback vers Hugging Face pour la génération
                    from .huggingface_rag import huggingface_rag_service

                    # Utiliser les résultats de similarité comme contexte
                    context_texts = [r.get('text', '') for r in similarity_results.get('results', [])]
                    context = "\n\n".join(context_texts[:3])  # Limiter à 3 documents

                    # Générer une réponse avec ce contexte
                    generation_response = huggingface_rag_service.generate_response(
                        f"Context: {context}\n\nQuestion: {query}",
                        use_custom_context=False  # On fournit déjà le contexte
                    )

                    # Combiner les résultats
                    took_ms = int((timezone.now() - started_at).total_seconds() * 1000)

                    return {
                        **similarity_results,
                        'mode': RAGMode.HYBRID.value,
                        'took_ms': took_ms,
                        'generated_response': generation_response.get('response', ''),
                        'generation_metadata': {
                            'generation_took_ms': generation_response.get('took_ms', 0),
                            'generation_method': 'huggingface_fallback',
                            'fallback_reason': 'gemini_error',
                            'error': generation_response.get('error')
                        }
                    }

                except Exception as hf_error:
                    logger.warning(f'Fallback Hugging Face également échoué: {str(hf_error)}')

                    # Retourner au moins les résultats de similarité
                    return {
                        **similarity_results,
                        'mode': RAGMode.HYBRID.value,
                        'generated_response': None,
                        'generation_error': f'Erreur Gemini: {str(gemini_error)}, Erreur HF: {str(hf_error)}'
                    }
        else:
            return {
                **similarity_results,
                'mode': RAGMode.HYBRID.value
            }

    def generate_answer(self, query: str, k: int = 5, scope: str = 'all',
                       document: Optional[Document] = None, **generation_kwargs) -> Dict[str, Any]:
        """
        Génère une réponse complète à une question (raccourci pour mode HUGGINGFACE_RAG)

        Args:
            query: La question
            k: Nombre de documents de contexte
            scope: Portée de recherche
            document: Document spécifique (optionnel)
            **generation_kwargs: Arguments pour la génération

        Returns:
            Réponse générée avec métadonnées
        """
        return self.retrieve(query, k=k, scope=scope, document=document, mode=RAGMode.HUGGINGFACE_RAG)


# Singleton retriever
rag_retriever = RAGRetriever()
