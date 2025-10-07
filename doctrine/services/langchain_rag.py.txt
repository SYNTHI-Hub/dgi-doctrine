"""
Service RAG utilisant LangChain + Gemini
Basé sur la documentation LangChain RAG Tutorial
"""

import logging
import os
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

from django.utils import timezone
from django.conf import settings
from django.db.models import Q

logger = logging.getLogger(__name__)

# Imports LangChain
try:
    import google.generativeai as genai
    from langchain_core.documents import Document as LangChainDocument
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_core.prompts import PromptTemplate
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langgraph.graph import START, StateGraph
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain imports reussis")
except ImportError as e:
    logger.error(f"LangChain non disponible: {e}")
    LANGCHAIN_AVAILABLE = False
    # Definir des classes factices pour eviter les erreurs
    class LangChainDocument:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class InMemoryVectorStore:
        def __init__(self, embeddings):
            self.embeddings = embeddings
            self._docs = []

        def add_documents(self, documents):
            self._docs.extend(documents)

        def similarity_search(self, query, k=5):
            return self._docs[:k]

    class PromptTemplate:
        @staticmethod
        def from_template(template):
            return template

        def invoke(self, data):
            return data

    class RecursiveCharacterTextSplitter:
        def __init__(self, **kwargs):
            pass

        def split_documents(self, docs):
            return docs

    class StateGraph:
        def __init__(self, state_type):
            pass

        def add_sequence(self, nodes):
            return self

        def add_edge(self, start, end):
            return self

        def compile(self):
            return self

    START = None
    genai = None

# Imports Django models
from doctrine.models import Document, DocumentContent


class RAGState(TypedDict):
    """État du pipeline RAG LangChain"""
    question: str
    context: List[LangChainDocument]
    answer: str
    metadata: Dict[str, Any]


class LangChainRAGService:
    """Service RAG utilisant LangChain + Gemini"""

    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.rag_graph = None
        self.prompt = None
        self._initialized = False

    def ensure_ready(self):
        """Initialise le service si nécessaire"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain non disponible, utilisation du mode fallback")
            self._initialize_fallback_mode()
            return

        if not self._initialized:
            try:
                self._initialize_components()
            except Exception as e:
                logger.warning(f"Echec initialisation LangChain, utilisation du mode fallback: {e}")
                self._initialize_fallback_mode()

    def _initialize_components(self):
        """Initialise les composants LangChain"""
        try:
            logger.info("Initialisation du service LangChain RAG...")

            # 1. Initialiser le modèle Gemini
            self._initialize_llm()

            # 2. Initialiser les embeddings (utilise sentence-transformers local)
            self._initialize_embeddings()

            # 3. Initialiser le vector store
            self._initialize_vector_store()

            # 4. Charger les documents dans le vector store
            self._load_documents()

            # 5. Initialiser le prompt
            self._initialize_prompt()

            # 6. Construire le graphe RAG
            self._build_rag_graph()

            self._initialized = True
            logger.info("Service LangChain RAG initialisé avec succès")

        except Exception as e:
            logger.exception(f"Erreur lors de l'initialisation LangChain RAG: {e}")
            raise

    def _initialize_fallback_mode(self):
        """Initialise le service en mode fallback sans LangChain"""
        try:
            logger.info("Initialisation du mode fallback sans LangChain...")

            # Utiliser le service RAG existant comme backend
            from .rag import rag_retriever

            # Configuration minimale
            self.llm = None
            self.embeddings = None
            self.vector_store = None
            self.rag_graph = None
            self.prompt = None
            self.rag_retriever = rag_retriever

            self._initialized = True
            logger.info("Mode fallback initialise avec succes")

        except Exception as e:
            logger.exception(f"Erreur lors de l'initialisation du mode fallback: {e}")
            self._initialized = False

    def _initialize_llm(self):
        """Initialise le modèle Gemini directement"""
        api_key = getattr(settings, 'GOOGLE_AI_API_KEY', os.getenv('GOOGLE_AI_API_KEY'))
        if not api_key:
            raise Exception("GOOGLE_AI_API_KEY manquante")

        if not genai:
            raise Exception("google.generativeai non disponible")

        # Configurer Gemini directement
        genai.configure(api_key=api_key)
        model_name = getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash')

        # Creer un wrapper simple pour Gemini
        class GeminiWrapper:
            def __init__(self, model_name):
                self.model = genai.GenerativeModel(model_name)

            def invoke(self, messages):
                if isinstance(messages, dict):
                    prompt = messages.get('context', '') + "\n\nQuestion: " + messages.get('question', '')
                else:
                    prompt = str(messages)

                try:
                    response = self.model.generate_content(prompt)
                    return type('Response', (), {'content': response.text if response.text else "Aucune reponse generee"})()
                except Exception as e:
                    logger.error(f"Erreur generation Gemini: {e}")
                    return type('Response', (), {'content': f"Erreur: {str(e)}"})()

        self.llm = GeminiWrapper(model_name)
        logger.info(f"Modele Gemini initialise: {model_name}")

    def _initialize_embeddings(self):
        """Initialise les embeddings (utilise sentence-transformers local)"""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            # Utiliser sentence-transformers pour les embeddings locaux
            class SentenceTransformerEmbeddings:
                def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
                    self.model = SentenceTransformer(model_name)

                def embed_documents(self, texts):
                    return self.model.encode(texts, normalize_embeddings=True).tolist()

                def embed_query(self, text):
                    return self.model.encode([text], normalize_embeddings=True)[0].tolist()

            self.embeddings = SentenceTransformerEmbeddings()
            logger.info("Embeddings sentence-transformers initialisés")

        except ImportError:
            logger.warning("sentence-transformers non disponible, utilisation d'embeddings basiques")
            # Fallback vers embeddings basiques
            self.embeddings = self._create_basic_embeddings()

    def _create_basic_embeddings(self):
        """Crée des embeddings basiques en fallback"""
        class BasicEmbeddings:
            def embed_documents(self, texts):
                # Embeddings très basiques pour le fallback
                return [[hash(text) % 1000 / 1000.0 for _ in range(384)] for text in texts]

            def embed_query(self, text):
                return [hash(text) % 1000 / 1000.0 for _ in range(384)]

        return BasicEmbeddings()

    def _initialize_vector_store(self):
        """Initialise le vector store"""
        self.vector_store = InMemoryVectorStore(self.embeddings)
        logger.info("Vector store initialisé")

    def _load_documents(self, force_reload=False):
        """Charge les documents depuis la base de données Django dans le vector store"""
        logger.info("Chargement des documents depuis la base de données Django...")

        # Vérifier si déjà chargé (sauf si force_reload)
        if hasattr(self.vector_store, '_docs') and len(self.vector_store._docs) > 0 and not force_reload:
            logger.info(f"Documents déjà chargés ({len(self.vector_store._docs)} chunks)")
            return

        # Récupérer les contenus de documents depuis la base Django
        content_qs = DocumentContent.objects.filter(
            document__is_deleted=False,
            document__status=Document.StatusChoices.PROCESSED,
            document__is_searchable=True,
            clean_content__isnull=False
        ).exclude(clean_content='').select_related('document')

        documents = []
        total_docs = content_qs.count()

        logger.info(f"Traitement de {total_docs} documents de la base de données...")

        for i, content in enumerate(content_qs, 1):
            try:
                if not content.clean_content.strip():
                    continue

                # Log de progression
                if i % 10 == 0:
                    logger.info(f"Traitement document {i}/{total_docs}")

                # Créer un document LangChain avec métadonnées enrichies
                doc = LangChainDocument(
                    page_content=content.clean_content,
                    metadata={
                        'document_id': str(content.document.id),
                        'document_title': content.document.title,
                        'document_filename': content.document.original_filename,
                        'upload_date': content.document.upload_date.isoformat() if content.document.upload_date else None,
                        'content_id': str(content.id),
                        'word_count': content.word_count,
                        'extraction_method': content.extraction_method,
                        'processing_status': content.processing_status,
                        'document_category': content.document.category.name if content.document.category else None,
                        'document_theme': content.document.theme.name if content.document.theme else None,
                        'file_size': content.document.file_size,
                        'language': content.document.language,
                        'visibility': content.document.visibility,
                        'source': 'django_db'
                    }
                )

                # Diviser le document en chunks si nécessaire
                if len(content.clean_content) > 1000:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        add_start_index=True
                    )
                    chunks = text_splitter.split_documents([doc])

                    # Enrichir les métadonnées des chunks
                    for j, chunk in enumerate(chunks):
                        chunk.metadata.update({
                            'chunk_index': j,
                            'total_chunks': len(chunks),
                            'is_chunk': True
                        })

                    documents.extend(chunks)
                else:
                    # Document assez petit, pas de chunking nécessaire
                    doc.metadata['is_chunk'] = False
                    documents.append(doc)

            except Exception as e:
                logger.warning(f"Erreur lors du traitement du document {content.document.id}: {e}")
                continue

        # Ajouter tous les documents au vector store
        if documents:
            logger.info(f"Ajout de {len(documents)} chunks au vector store...")
            try:
                self.vector_store.add_documents(documents)
                logger.info(f"✅ {len(documents)} chunks ajoutés avec succès au vector store")

                # Statistiques
                unique_docs = len(set(doc.metadata['document_id'] for doc in documents))
                logger.info(f"📊 Statistiques: {unique_docs} documents originaux -> {len(documents)} chunks")

            except Exception as e:
                logger.error(f"Erreur lors de l'ajout au vector store: {e}")
                raise
        else:
            logger.warning("❌ Aucun document valide trouvé dans la base de données")

    def reload_knowledge_base(self):
        """Recharge la base de connaissance depuis la base de données"""
        logger.info("🔄 Rechargement de la base de connaissance...")

        # Réinitialiser le vector store
        self.vector_store = InMemoryVectorStore(self.embeddings)

        # Recharger les documents
        self._load_documents(force_reload=True)

        logger.info("✅ Base de connaissance rechargée")

    def get_knowledge_stats(self):
        """Retourne les statistiques de la base de connaissance"""
        try:
            # Stats du vector store
            total_chunks = len(self.vector_store._docs) if hasattr(self.vector_store, '_docs') else 0

            # Stats de la base de données
            db_docs = DocumentContent.objects.filter(
                document__is_deleted=False,
                document__status=Document.StatusChoices.PROCESSED,
                document__is_searchable=True,
                clean_content__isnull=False
            ).exclude(clean_content='').count()

            return {
                'total_chunks_in_vectorstore': total_chunks,
                'total_docs_in_database': db_docs,
                'vector_store_initialized': self.vector_store is not None,
                'last_loaded': getattr(self, '_last_loaded', None)
            }

        except Exception as e:
            logger.error(f"Erreur lors du calcul des stats: {e}")
            return {
                'error': str(e),
                'total_chunks_in_vectorstore': 0,
                'total_docs_in_database': 0
            }

    def _initialize_prompt(self):
        """Initialise le prompt RAG"""
        # Utiliser un prompt local simple
        template = """Tu es un assistant pour repondre aux questions. Utilise les informations du contexte fourni pour repondre a la question.
Si tu ne connais pas la reponse, dis simplement que tu ne sais pas. Utilise maximum trois phrases et garde la reponse concise.

Question: {question}
Contexte: {context}

Reponse:"""

        if LANGCHAIN_AVAILABLE:
            self.prompt = PromptTemplate.from_template(template)
        else:
            # Fallback simple pour le prompt
            class SimplePrompt:
                def __init__(self, template):
                    self.template = template

                def invoke(self, data):
                    return self.template.format(**data)

            self.prompt = SimplePrompt(template)

        logger.info("Prompt RAG initialise")

    def _build_rag_graph(self):
        """Construit le workflow RAG simplifie"""
        if LANGCHAIN_AVAILABLE:
            try:
                # Workflow LangGraph complet si disponible
                self._build_langgraph_workflow()
            except Exception as e:
                logger.warning(f"Echec workflow LangGraph: {e}, utilisation workflow simple")
                self._build_simple_workflow()
        else:
            # Workflow simplifie en fallback
            self._build_simple_workflow()

    def _build_langgraph_workflow(self):
        """Construit le workflow avec LangGraph"""
        def retrieve(state: RAGState):
            metadata = state.get("metadata", {})
            question = state["question"]
            k = metadata.get("k", 5)
            scope = metadata.get("scope", "all")
            document_id = metadata.get("document_id")

            if scope == 'document' and document_id:
                retrieved_docs = self._search_specific_document(question, document_id, k)
            else:
                retrieved_docs = self.vector_store.similarity_search(question, k=k)

            return {"context": retrieved_docs}

        def generate(state: RAGState):
            context = state.get("context", [])
            question = state["question"]

            if not context:
                return {"answer": "Je n'ai pas trouve d'informations pertinentes dans la base de connaissances."}

            docs_content = "\n\n".join([doc.page_content for doc in context[:5]])
            messages = self.prompt.invoke({"question": question, "context": docs_content})
            response = self.llm.invoke(messages)
            return {"answer": response.content}

        graph_builder = StateGraph(RAGState)
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("generate", generate)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        self.rag_graph = graph_builder.compile()
        logger.info("Workflow LangGraph construit avec succes")

    def _build_simple_workflow(self):
        """Construit un workflow simplifie sans LangGraph"""
        class SimpleWorkflow:
            def __init__(self, service):
                self.service = service

            def invoke(self, state):
                question = state["question"]
                metadata = state.get("metadata", {})
                k = metadata.get("k", 5)
                scope = metadata.get("scope", "all")
                document_id = metadata.get("document_id")

                # Etape 1: Recuperation
                if scope == 'document' and document_id:
                    context = self.service._search_specific_document(question, document_id, k)
                else:
                    context = self.service.vector_store.similarity_search(question, k=k)

                # Etape 2: Generation
                if not context:
                    answer = "Je n'ai pas trouve d'informations pertinentes dans la base de connaissances."
                else:
                    docs_content = "\n\n".join([doc.page_content for doc in context[:5]])
                    prompt_text = self.service.prompt.invoke({"question": question, "context": docs_content})
                    response = self.service.llm.invoke(prompt_text)
                    answer = response.content

                return {
                    "question": question,
                    "context": context,
                    "answer": answer,
                    "metadata": metadata
                }

        self.rag_graph = SimpleWorkflow(self)
        logger.info("Workflow simplifie construit avec succes")

    def query(self, question: str, k: int = 5, document_id: str = None, scope: str = 'all', **kwargs) -> Dict[str, Any]:
        """
        Interroge le système RAG avec LangGraph pour un workflow complet

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

        try:
            self.ensure_ready()

            # Mode fallback : utiliser le service RAG existant
            if not self._initialized or hasattr(self, 'rag_retriever'):
                return self._query_fallback_mode(question, k, document_id, scope, start_time, **kwargs)

            # Mode LangChain complet
            initial_state = {
                "question": question,
                "context": [],
                "answer": "",
                "metadata": {"k": k, "document_id": document_id, "scope": scope, **kwargs}
            }

            logger.info(f"Execution du workflow LangGraph pour: {question}")
            final_state = self.rag_graph.invoke(initial_state)

            # Calculer le temps de traitement
            end_time = timezone.now()
            took_ms = int((end_time - start_time).total_seconds() * 1000)

            # Formater la réponse
            return {
                "query": question,
                "answer": final_state.get("answer", "Aucune réponse générée"),
                "context_documents": [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata,
                        "score": 1.0  # Score placeholder
                    }
                    for doc in final_state.get("context", [])
                ],
                "count": len(final_state.get("context", [])),
                "took_ms": took_ms,
                "mode": "langchain_langgraph",
                "scope": scope,
                "generation_metadata": {
                    "generation_method": "langchain_langgraph",
                    "model": getattr(settings, 'GEMINI_MODEL', 'gemini-2.5-flash'),
                    "knowledge_source": "django_database",
                    "filtered_by_document": document_id is not None,
                    "workflow_steps": ["retrieve", "generate"],
                    "error": None
                }
            }

        except Exception as e:
            logger.exception(f"Erreur lors de la requete RAG: {e}")
            return self._query_fallback_mode(question, k, document_id, scope, start_time, error=str(e), **kwargs)

    def _search_specific_document(self, query: str, document_id: str, k: int) -> List[LangChainDocument]:
        """
        Recherche dans un document spécifique de la base de données

        Args:
            query: Requête de recherche
            document_id: ID du document à rechercher
            k: Nombre de résultats

        Returns:
            Liste des documents LangChain trouvés
        """
        try:
            # Recherche directe dans la base de données pour ce document
            content = DocumentContent.objects.filter(
                document__id=document_id,
                document__is_deleted=False,
                document__status=Document.StatusChoices.PROCESSED,
                document__is_searchable=True,
                clean_content__isnull=False
            ).exclude(clean_content='').select_related('document').first()

            if not content:
                logger.warning(f"Document {document_id} non trouvé dans la base de données")
                return []

            # Créer un document LangChain
            doc = LangChainDocument(
                page_content=content.clean_content,
                metadata={
                    'document_id': str(content.document.id),
                    'document_title': content.document.title,
                    'document_filename': content.document.original_filename,
                    'upload_date': content.document.upload_date.isoformat() if content.document.upload_date else None,
                    'content_id': str(content.id),
                    'word_count': content.word_count,
                    'extraction_method': content.extraction_method,
                    'source': 'django_db_direct',
                    'search_type': 'specific_document'
                }
            )

            # Diviser en chunks pour analyse
            if len(content.clean_content) > 1000:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents([doc])

                # Sélectionner les chunks les plus pertinents (simulation de similarité)
                # En production, vous pourriez faire une recherche sémantique sur les chunks
                return chunks[:k]
            else:
                return [doc]

        except Exception as e:
            logger.error(f"Erreur lors de la recherche dans le document {document_id}: {e}")
            return []

    def search_in_database(self, query: str, k: int = 5, category: str = None, theme: str = None) -> List[Dict[str, Any]]:
        """
        Recherche directe dans la base de données avec filtres

        Args:
            query: Requête de recherche
            k: Nombre de résultats
            category: Filtrer par catégorie (optionnel)
            theme: Filtrer par thème (optionnel)

        Returns:
            Liste des résultats de la base de données
        """
        try:
            # Construire la requête de base
            content_qs = DocumentContent.objects.filter(
                document__is_deleted=False,
                document__status=Document.StatusChoices.PROCESSED,
                document__is_searchable=True,
                clean_content__isnull=False
            ).exclude(clean_content='').select_related('document')

            # Appliquer les filtres
            if category:
                content_qs = content_qs.filter(document__category__name__icontains=category)

            if theme:
                content_qs = content_qs.filter(document__theme__name__icontains=theme)

            # Recherche textuelle simple (en attendant une recherche full-text)
            query_terms = query.lower().split()
            q_objects = Q()
            for term in query_terms:
                q_objects |= Q(clean_content__icontains=term)

            if q_objects:
                content_qs = content_qs.filter(q_objects)

            # Limiter les résultats
            results = []
            for content in content_qs[:k*2]:  # Récupérer plus pour filtrer
                if len(results) >= k:
                    break

                # Extraire un extrait pertinent
                excerpt = self._extract_relevant_excerpt(content.clean_content, query)
                if excerpt:
                    results.append({
                        'content': excerpt,
                        'document_id': str(content.document.id),
                        'document_title': content.document.title,
                        'document_filename': content.document.original_filename,
                        'category': content.document.category.name if content.document.category else None,
                        'theme': content.document.theme.name if content.document.theme else None,
                        'word_count': content.word_count,
                        'source': 'django_db_search'
                    })

            return results

        except Exception as e:
            logger.error(f"Erreur lors de la recherche dans la base de données: {e}")
            return []

    def _extract_relevant_excerpt(self, text: str, query: str, max_length: int = 500) -> str:
        """Extrait un extrait pertinent basé sur la requête"""
        if not text:
            return ""

        query_terms = [term.lower() for term in query.split()]
        text_lower = text.lower()

        # Trouver la première occurrence d'un terme
        best_position = 0
        for term in query_terms:
            pos = text_lower.find(term)
            if pos != -1:
                best_position = max(0, pos - 100)
                break

        # Extraire l'extrait
        excerpt = text[best_position:best_position + max_length]

        if best_position > 0:
            excerpt = "..." + excerpt
        if len(text) > best_position + max_length:
            excerpt = excerpt + "..."

        return excerpt.strip()

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Recherche par similarité uniquement

        Args:
            query: Requête de recherche
            k: Nombre de résultats

        Returns:
            Liste des documents similaires
        """
        try:
            self.ensure_ready()

            docs = self.vector_store.similarity_search(query, k=k)

            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 1.0  # Score placeholder
                }
                for doc in docs
            ]

        except Exception as e:
            logger.exception(f"Erreur lors de la recherche par similarité: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du service"""
        return {
            "initialized": self._initialized,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "vector_store_initialized": self.vector_store is not None,
            "llm_initialized": self.llm is not None,
            "documents_in_store": len(self.vector_store._docs) if self.vector_store else 0
        }

    def _query_fallback_mode(self, question: str, k: int, document_id: str, scope: str, start_time, error: str = None, **kwargs) -> Dict[str, Any]:
        """Mode fallback utilisant le service RAG existant"""
        try:
            logger.info(f"Utilisation du mode fallback pour: {question}")

            # Utiliser le service RAG existant
            if hasattr(self, 'rag_retriever'):
                document = None
                if scope == 'document' and document_id:
                    from doctrine.models import Document
                    try:
                        document = Document.objects.get(id=document_id)
                    except Document.DoesNotExist:
                        pass

                # Utiliser le mode Gemini du service RAG existant
                from doctrine.services.rag import RAGMode
                result = self.rag_retriever.retrieve(
                    question,
                    k=k,
                    scope=scope,
                    document=document,
                    mode=RAGMode.HUGGINGFACE_RAG  # Mode avec generation Gemini
                )

                end_time = timezone.now()
                took_ms = int((end_time - start_time).total_seconds() * 1000)

                # Extraire la reponse generee si disponible
                generated_response = result.get('generated_response', '')
                if not generated_response and result.get('results'):
                    # Fallback sur le premier resultat si pas de generation
                    generated_response = f"Basé sur la recherche: {result.get('results', [{}])[0].get('text', 'Aucun resultat trouve')}"
                elif not generated_response:
                    generated_response = "Aucune information trouvee dans la base de connaissances pour cette question."

                return {
                    "query": question,
                    "answer": generated_response,
                    "context_documents": result.get('results', []),
                    "count": result.get('count', 0),
                    "took_ms": took_ms,
                    "mode": "langchain_fallback_with_generation",
                    "scope": scope,
                    "generation_metadata": {
                        "generation_method": "langchain_fallback_gemini",
                        "fallback_reason": error or "LangChain non disponible",
                        "original_mode": result.get('mode', 'unknown'),
                        "backend_took_ms": result.get('took_ms', 0),
                        "error": error
                    }
                }

        except Exception as fallback_error:
            logger.error(f"Echec du mode fallback: {fallback_error}")

        # Dernier recours
        end_time = timezone.now()
        took_ms = int((end_time - start_time).total_seconds() * 1000)

        return {
            "query": question,
            "answer": "Service temporairement indisponible. Veuillez reessayer plus tard.",
            "context_documents": [],
            "count": 0,
            "took_ms": took_ms,
            "mode": "langchain_error",
            "generation_metadata": {
                "generation_method": "error",
                "error": error or "Service indisponible"
            }
        }


# Instance singleton
langchain_rag_service = LangChainRAGService()