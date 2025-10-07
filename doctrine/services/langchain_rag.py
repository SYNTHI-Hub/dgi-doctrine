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
    from langchain.chat_models import init_chat_model
    from langchain_core.documents import Document as LangChainDocument
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_core.prompts import PromptTemplate
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langgraph.graph import START, StateGraph
    from langchain import hub
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain non disponible: {e}")
    LANGCHAIN_AVAILABLE = False

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
            raise Exception("LangChain n'est pas disponible. Installez: pip install langchain langchain-google-genai langgraph")

        if not self._initialized:
            self._initialize_components()

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

    def _initialize_llm(self):
        """Initialise le modèle Gemini via LangChain"""
        api_key = getattr(settings, 'GOOGLE_AI_API_KEY', os.getenv('GOOGLE_AI_API_KEY'))
        if not api_key:
            raise Exception("GOOGLE_AI_API_KEY manquante")

        # Définir la clé API
        os.environ["GOOGLE_API_KEY"] = api_key

        # Initialiser le modèle Gemini
        model_name = getattr(settings, 'GEMINI_MODEL', 'gemini-2.5-flash')
        self.llm = init_chat_model(model_name, model_provider="google_genai")

        logger.info(f"Modèle Gemini initialisé: {model_name}")

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
        try:
            # Essayer de charger depuis LangChain Hub
            self.prompt = hub.pull("rlm/rag-prompt")
            logger.info("Prompt chargé depuis LangChain Hub")
        except Exception as e:
            logger.warning(f"Impossible de charger depuis Hub: {e}, utilisation d'un prompt local")

            # Prompt local en fallback
            template = """Tu es un assistant pour répondre aux questions. Utilise les informations du contexte fourni pour répondre à la question.
Si tu ne connais pas la réponse, dis simplement que tu ne sais pas. Utilise maximum trois phrases et garde la réponse concise.

Question: {question}
Contexte: {context}

Réponse:"""

            self.prompt = PromptTemplate.from_template(template)

    def _build_rag_graph(self):
        """Construit le graphe RAG sophistiqué avec LangGraph"""

        def retrieve(state: RAGState):
            """Étape de récupération avec logique conditionnelle"""
            metadata = state.get("metadata", {})
            question = state["question"]
            k = metadata.get("k", 5)
            scope = metadata.get("scope", "all")
            document_id = metadata.get("document_id")

            logger.info(f"Récupération: question='{question}', scope='{scope}', k={k}")

            # Logique de récupération selon le scope
            if scope == 'document' and document_id:
                # Recherche dans un document spécifique
                retrieved_docs = self._search_specific_document(question, document_id, k)
                logger.info(f"Recherche document spécifique {document_id}: {len(retrieved_docs)} docs trouvés")
            else:
                # Recherche globale dans le vector store
                retrieved_docs = self.vector_store.similarity_search(question, k=k)
                logger.info(f"Recherche globale: {len(retrieved_docs)} docs trouvés")

            # Enrichir les métadonnées avec des infos de récupération
            retrieval_metadata = {
                "retrieval_method": "specific_document" if scope == 'document' else "global_similarity",
                "docs_found": len(retrieved_docs),
                "search_scope": scope,
                "target_document": document_id
            }

            return {
                "context": retrieved_docs,
                "metadata": {**metadata, "retrieval": retrieval_metadata}
            }

        def validate_context(state: RAGState):
            """Étape de validation du contexte récupéré"""
            context = state.get("context", [])
            question = state["question"]

            if not context:
                logger.warning(f"Aucun contexte trouvé pour: {question}")
                return {
                    "context": [],
                    "metadata": {
                        **state.get("metadata", {}),
                        "validation": {
                            "status": "no_context",
                            "action": "generate_empty_response"
                        }
                    }
                }

            # Filtrer les documents très courts ou non pertinents
            filtered_context = []
            for doc in context:
                if len(doc.page_content.strip()) > 50:  # Minimum 50 caractères
                    filtered_context.append(doc)

            logger.info(f"Contexte validé: {len(filtered_context)}/{len(context)} documents conservés")

            return {
                "context": filtered_context,
                "metadata": {
                    **state.get("metadata", {}),
                    "validation": {
                        "status": "valid" if filtered_context else "no_valid_context",
                        "original_count": len(context),
                        "filtered_count": len(filtered_context)
                    }
                }
            }

        def generate(state: RAGState):
            """Étape de génération avec gestion des cas d'échec"""
            context = state.get("context", [])
            question = state["question"]
            metadata = state.get("metadata", {})

            if not context:
                logger.warning("Génération sans contexte")
                return {
                    "answer": "Je n'ai pas trouvé d'informations pertinentes dans la base de connaissances pour répondre à votre question.",
                    "metadata": {
                        **metadata,
                        "generation": {
                            "status": "no_context",
                            "context_used": False
                        }
                    }
                }

            # Construire le contexte textuel
            docs_content = "\n\n".join([
                f"Document {i+1}: {doc.page_content}"
                for i, doc in enumerate(context[:5])  # Limiter à 5 documents
            ])

            logger.info(f"Génération avec {len(context)} documents de contexte")

            try:
                # Générer la réponse
                messages = self.prompt.invoke({
                    "question": question,
                    "context": docs_content
                })
                response = self.llm.invoke(messages)

                return {
                    "answer": response.content,
                    "metadata": {
                        **metadata,
                        "generation": {
                            "status": "success",
                            "context_docs_used": len(context),
                            "context_length": len(docs_content)
                        }
                    }
                }

            except Exception as e:
                logger.error(f"Erreur lors de la génération: {e}")
                return {
                    "answer": f"Désolé, une erreur s'est produite lors de la génération de la réponse: {str(e)}",
                    "metadata": {
                        **metadata,
                        "generation": {
                            "status": "error",
                            "error": str(e)
                        }
                    }
                }

        # Construire le graphe avec étapes séquentielles
        graph_builder = StateGraph(RAGState)

        # Ajouter les nœuds
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("validate_context", validate_context)
        graph_builder.add_node("generate", generate)

        # Définir les connexions
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "validate_context")
        graph_builder.add_edge("validate_context", "generate")

        # Compiler le graphe
        self.rag_graph = graph_builder.compile()

        logger.info("Graphe RAG sophistiqué construit avec succès (retrieve -> validate -> generate)")

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

            # Préparer l'état initial pour LangGraph
            initial_state = {
                "question": question,
                "context": [],
                "answer": "",
                "metadata": {"k": k, "document_id": document_id, "scope": scope, **kwargs}
            }

            # Exécuter le workflow LangGraph
            logger.info(f"Exécution du workflow LangGraph pour: {question}")
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
            logger.exception(f"Erreur lors de la requête RAG avec LangGraph: {e}")

            end_time = timezone.now()
            took_ms = int((end_time - start_time).total_seconds() * 1000)

            return {
                "query": question,
                "answer": f"Désolé, une erreur s'est produite lors de la recherche dans la base de connaissances: {str(e)}",
                "context_documents": [],
                "count": 0,
                "took_ms": took_ms,
                "mode": "langchain_langgraph",
                "generation_metadata": {
                    "generation_method": "langchain_langgraph",
                    "error": str(e)
                }
            }

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


# Instance singleton
langchain_rag_service = LangChainRAGService()