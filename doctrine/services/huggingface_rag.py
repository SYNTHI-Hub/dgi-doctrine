import logging
import torch
from typing import Dict, Any, Optional, List
from django.utils import timezone

try:
    from django.conf import settings
    DJANGO_AVAILABLE = True
except Exception:
    settings = None
    DJANGO_AVAILABLE = False

logger = logging.getLogger(__name__)

try:
    from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HF_AVAILABLE = True
except ImportError:
    RagTokenizer = None
    RagRetriever = None
    RagSequenceForGeneration = None
    SentenceTransformer = None
    np = None
    HF_AVAILABLE = False


class HuggingFaceRAGDependenciesMissing(Exception):
    """Exception levée quand les dépendances Hugging Face ne sont pas disponibles"""
    pass


class HuggingFaceRAGService:
    """
    Service RAG utilisant Hugging Face avec singleton pattern
    Intègre avec l'architecture existante tout en respectant DRY
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HuggingFaceRAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.tokenizer = None
        self.retriever = None
        self.model = None
        self.device = None
        self.embedder = None
        self._custom_index = None

        if DJANGO_AVAILABLE and settings is not None:
            self.model_name = getattr(settings, 'RAG_MODEL_NAME', 'facebook/rag-token-nq')
            self.retriever_name = getattr(settings, 'RAG_RETRIEVER_NAME', 'facebook/rag-token-nq')
            self.max_new_tokens = getattr(settings, 'RAG_MAX_NEW_TOKENS', 100)
            self.use_gpu = getattr(settings, 'RAG_USE_GPU', torch.cuda.is_available())
            self.use_dummy_dataset = getattr(settings, 'RAG_USE_DUMMY_DATASET', True)
            self.use_custom_retriever = getattr(settings, 'RAG_USE_CUSTOM_RETRIEVER', True)
        else:

            self.model_name = 'facebook/rag-token-nq'
            self.retriever_name = 'facebook/rag-token-nq'
            self.max_new_tokens = 100
            self.use_gpu = torch.cuda.is_available()
            self.use_dummy_dataset = True
            self.use_custom_retriever = True

        self._initialized = True

    def ensure_ready(self):
        """Initialise le modèle RAG si nécessaire"""
        if not HF_AVAILABLE:
            raise HuggingFaceRAGDependenciesMissing(
                "Dépendances Hugging Face manquantes. Installez: pip install transformers torch sentence-transformers accelerate"
            )

        if self.model is None:
            self._initialize_model()

    def _initialize_model(self):
        """Initialise le modèle RAG avec optimisations"""
        try:
            logger.info(f"Initialisation du modèle RAG: {self.model_name}")
            start_time = timezone.now()

            if self.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Utilisation du GPU pour RAG")
            else:
                self.device = torch.device("cpu")
                logger.info("Utilisation du CPU pour RAG")

            self.tokenizer = RagTokenizer.from_pretrained(self.model_name)

            if self.use_custom_retriever:
                logger.info("Utilisation du retriever personnalisé basé sur la base de données")
                self._initialize_custom_retriever()
            else:

                logger.info(f"Chargement du retriever Hugging Face: {self.retriever_name}")
                self.retriever = RagRetriever.from_pretrained(
                    self.retriever_name,
                    index_name="compressed",
                    use_dummy_dataset=self.use_dummy_dataset,
                )

            # Chargement du modèle avec optimisations
            model_kwargs = {
                "retriever": self.retriever,
                "low_cpu_mem_usage": True,
            }

            if self.use_gpu and torch.cuda.is_available():
                model_kwargs.update({
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                })

            self.model = RagSequenceForGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            if not self.use_gpu or not torch.cuda.is_available():
                self.model = self.model.to(self.device)

            end_time = timezone.now()
            loading_time = (end_time - start_time).total_seconds()
            logger.info(f"Modèle RAG initialisé en {loading_time:.1f}s")

        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle RAG: {str(e)}")
            raise

    def _initialize_custom_retriever(self):
        """Initialise un retriever personnalisé basé sur notre base de données"""
        from .rag import rag_retriever  # Réutiliser notre retriever existant

        # Initialiser l'embedder pour la cohérence
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Pour l'instant, on crée un retriever factice qui utilisera notre logique
        # En production, vous pourriez créer un retriever Hugging Face personnalisé
        class CustomRetriever:
            def __init__(self, rag_retriever):
                self.rag_retriever = rag_retriever

            def __call__(self, question_input_ids, question_hidden_states, docs_dict):
                # Cette méthode sera appelée par le modèle RAG
                # Pour l'instant, on retourne des documents factices
                # En production, intégrez avec rag_retriever.retrieve()
                return docs_dict

        # Créer un retriever factice pour la compatibilité
        # Note: Pour une intégration complète, vous devriez créer un RagRetriever personnalisé
        try:
            # Utiliser directement un modèle RAG complet au lieu d'un DPR seul
            rag_model_name = "facebook/rag-token-nq"  # Modèle RAG complet avec question_encoder et generator

            self.retriever = RagRetriever.from_pretrained(
                rag_model_name,
                index_name="compressed",
                use_dummy_dataset=True,  # Utiliser dummy pour éviter le téléchargement
            )
            logger.info(f"Retriever RAG initialisé avec {rag_model_name}")

        except Exception as e:
            logger.warning(f"Impossible d'initialiser le retriever RAG: {e}")
            logger.info("Désactivation du retriever Hugging Face - utilisation du mode dégradé")
            self.retriever = None

    def generate_response(self, query: str, max_new_tokens: Optional[int] = None,
                         use_custom_context: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Génère une réponse à partir d'une requête

        Args:
            query: La question à traiter
            max_new_tokens: Nombre maximum de tokens à générer
            use_custom_context: Si True, utilise notre base de données pour le contexte
            **kwargs: Arguments supplémentaires pour la génération

        Returns:
            Dict contenant la réponse et les métadonnées
        """
        try:
            self.ensure_ready()
        except Exception as e:
            logger.error(f"Impossible d'initialiser le service RAG: {e}")
            return {
                'generated_text': 'Service RAG temporairement indisponible. Veuillez réessayer plus tard.',
                'error': str(e),
                'success': False,
                'query': query,
                'generation_time': 0,
                'model_info': {
                    'model_name': 'N/A',
                    'error': 'Service indisponible'
                }
            }

        start_time = timezone.now()

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        try:
            response_data = {
                'query': query,
                'response': '',
                'context_documents': [],
                'generation_method': 'huggingface_rag',
                'took_ms': 0,
                'error': None
            }

            # Si utilisation du contexte personnalisé, récupérer d'abord les documents pertinents
            if use_custom_context:
                context_data = self._get_custom_context(query)
                response_data['context_documents'] = context_data.get('results', [])

                # Construire le contexte pour le modèle
                context_text = self._build_context_text(context_data.get('results', []))

                # Intégrer le contexte dans la requête
                enriched_query = f"Context: {context_text}\n\nQuestion: {query}"
            else:
                enriched_query = query

            # Préparer l'input pour le modèle
            input_dict = self.tokenizer.prepare_seq2seq_batch(
                enriched_query,
                return_tensors="pt"
            )

            # Déplacer vers le bon device
            if self.device:
                input_dict = {k: v.to(self.device) for k, v in input_dict.items()}

            # Génération
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'num_beams': kwargs.get('num_beams', 4),
                'do_sample': kwargs.get('do_sample', False),
                'early_stopping': kwargs.get('early_stopping', True),
                'temperature': kwargs.get('temperature', 1.0),
                'top_p': kwargs.get('top_p', 1.0),
            }

            with torch.no_grad():
                generated = self.model.generate(**input_dict, **generation_kwargs)

            # Décoder la réponse
            generated_text = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )[0]

            # Nettoyer la réponse (enlever la question si elle est répétée)
            if enriched_query in generated_text:
                generated_text = generated_text.replace(enriched_query, "").strip()

            response_data['response'] = generated_text

        except Exception as e:
            logger.error(f"Erreur lors de la génération RAG: {str(e)}")
            response_data['error'] = f"Erreur de génération: {str(e)}"
            response_data['response'] = "Désolé, je n'ai pas pu générer une réponse pour cette question."

        finally:
            end_time = timezone.now()
            response_data['took_ms'] = int((end_time - start_time).total_seconds() * 1000)

        return response_data

    def _get_custom_context(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Récupère le contexte depuis notre base de données existante"""
        from .rag import rag_retriever

        try:
            return rag_retriever.retrieve(query, k=k, scope='all')
        except Exception as e:
            logger.warning(f"Erreur lors de la récupération du contexte: {str(e)}")
            return {'results': [], 'error': str(e)}

    def _build_context_text(self, context_documents: List[Dict]) -> str:
        """Construit le texte de contexte à partir des documents récupérés"""
        if not context_documents:
            return ""

        context_parts = []
        for i, doc_data in enumerate(context_documents[:5]):  # Limiter à 5 documents
            text = doc_data.get('text', '')
            if text:
                # Limiter la longueur de chaque document
                if len(text) > 500:
                    text = text[:500] + "..."
                context_parts.append(f"Document {i+1}: {text}")

        return "\n\n".join(context_parts)

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne des informations sur le modèle chargé"""
        return {
            'model_name': self.model_name,
            'retriever_name': self.retriever_name,
            'device': str(self.device) if self.device else 'not_initialized',
            'model_loaded': self.model is not None,
            'use_custom_retriever': self.use_custom_retriever,
            'use_gpu': self.use_gpu,
            'dependencies_available': HF_AVAILABLE
        }

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Interface compatible avec OpenAI pour les conversations

        Args:
            messages: Liste de messages au format [{"role": "user", "content": "..."}]
            **kwargs: Arguments supplémentaires

        Returns:
            Réponse au format compatible OpenAI
        """
        if not messages:
            return {
                'error': 'Aucun message fourni',
                'choices': []
            }

        # Extraire la dernière question utilisateur
        user_messages = [m for m in messages if m.get('role') == 'user']
        if not user_messages:
            return {
                'error': 'Aucun message utilisateur trouvé',
                'choices': []
            }

        last_question = user_messages[-1].get('content', '')

        response_data = self.generate_response(last_question, **kwargs)

        # Formater au style OpenAI
        return {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': response_data['response']
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'total_tokens': len(last_question.split()) + len(response_data['response'].split()),
                'completion_tokens': len(response_data['response'].split()),
                'prompt_tokens': len(last_question.split())
            },
            'model': self.model_name,
            'metadata': {
                'took_ms': response_data['took_ms'],
                'context_documents_count': len(response_data.get('context_documents', [])),
                'generation_method': response_data['generation_method']
            }
        }


# Singleton global
huggingface_rag_service = HuggingFaceRAGService()