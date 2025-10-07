from django.urls import path, include
from rest_framework.routers import DefaultRouter

from . import views_1 as processing_views

main_router = DefaultRouter()

processing_router = DefaultRouter(trailing_slash=True)
processing_router.register(
    r'processing/documents',
    processing_views.DocumentProcessingViewSet,
    basename='document-processing'
)

# URL patterns
urlpatterns = [
    # Main API endpoints
    path('api/v1/', include(main_router.urls)),

    path('api/v1/', include(processing_router.urls)),

    # Processing statistics endpoint
    path('api/v1/processing/statistics/',
         processing_views.ProcessingStatisticsView.as_view(),
         name='processing-statistics'),

    path('api/v1/processing/search/',
         processing_views.SearchContentView.as_view(),
         name='content-search'),

    # RAG semantic search endpoint (multimode)
    path('api/v1/processing/rag/query/',
         processing_views.RAGQueryView.as_view(),
         name='rag-query'),

    # Hugging Face RAG generation endpoint
    path('api/v1/processing/rag/generate/',
         processing_views.HuggingFaceRAGView.as_view(),
         name='rag-generate'),

    # Chat completion endpoint (compatible OpenAI)
    path('api/v1/processing/rag/chat/completions/',
         processing_views.ChatCompletionView.as_view(),
         name='rag-chat-completions'),

    # RAG model information endpoint
    path('api/v1/processing/rag/models/info/',
         processing_views.RAGModelInfoView.as_view(),
         name='rag-model-info'),

    # Endpoint non-protégé pour tous les contenus extraits
    path('api/v1/public/all-extracted-content/',
         processing_views.AllExtractedContentView.as_view(),
         name='all-extracted-content'),

    # Endpoint public pour la structure d'un document (sections et paragraphes)
    path('api/v1/public/documents/<uuid:document_id>/structure/',
         processing_views.DocumentStructureView.as_view(),
         name='document-structure'),

    # Endpoint pour récupérer le contenu par page d'un document
    path('api/v1/public/documents/<uuid:document_id>/pages/',
         processing_views.DocumentPageContentView.as_view(),
         name='document-page-content'),
]

app_name = 'doctrine'