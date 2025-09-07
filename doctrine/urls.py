from django.urls import path, include
from rest_framework.routers import DefaultRouter

from . import views_1 as processing_views


# Main router for standard viewsets
main_router = DefaultRouter()

# Uncomment and configure these viewsets as needed
# main_router.register(r'users', views.UserViewSet, basename='user')
# main_router.register(r'themes', views.ThemeViewSet, basename='theme')
# main_router.register(r'document-categories', views.DocumentCategoryViewSet, basename='documentcategory')
# main_router.register(r'documents', views.DocumentViewSet, basename='document')
# main_router.register(r'document-contents', views.DocumentContentViewSet, basename='documentcontent')
# main_router.register(r'topics', views.TopicsViewSet, basename='topic')
# main_router.register(r'sections', views.SectionViewSet, basename='section')
# main_router.register(r'paragraphs', views.ParagraphViewSet, basename='paragraph')
# main_router.register(r'tables', views.TableViewSet, basename='table')

# Processing router for document processing viewset
processing_router = DefaultRouter()
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
]

app_name = 'dgi_extractor_api'