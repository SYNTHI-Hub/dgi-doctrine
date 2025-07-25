from django.urls import path, include
from rest_framework.documentation import include_docs_urls
from rest_framework.routers import DefaultRouter
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView

from . import views
from . import views_1 as processing_views


main_router = DefaultRouter()

main_router.register(r'users', views.UserViewSet, basename='user')
main_router.register(r'themes', views.ThemeViewSet, basename='theme')
main_router.register(r'document-categories', views.DocumentCategoryViewSet, basename='documentcategory')
main_router.register(r'documents', views.DocumentViewSet, basename='document')
main_router.register(r'document-contents', views.DocumentContentViewSet, basename='documentcontent')
main_router.register(r'topics', views.TopicViewSet, basename='topic')
main_router.register(r'sections', views.SectionViewSet, basename='section')
main_router.register(r'paragraphs', views.ParagraphViewSet, basename='paragraph')
main_router.register(r'tables', views.TableViewSet, basename='table')


processing_router = DefaultRouter()

processing_router.register(
    r'processing/documents',
    processing_views.DocumentProcessingViewSet,
    basename='document-processing'
)


urlpatterns = [
    path('api/v1/', include(main_router.urls)),

    path('api/v1/', include(processing_router.urls)),

    path('api/v1/processing/statistics/',
         processing_views.processing_statistics,
         name='processing-statistics'),
    path('api/v1/processing/search/',
         processing_views.search_content,
         name='content-search'),

    path('api/docs/', include_docs_urls(title='Document Management API')),
    path('api-auth/', include('rest_framework.urls')),

]

app_name = 'dgi_extractor_api'



"""
    path('api/v1/schema/',
         SpectacularAPIView.as_view(),
         name='schema'),
    path('api/v1/docs/',
         SpectacularSwaggerView.as_view(url_name='schema'),
         name='swagger-ui'),
    path('api/v1/redoc/',
         SpectacularRedocView.as_view(url_name='schema'),
         name='redoc'),
""",