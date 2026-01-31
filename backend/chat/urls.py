from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'conversations', views.ConversationViewSet, basename='conversation')

urlpatterns = [
    path('', include(router.urls)),
    path('messages/<uuid:message_id>', views.delete_message, name='delete-message'),
    path('chat/', views.chat_api, name='chat_api'),
]
