from django.urls import path
from . import views

urlpatterns = [
    path('register', views.register, name='register'),
    path('login', views.login, name='login'),
    path('logout', views.logout, name='logout'),
    path('me', views.me, name='me'),
    path('preferences', views.update_user_preferences, name='user-preferences'),
    path('organization/users', views.get_organization_users, name='organization-users'),
]
