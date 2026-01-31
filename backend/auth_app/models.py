import uuid
from django.db import models
from django.contrib.auth.models import User


class Organization(models.Model):
    """Multi-tenant organization model - each business that signs up"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, help_text="Company/Organization name")
    slug = models.SlugField(max_length=100, unique=True, help_text="URL-friendly identifier")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Subscription
    subscription_tier = models.CharField(
        max_length=50, 
        choices=[
            ('free', 'Free'),
            ('pro', 'Professional'),
            ('enterprise', 'Enterprise')
        ],
        default='free'
    )
    is_active = models.BooleanField(default=True)
    
    # Settings (JSON field for flexibility)
    settings = models.JSONField(
        default=dict,
        help_text="Organization-wide settings like theme, language, etc."
    )
    
    # Metadata
    max_users = models.IntegerField(default=5, help_text="Max users allowed")
    max_conversations = models.IntegerField(default=100, help_text="Max conversations")
    
    class Meta:
        db_table = 'organizations'
        ordering = ['-created_at']
    
    def __str__(self):
        return self.name


class UserProfile(models.Model):
    """Extended user profile linked to organization"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='users')
    
    # Role
    role = models.CharField(
        max_length=20,
        choices=[
            ('admin', 'Administrator'),
            ('manager', 'Manager'),
            ('user', 'User')
        ],
        default='user'
    )
    
    # Profile
    avatar_url = models.URLField(max_length=500, null=True, blank=True)
    phone = models.CharField(max_length=20, null=True, blank=True)
    
    # Preferences (JSON)
    preferences = models.JSONField(
        default=dict,
        help_text="User preferences: theme, language, notifications, etc."
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    last_active = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'user_profiles'
        unique_together = ('user', 'organization')
    
    def __str__(self):
        return f"{self.user.get_full_name() or self.user.username} - {self.organization.name}"
