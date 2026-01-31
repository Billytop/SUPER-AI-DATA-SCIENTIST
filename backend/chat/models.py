import uuid
from django.db import models
from django.contrib.auth.models import User
from auth_app.models import Organization


class Conversation(models.Model):
    """Conversation/Chat session"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='conversations')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    
    # Info
    title = models.CharField(
        max_length=255, 
        help_text="Auto-generated from first message or user-defined"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Status
    is_archived = models.BooleanField(default=False)
    is_pinned = models.BooleanField(default=False)
    
    # Metadata
    metadata = models.JSONField(
        default=dict,
        help_text="Tags, category, custom fields"
    )
    
    class Meta:
        db_table = 'conversations'
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['organization', '-updated_at']),
            models.Index(fields=['user', '-updated_at']),
        ]
    
    def __str__(self):
        return f"{self.title} - {self.user.username}"
    


class Message(models.Model):
    """Individual message in a conversation"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    
    # Role
    role = models.CharField(
        max_length=10,
        choices=[
            ('user', 'User'),
            ('assistant', 'Assistant'),
            ('system', 'System')
        ]
    )
    
    # Content
    content = models.TextField(help_text="Message text")
    
    # AI Metadata (for assistant messages)
    intent = models.CharField(
        max_length=50, 
        null=True, 
        blank=True,
        help_text="Detected intent: SALES, INVENTORY, etc."
    )
    sql_query = models.TextField(
        null=True, 
        blank=True,
        help_text="SQL executed (for debugging)"
    )
    
    # Attachments
    chart_path = models.CharField(
        max_length=500, 
        null=True, 
        blank=True,
        help_text="Path to generated chart"
    )
    export_paths = models.JSONField(
        default=list,
        help_text="Paths to exported files [excel, pdf, etc.]"
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    tokens = models.IntegerField(
        default=0,
        help_text="Token count for billing"
    )
    processing_time = models.FloatField(
        null=True,
        blank=True,
        help_text="Response time in seconds"
    )
    
    class Meta:
        db_table = 'messages'
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['conversation', 'created_at']),
        ]
    
    def __str__(self):
        preview = self.content[:50] + '...' if len(self.content) > 50 else self.content
        return f"{self.role}: {preview}"
