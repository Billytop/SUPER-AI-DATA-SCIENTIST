from rest_framework import serializers
from .models import Conversation, Message
from django.contrib.auth.models import User


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'conversation', 'role', 'content', 'intent', 'sql_query',
                  'chart_path', 'export_paths', 'created_at', 'tokens', 'processing_time']
        read_only_fields = ['id', 'created_at']


class ConversationSerializer(serializers.ModelSerializer):
    message_count = serializers.SerializerMethodField()
    messages = MessageSerializer(many=True, read_only=True)
    user_name = serializers.CharField(source='user.get_full_name', read_only=True)
    
    def get_message_count(self, obj):
        if hasattr(obj, 'message_count'):
            return obj.message_count
        return obj.messages.count()
    
    class Meta:
        model = Conversation
        fields = ['id', 'organization', 'user', 'user_name', 'title', 'created_at', 
                  'updated_at', 'is_archived', 'is_pinned', 'metadata', 'message_count', 'messages']
        read_only_fields = ['id', 'created_at', 'updated_at', 'organization', 'user']


class ConversationListSerializer(serializers.ModelSerializer):
    """Lightweight version for listing conversations"""
    message_count = serializers.SerializerMethodField()
    user_name = serializers.CharField(source='user.get_full_name', read_only=True)
    last_message = serializers.SerializerMethodField()

    def get_message_count(self, obj):
        if hasattr(obj, 'message_count'):
            return obj.message_count
        return obj.messages.count()
    
    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'updated_at', 'is_archived', 
                  'is_pinned', 'message_count', 'user_name', 'last_message']
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_last_message(self, obj):
        last_msg = obj.messages.order_by('-created_at').first()
        if last_msg:
            return {
                'content': last_msg.content[:100],
                'role': last_msg.role,
                'created_at': last_msg.created_at
            }
        return None


class MessageCreateSerializer(serializers.Serializer):
    """For creating a new user message"""
    content = serializers.CharField(required=False)
    message = serializers.CharField(required=False)
    conversation_id = serializers.UUIDField(required=False)
    
    def validate(self, data):
        content = data.get('content') or data.get('message')
        if not content or not content.strip():
            raise serializers.ValidationError("Either 'content' or 'message' is required and must not be empty.")
        data['content'] = content.strip() # Normalize to 'content'
        return data
