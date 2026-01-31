from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from django.db.models import Count
import time
import logging

logger = logging.getLogger(__name__)

from .models import Conversation, Message
from .serializers import (
    ConversationSerializer,
    ConversationListSerializer,
    MessageSerializer,
    MessageCreateSerializer
)


class ConversationViewSet(viewsets.ModelViewSet):
    """CRUD operations for conversations"""
    permission_classes = [IsAuthenticated]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return ConversationListSerializer
        return ConversationSerializer
    
    def get_queryset(self):
        """Filter by user's organization"""
        user = self.request.user
        try:
            org = user.profile.organization
            return Conversation.objects.filter(
                organization=org
            ).annotate(
                message_count=Count('messages')
            ).order_by('-updated_at')
        except:
            return Conversation.objects.none()
    
    def perform_create(self, serializer):
        """Auto-set organization and user"""
        user = self.request.user
        serializer.save(
            user=user,
            organization=user.profile.organization
        )
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Get all messages for a conversation"""
        conversation = self.get_object()
        messages = conversation.messages.all().order_by('created_at')
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def send_message(self, request, pk=None):
        """Send a message and get AI response"""
        conversation = self.get_object()
        serializer = MessageCreateSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        user_content = serializer.validated_data['content']
        
        # Create user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=user_content
        )
        
        # Get AI response
        try:
            from laravel_modules.ai_brain.central_integration_layer import CentralAI
            
            start_time = time.time()
            agent = CentralAI()
            agent.initialize()
            
            context = {
                "connection_id": "TENANT_001",
                "user_id": request.user.id
            }
            
            response = agent.query(user_content, context)
            processing_time = time.time() - start_time
            
            # Create assistant message
            assistant_message = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=response.get('answer', 'No response'),
                intent=response.get('intent'),
                sql_query=response.get('sql'),
                processing_time=processing_time
            )
            
            # Update conversation title if first message
            if conversation.message_count == 0:
                # Use first few words as title
                title = ' '.join(user_content.split()[:7])
                if len(title) > 50:
                    title = title[:47] + '...'
                conversation.title = title
                conversation.save()
            
            conversation.updated_at = timezone.now()
            conversation.save()
            
            return Response({
                'user_message': MessageSerializer(user_message).data,
                'assistant_message': MessageSerializer(assistant_message).data,
                'data': response.get('data'), # Ephemeral key for frontend
                'processing_time': processing_time
            })
            
        except Exception as e:
            # Create error message
            error_message = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=f"⚠️ **Error:** {str(e)}"
            )
            
            return Response({
                'user_message': MessageSerializer(user_message).data,
                'assistant_message': MessageSerializer(error_message).data,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_conversation(request):
    """Create a new conversation"""
    user = request.user
    title = request.data.get('title', 'New Conversation')
    
    conversation = Conversation.objects.create(
        organization=user.profile.organization,
        user=user,
        title=title
    )
    
    return Response(
        ConversationSerializer(conversation).data,
        status=status.HTTP_201_CREATED
    )


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_message(request, message_id):
    """Delete a specific message"""
    try:
        message = Message.objects.get(id=message_id)
        
        # Check ownership
        if message.conversation.user != request.user:
            return Response(
                {'error': 'Permission denied'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        message.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
        
    except Message.DoesNotExist:
        return Response(
            {'error': 'Message not found'},
            status=status.HTTP_404_NOT_FOUND
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat_api(request):
    """
    Simplified chat endpoint for the frontend.
    Auto-creates or retrieves a 'General' conversation and routes the message.
    """
    user = request.user
    message_content = request.data.get('message') or request.data.get('content')
    conversation_id = request.data.get('conversation_id')
    
    if not message_content:
        return Response({'error': 'Message is required (use "message" or "content" key)'}, status=status.HTTP_400_BAD_REQUEST)

    # 1. Find or create a conversation for this session/user
    try:
        conversation = None
        if conversation_id:
            conversation = Conversation.objects.filter(id=conversation_id, user=user).first()
        
        if not conversation:
            # Check if profile exists; fail gracefully if not
             if hasattr(user, 'profile') and user.profile and user.profile.organization:
                  org = user.profile.organization
             else:
                  # Fallback/Edge case
                  logger.warning(f"User {user.username} (ID: {user.id}) tried to chat but has no profile or organization.")
                  return Response({
                      'error': 'User profile or organization missing. Please ensure your account is fully set up.',
                      'detail': 'SaaS context requires a linked organization.'
                  }, status=status.HTTP_400_BAD_REQUEST)

             # Initial Title (will be refined later)
             title = ' '.join(message_content.split()[:5])
             if len(title) > 50: title = title[:47] + "..."
             if not title: title = "New Conversation"

             conversation = Conversation.objects.create(
                 organization=org,
                 user=user,
                 title=title
             )
    except Exception as e:
         return Response({'error': f'Configuration error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # 2. Save User Message
    Message.objects.create(
        conversation=conversation,
        role='user',
        content=message_content
    )

    # 3. Run AI Agent
    try:
        from laravel_modules.ai_brain.central_integration_layer import CentralAI
        import time
        
        start_time = time.time()
        # Use CentralAI for consistent multi-tenant / multi-logic routing
        agent = CentralAI()
        agent.initialize()
        
        # Prepare context for CentralAI
        context = {
            "connection_id": "TENANT_001", # Default for now
            "user_id": user.id,
            "conversation_id": conversation.id
        }
        
        ai_response = agent.query(message_content, context)
        execution_time = time.time() - start_time

        # Translate CentralAI format to Message content
        # Note: CentralAI returns a Dict with 'answer', 'intent', etc.
        answer = ai_response.get("answer", "No response")
        intent = ai_response.get("intent", "UNKNOWN")
        sql_query = ai_response.get("sql") # CentralAI uses 'sql' key

        # 4. Save AI Message
        Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=answer,
            intent=intent,
            sql_query=sql_query,
            processing_time=execution_time
        )

        # 5. Dynamic AI Titling (Advanced)
        # Re-evaluate title on message pairs to capture the 'real' topic
        msg_count = conversation.messages.count()
        if msg_count in [2, 4, 6]: # Check after complete pairs
             try:
                 history = [
                     {"role": m.role, "content": m.content} 
                     for m in conversation.messages.all().order_by('created_at')
                 ]
                 new_title = agent.generate_conversation_title(history)
                 if new_title and len(new_title) > 3:
                     conversation.title = new_title
                     conversation.save()
             except Exception as e:
                 logger.error(f"Failed to refine title: {e}")

        # 6. Return Format matching Chat.tsx
        return Response({
            'response': answer,
            'confidence': ai_response.get('confidence', 95),
            'intent': intent,
            'data': ai_response.get('data', []),
            'sql': sql_query,
            'execution_time': execution_time,
            'insights': ai_response.get('insights', []),
            'conversation_id': str(conversation.id),
            'title': conversation.title
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({
            'response': f"System Error: {str(e)}",
            'confidence': 0
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

