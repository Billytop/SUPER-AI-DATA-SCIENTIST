
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat_api(request):
    """
    Simplified chat endpoint for the frontend.
    Auto-creates or retrieves a 'General' conversation and routes the message.
    """
    user = request.user
    message_content = request.data.get('message')
    
    if not message_content:
        return Response({'error': 'Message is required'}, status=status.HTTP_400_BAD_REQUEST)

    # 1. Find or create a conversation for this session/user
    # For simplicity, we'll try to get the most recent one or create new
    try:
        conversation = Conversation.objects.filter(user=user).order_by('-updated_at').first()
        if not conversation:
            conversation = Conversation.objects.create(
                organization=user.profile.organization,
                user=user,
                title='General Chat'
            )
    except Exception as e:
         # Fallback if profile/org is missing
         return Response({'error': f'Configuration error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # 2. Save User Message
    Message.objects.create(
        conversation=conversation,
        role='user',
        content=message_content
    )

    # 3. Run AI Agent
    try:
        from reasoning.agents import SQLReasoningAgent
        import time
        
        start_time = time.time()
        agent = SQLReasoningAgent(user_id=user.id)
        ai_response = agent.run(message_content)
        execution_time = time.time() - start_time

        # 4. Save AI Message
        Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=ai_response.get('answer', 'No response'),
            intent=ai_response.get('intent'),
            sql_query=ai_response.get('_internal_sql'),
            processing_time=execution_time
        )

        # 5. Return Format matching Chat.tsx
        return Response({
            'response': ai_response.get('answer'),
            'confidence': 95, # Mock/Default
            'intent': ai_response.get('intent'),
            'data': ai_response.get('data', []),
            'sql': ai_response.get('_internal_sql'),
            'execution_time': execution_time,
            'insights': ai_response.get('insights', [])
        })

    except Exception as e:
        return Response({
            'response': f"System Error: {str(e)}",
            'confidence': 0
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
