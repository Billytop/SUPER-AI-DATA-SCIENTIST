from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import AIQueryLog
from .agents import SQLReasoningAgent
import time

class AskAIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        query = request.data.get('query')
        if not query:
            return Response({"error": "No query provided"}, status=400)
            
        start_time = time.time()
        
        # Initialize Agent with User Context
        agent = SQLReasoningAgent(user_id=request.user.id)
        
        # Run Reasoning
        try:
            result = agent.run(query)
            answer = result['answer']
            sql = result.get('sql', '')
        except Exception as e:
            return Response({"error": str(e)}, status=500)
            
        duration = int((time.time() - start_time) * 1000)
        
        # Log Interaction
        AIQueryLog.objects.create(
            user=request.user,
            query=query,
            generated_sql=sql,
            natural_language_response=answer,
            response_time_ms=duration
        )
        
        return Response({
            "answer": answer,
            "sql": sql,
            "duration": duration
        })
