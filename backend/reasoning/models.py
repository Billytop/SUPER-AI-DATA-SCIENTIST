from django.db import models
from django.contrib.auth.models import User

class AIQueryLog(models.Model):
    """
    Stores all user interactions with the AI for RLHF (Reinforcement Learning from Human Feedback).
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    query = models.TextField()
    generated_sql = models.TextField(blank=True, null=True)
    natural_language_response = models.TextField()
    
    response_time_ms = models.IntegerField(default=0)
    tokens_used = models.IntegerField(default=0)
    
    # Feedback
    feedback_score = models.IntegerField(default=0) # 1 = Thumbs Up, -1 = Thumbs Down
    feedback_text = models.TextField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}: {self.query[:50]}..."

class KnowledgeFact(models.Model):
    """
    Stores persistent business rules or facts for the AI.
    Example: "Fiscal year ends in March", "CEO is Jane Doe".
    """
    category = models.CharField(max_length=100, default='general') # rule, fact, policy
    content = models.TextField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"[{self.category}] {self.content[:50]}"

class FineTuningBatch(models.Model):
    """
    Tracks exports of data for OpenAI Fine-Tuning.
    """
    file_path = models.CharField(max_length=255)
    example_count = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Batch {self.id} ({self.example_count} examples)"
