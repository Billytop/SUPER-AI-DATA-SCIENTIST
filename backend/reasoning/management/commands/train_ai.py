import json
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from reasoning.models import AIQueryLog, FineTuningBatch

class Command(BaseCommand):
    help = 'Exports high-quality AI interactions for Fine-Tuning'

    def handle(self, *args, **kwargs):
        self.stdout.write("Scanning for positive feedback interactions...")
        
        # 1. Select High-Quality Data (Feedback > 0)
        logs = AIQueryLog.objects.filter(feedback_score__gt=0)
        
        if not logs.exists():
            self.stdout.write(self.style.WARNING("No positive feedback logs found. Cannot generate training data."))
            return

        # 2. Format for OpenAI (JSONL)
        # {"messages": [{"role": "system", "content": "Marv is a chatbot that answers questions with SQL."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris", "weight": 0}, {"role": "user", "content": "What's the capital of Germany?"}, {"role": "assistant", "content": "Berlin", "weight": 1}]}
        training_data = []
        for log in logs:
            entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are SephlightyAI – an ENTERPRISE-GRADE HYBRID BUSINESS BRAIN. "
                            "You behave like a combined LSTM + Transformer engine with a business knowledge graph. "
                            "You generate accurate SQL, a clear explanation, and a business-ready answer "
                            "for each query, in English or Swahili, following the same behavior defined in "
                            "backend.reasoning.prompts.SYSTEM_ROLE."
                        ),
                    },
                    {"role": "user", "content": log.query},
                    {"role": "assistant", "content": log.natural_language_response},
                ]
            }
            training_data.append(entry)
            
        # 3. Save to File
        export_dir = os.path.join(settings.BASE_DIR, 'reasoning', 'finetuning')
        os.makedirs(export_dir, exist_ok=True)
        
        filename = f"training_batch_{len(FineTuningBatch.objects.all()) + 1}.jsonl"
        filepath = os.path.join(export_dir, filename)
        
        with open(filepath, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")
                
        # 4. Record Batch
        FineTuningBatch.objects.create(file_path=filepath, example_count=len(training_data))
        
        self.stdout.write(self.style.SUCCESS(f"Exported {len(training_data)} examples to {filepath}"))
