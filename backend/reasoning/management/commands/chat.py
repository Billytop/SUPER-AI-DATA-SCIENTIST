from django.core.management.base import BaseCommand
from reasoning.agents import SQLReasoningAgent
import time

class Command(BaseCommand):
    help = 'Chat with SephlightyAI via CLI'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS("Initializing SephlightyAI Brain..."))
        agent = SQLReasoningAgent()
        time.sleep(1)
        self.stdout.write(self.style.SUCCESS("System Online. Type 'exit' to quit.\n"))
        
        while True:
            try:
                query = input(self.style.HTTP_INFO("You > "))
                if query.lower() in ['exit', 'quit']:
                    self.stdout.write("Shutting down...")
                    break
                
                if not query.strip():
                    continue

                response = agent.run(query)
                answer = response.get('answer', 'No answer')
                visual = response.get('visual', '')
                # SQL is now silent (background only, not displayed to user)

                self.stdout.write(self.style.MIGRATE_HEADING(f"AI > {answer}"))
                
                if visual == 'chart':
                     self.stdout.write(self.style.NOTICE(f"[Visual Generated: Chart Data Ready]"))

            except KeyboardInterrupt:
                self.stdout.write("\nShutting down...")
                break
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error: {e}"))
