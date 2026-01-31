from django.core.management.base import BaseCommand
from messaging.services import WhatsAppService, EmailService, TelegramService #, SlackService
import asyncio

class Command(BaseCommand):
    help = 'Test messaging integrations'

    def add_arguments(self, parser):
        parser.add_argument('--channel', type=str, help='whatsapp, email, or telegram')
        parser.add_argument('--target', type=str, help='Phone number, Email, or Chat ID')

    def handle(self, *args, **kwargs):
        channel = kwargs['channel']
        target = kwargs['target']
        
        if channel == 'whatsapp':
            self.stdout.write(f"Sending WhatsApp to {target}...")
            sid = WhatsAppService.send_message(target, "Hello from SephlightyAI! 🚀")
            if sid:
                self.stdout.write(self.style.SUCCESS(f"Success! SID: {sid}"))
            else:
                self.stdout.write(self.style.ERROR("Failed. Check logs/credentials."))
                
        elif channel == 'email':
            self.stdout.write(f"Sending Email to {target}...")
            success = EmailService.send_report(target, "SephlightyAI Test", "<h1>It Works!</h1>")
            if success:
                self.stdout.write(self.style.SUCCESS("Email Sent!"))
            else:
                 self.stdout.write(self.style.ERROR("Failed."))
                 
        else:
            self.stdout.write("Please specify --channel and --target")
