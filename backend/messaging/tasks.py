from celery import shared_task
from .services import WhatsAppService, EmailService
import asyncio

@shared_task
def send_whatsapp_alert(to_number, message):
    """
    Async task to send WhatsApp alert.
    """
    WhatsAppService.send_message(to_number, message)
    return f"WhatsApp sent to {to_number}"

@shared_task
def send_email_report(to_email, subject, body):
    """
    Async task to send Email.
    """
    EmailService.send_report(to_email, subject, body)
    return f"Email sent to {to_email}"
