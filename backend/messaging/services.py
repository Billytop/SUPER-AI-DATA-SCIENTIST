import os
import logging
from io import BytesIO
from datetime import datetime
import pandas as pd
from xhtml2pdf import pisa
from django.template.loader import render_to_string
from twilio.rest import Client
from telegram import Bot
from django.core.mail import send_mail, EmailMessage
from django.conf import settings

logger = logging.getLogger(__name__)

class WhatsAppService:
    @staticmethod
    def send_message(to_number, body):
        try:
            account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
            auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
            from_number = os.environ.get('TWILIO_WHATSAPP_NUMBER')
            
            if not (account_sid and auth_token and from_number):
                logger.warning("Twilio credentials not set. Skipping WhatsApp.")
                return False

            client = Client(account_sid, auth_token)
            message = client.messages.create(
                body=body,
                from_=from_number,
                to=f"whatsapp:{to_number}"
            )
            return message.sid
        except Exception as e:
            logger.error(f"WhatsApp Error: {e}")
            return False

class TelegramService:
    @staticmethod
    async def send_alert(chat_id, message):
        try:
            token = os.environ.get('TELEGRAM_BOT_TOKEN')
            if not token:
                logger.warning("Telegram Token not set.")
                return False
                
            bot = Bot(token=token)
            await bot.send_message(chat_id=chat_id, text=message)
            return True
        except Exception as e:
            logger.error(f"Telegram Error: {e}")
            return False

class EmailService:
    @staticmethod
    def send_report(to_email, subject, html_content, pdf_attachment=None):
        try:
            msg = EmailMessage(
                subject,
                html_content,
                settings.DEFAULT_FROM_EMAIL,
                [to_email],
            )
            msg.content_subtype = "html"
            
            if pdf_attachment:
                msg.attach('report.pdf', pdf_attachment, 'application/pdf')
                
            msg.send()
            return True
        except Exception as e:
            logger.error(f"Email Error: {e}")
            return False

class ExportService:
    """Export data to Excel, PDF, or CSV formats"""
    
    @staticmethod
    def to_excel(data, title="Report", output_path=None):
        """Export DataFrame or dict to Excel"""
        try:
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
            
            if not output_path:
                downloads = os.path.join(os.path.expanduser("~"), "Downloads")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(downloads, f"{title}_{timestamp}.xlsx")
            
            df.to_excel(output_path, index=False, engine='openpyxl')
            return output_path
        except Exception as e:
            logger.error(f"Excel Export Error: {e}")
            return None
    
    @staticmethod
    def to_pdf(html_content, title="Report", output_path=None):
        """Convert HTML to PDF"""
        try:
            if not output_path:
                downloads = os.path.join(os.path.expanduser("~"), "Downloads")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(downloads, f"{title}_{timestamp}.pdf")
            
            with open(output_path, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
            
            return output_path if not pisa_status.err else None
        except Exception as e:
            logger.error(f"PDF Export Error: {e}")
            return None
    
    @staticmethod
    def to_csv(data, title="Report", output_path=None):
        """Export DataFrame or dict to CSV"""
        try:
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
            
            if not output_path:
                downloads = os.path.join(os.path.expanduser("~"), "Downloads")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(downloads, f"{title}_{timestamp}.csv")
            
            df.to_csv(output_path, index=False)
            return output_path
        except Exception as e:
            logger.error(f"CSV Export Error: {e}")
            return None
