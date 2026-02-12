
import os
import datetime
from fpdf import FPDF
import logging

logger = logging.getLogger("OMNIBRAIN_PDF")
logger.setLevel(logging.INFO)

class PDFReportGenerator:
    """
    Sovereign PDF Engine: Generates professional financial documents.
    """
    def __init__(self, output_dir="public/media/reports"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
    def generate_ledger_pdf(self, customer_name, transactions):
        """
        Generates a PDF Ledger for a specific customer.
        transactions: List of dicts {date, ref, type, total, status}
        """
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, f"Customer Ledger: {customer_name}", ln=True, align="C")
            pdf.set_font("Arial", "I", 10)
            pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
            pdf.ln(10)
            
            # Table Header
            pdf.set_font("Arial", "B", 10)
            pdf.set_fill_color(200, 220, 255)
            pdf.cell(30, 10, "Date", 1, 0, "C", True)
            pdf.cell(40, 10, "Reference", 1, 0, "C", True)
            pdf.cell(30, 10, "Type", 1, 0, "C", True)
            pdf.cell(40, 10, "Amount (TZS)", 1, 0, "C", True)
            pdf.cell(30, 10, "Status", 1, 1, "C", True)
            
            # Table Rows
            pdf.set_font("Arial", "", 10)
            total_debt = 0
            
            for t in transactions:
                pdf.cell(30, 10, str(t.get('date', '')), 1)
                pdf.cell(40, 10, str(t.get('ref', '')), 1)
                pdf.cell(30, 10, str(t.get('type', '')).upper(), 1)
                
                amount = float(t.get('total', 0))
                if t.get('type') == 'sell' and t.get('status') != 'paid':
                    total_debt += amount
                    
                pdf.cell(40, 10, f"{amount:,.2f}", 1, 0, "R")
                pdf.cell(30, 10, str(t.get('status', '')).title(), 1, 1)
                
            pdf.ln(10)
            
            # Summary
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Total Outstanding Debt: {total_debt:,.2f} TZS", ln=True, align="R")
            
            # Save
            filename = f"Ledger_{customer_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            pdf.output(filepath)
            
            # Return relative URL for frontend
            return f"/media/reports/{filename}"
            
        except Exception as e:
            logger.error(f"PDF Generation Failed: {e}")
            return None
