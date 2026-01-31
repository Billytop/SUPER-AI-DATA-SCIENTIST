"""
PDF Report Generator
Creates professional PDF reports with charts, tables, and branding.
"""

from typing import Dict, List, Optional
from datetime import datetime
import io


class PDFReportGenerator:
    """
    Generates PDF reports for business data.
    Note: This is a framework. In production, would use ReportLab or WeasyPrint.
    """
    
    def __init__(self, company_name: str = "SephlightyERP"):
        self.company_name = company_name
        self.report_templates = self._build_templates()
        
    def generate_sales_report(self, data: Dict, period: str) -> Dict:
        """
        Generate sales performance PDF report.
        
        Args:
            data: Sales data
            period: Report period (e.g., "January 2026")
            
        Returns:
            Dict with report metadata and content structure
        """
        report = {
            'title': f"Sales Performance Report - {period}",
            'generated_at': datetime.now().isoformat(),
            'company': self.company_name,
            'sections': []
        }
        
        # Executive Summary
        report['sections'].append({
            'type': 'summary',
            'title': 'Executive Summary',
            'content': self._format_executive_summary(data)
        })
        
        # Key Metrics
        report['sections'].append({
            'type': 'metrics',
            'title': 'Key Performance Indicators',
            'metrics': [
                {'label': 'Total Sales', 'value': data.get('total_sales', 0), 'format': 'currency'},
                {'label': 'Transactions', 'value': data.get('transaction_count', 0), 'format': 'number'},
                {'label': 'Average Order Value', 'value': data.get('avg_order_value', 0), 'format': 'currency'},
                {'label': 'Growth vs Previous Period', 'value': data.get('growth_percent', 0), 'format': 'percentage'}
            ]
        })
        
        # Sales Breakdown
        if 'category_breakdown' in data:
            report['sections'].append({
                'type': 'table',
                'title': 'Sales Breakdown by Category',
                'headers': ['Category', 'Sales', 'Transactions', '% of Total'],
                'rows': self._format_category_breakdown(data['category_breakdown'])
            })
        
        # Top Products
        if 'top_products' in data:
            report['sections'].append({
                'type': 'table',
                'title': 'Top 10 Products',
                'headers': ['Product', 'Units Sold', 'Revenue'],
                'rows': self._format_top_products(data['top_products'])
            })
        
        # Trend Chart
        if 'daily_sales' in data:
            report['sections'].append({
                'type': 'chart',
                'title': 'Daily Sales Trend',
                'chart_type': 'line',
                'data': data['daily_sales']
            })
        
        # Recommendations
        report['sections'].append({
            'type': 'recommendations',
            'title': 'Recommendations',
            'items': self._generate_recommendations(data)
        })
        
        return report
    
    def generate_inventory_report(self, data: Dict) -> Dict:
        """Generate inventory status PDF report."""
        report = {
            'title': 'Inventory Status Report',
            'generated_at': datetime.now().isoformat(),
            'company': self.company_name,
            'sections': []
        }
        
        # Summary
        report['sections'].append({
            'type': 'metrics',
            'title': 'Inventory Overview',
            'metrics': [
                {'label': 'Total Inventory Value', 'value': data.get('total_value', 0), 'format': 'currency'},
                {'label': 'Total Items', 'value': data.get('item_count', 0), 'format': 'number'},
                {'label': 'Low Stock Items', 'value': data.get('low_stock_count', 0), 'format': 'number'},
                {'label': 'Out of Stock Items', 'value': data.get('out_of_stock_count', 0), 'format': 'number'}
            ]
        })
        
        # Low Stock Alert
        if 'low_stock_items' in data:
            report['sections'].append({
                'type': 'table',
                'title': '⚠️ Low Stock Alert',
                'headers': ['Product', 'Current Stock', 'Reorder Point', 'Status'],
                'rows': self._format_low_stock_items(data['low_stock_items']),
                'highlight': 'warning'
            })
        
        # Inventory by Category
        if 'category_inventory' in data:
            report['sections'].append({
                'type': 'table',
                'title': 'Inventory Value by Category',
                'headers': ['Category', 'Items', 'Total Value', '% of Total'],
                'rows': data['category_inventory']
            })
        
        return report
    
    def generate_customer_report(self, data: Dict) -> Dict:
        """Generate customer analysis PDF report."""
        report = {
            'title': 'Customer Analysis Report',
            'generated_at': datetime.now().isoformat(),
            'company': self.company_name,
            'sections': []
        }
        
        # Customer Metrics
        report['sections'].append({
            'type': 'metrics',
            'title': 'Customer Metrics',
            'metrics': [
                {'label': 'Total Customers', 'value': data.get('total_customers', 0), 'format': 'number'},
                {'label': 'Active Customers', 'value': data.get('active_customers', 0), 'format': 'number'},
                {'label': 'Total Outstanding Debt', 'value': data.get('total_debt', 0), 'format': 'currency'},
                {'label': 'Average Customer Value', 'value': data.get('avg_customer_value', 0), 'format': 'currency'}
            ]
        })
        
        # Top Customers
        if 'top_customers' in data:
            report['sections'].append({
                'type': 'table',
                'title': 'Top 20 Customers by Revenue',
                'headers': ['Customer', 'Total Purchases', 'Transactions', 'Avg Order'],
                'rows': data['top_customers']
            })
        
        # Debt Aging
        if 'debt_aging' in data:
            report['sections'].append({
                'type': 'table',
                'title': 'Debt Aging Analysis',
                'headers': ['Customer', 'Total Debt', '0-30 Days', '31-60 Days', '60+ Days'],
                'rows': data['debt_aging'],
                'highlight': 'debt'
            })
        
        return report
    
    def generate_financial_report(self, data: Dict, period: str) -> Dict:
        """Generate financial performance PDF report."""
        report = {
            'title': f"Financial Performance Report - {period}",
            'generated_at': datetime.now().isoformat(),
            'company': self.company_name,
            'sections': []
        }
        
        # Financial Summary
        report['sections'].append({
            'type': 'metrics',
            'title': 'Financial Summary',
            'metrics': [
                {'label': 'Total Revenue', 'value': data.get('revenue', 0), 'format': 'currency'},
                {'label': 'Total Expenses', 'value': data.get('expenses', 0), 'format': 'currency'},
                {'label': 'Net Profit', 'value': data.get('profit', 0), 'format': 'currency'},
                {'label': 'Profit Margin', 'value': data.get('margin_percent', 0), 'format': 'percentage'}
            ]
        })
        
        # Income Statement
        report['sections'].append({
            'type': 'table',
            'title': 'Income Statement',
            'headers': ['Line Item', 'Amount'],
            'rows': [
                ['Sales Revenue', f"{data.get('revenue', 0):,.2f} TZS"],
                ['Cost of Goods Sold', f"{data.get('cogs', 0):,.2f} TZS"],
                ['Gross Profit', f"{data.get('gross_profit', 0):,.2f} TZS"],
                ['Operating Expenses', f"{data.get('operating_expenses', 0):,.2f} TZS"],
                ['Net Profit', f"{data.get('profit', 0):,.2f} TZS"]
            ]
        })
        
        # Cash Flow
        if 'cash_flow' in data:
            report['sections'].append({
                'type': 'chart',
                'title': 'Cash Flow Trend',
                'chart_type': 'area',
                'data': data['cash_flow']
            })
        
        return report
    
    def export_to_pdf_content(self, report: Dict) -> str:
        """
        Convert report structure to PDF-ready HTML content.
        
        Args:
            report: Report structure dict
            
        Returns:
            HTML string ready for PDF conversion
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
                .company {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .title {{ font-size: 20px; margin-top: 10px; }}
                .date {{ color: #7f8c8d; font-size: 12px; }}
                .section {{ margin-top: 30px; page-break-inside: avoid; }}
                .section-title {{ font-size: 16px; font-weight: bold; color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 15px; }}
                .metric {{ flex: 1; min-width: 200px; padding: 15px; background: #ecf0f1; border-left: 4px solid #3498db; }}
                .metric-label {{ font-size: 12px; color: #7f8c8d; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th {{ background: #34495e; color: white; padding: 10px; text-align: left; }}
                td {{ padding: 8px; border-bottom: 1px solid #ecf0f1; }}
                tr:hover {{ background: #f8f9fa; }}
                .warning {{ background: #fff3cd; }}
                ul {{ margin-top: 10px; }}
                li {{ margin-bottom: 8px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="company">{report['company']}</div>
                <div class="title">{report['title']}</div>
                <div class="date">Generated: {report['generated_at']}</div>
            </div>
        """
        
        for section in report['sections']:
            html += self._render_section(section)
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _render_section(self, section: Dict) -> str:
        """Render a report section as HTML."""
        html = f'<div class="section"><h2 class="section-title">{section["title"]}</h2>'
        
        if section['type'] == 'metrics':
            html += '<div class="metrics">'
            for metric in section['metrics']:
                value = self._format_value(metric['value'], metric['format'])
                html += f'''
                <div class="metric">
                    <div class="metric-label">{metric["label"]}</div>
                    <div class="metric-value">{value}</div>
                </div>
                '''
            html += '</div>'
        
        elif section['type'] == 'table':
            highlight = section.get('highlight', '')
            html += f'<table class="{highlight}">'
            html += '<tr>' + ''.join(f'<th>{h}</th>' for h in section['headers']) + '</tr>'
            for row in section['rows'][:20]:  # Limit to prevent huge PDFs
                html += '<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>'
            html += '</table>'
        
        elif section['type'] == 'summary':
            html += f'<p>{section["content"]}</p>'
        
        elif section['type'] == 'recommendations':
            html += '<ul>'
            for item in section['items']:
                html += f'<li>{item}</li>'
            html += '</ul>'
        
        html += '</div>'
        return html
    
    def _format_value(self, value: float, format_type: str) -> str:
        """Format value based on type."""
        if format_type == 'currency':
            return f"{value:,.2f} TZS"
        elif format_type == 'percentage':
            return f"{value:.1f}%"
        elif format_type == 'number':
            return f"{int(value):,}"
        else:
            return str(value)
    
    def _format_executive_summary(self, data: Dict) -> str:
        """Generate executive summary text."""
        total_sales = data.get('total_sales', 0)
        growth = data.get('growth_percent', 0)
        
        if growth > 0:
            trend = f"up {growth:.1f}% from previous period"
        elif growth < 0:
            trend = f"down {abs(growth):.1f}% from previous period"
        else:
            trend = "flat compared to previous period"
        
        return f"Total sales reached {total_sales:,.2f} TZS, {trend}. " + \
               f"This represents {data.get('transaction_count', 0):,} transactions " + \
               f"with an average order value of {data.get('avg_order_value', 0):,.2f} TZS."
    
    def _format_category_breakdown(self, breakdown: List[Dict]) -> List[List]:
        """Format category breakdown for table."""
        rows = []
        for item in breakdown:
            rows.append([
                item['category'],
                f"{item['sales']:,.2f} TZS",
                str(item['transactions']),
                f"{item['percent']:.1f}%"
            ])
        return rows
    
    def _format_top_products(self, products: List[Dict]) -> List[List]:
        """Format top products for table."""
        rows = []
        for product in products:
            rows.append([
                product['name'],
                str(product['quantity']),
                f"{product['revenue']:,.2f} TZS"
            ])
        return rows
    
    def _format_low_stock_items(self, items: List[Dict]) -> List[List]:
        """Format low stock items for table."""
        rows = []
        for item in items:
            status = '🔴 Critical' if item['current'] == 0 else '⚠️ Low'
            rows.append([
                item['product'],
                str(item['current']),
                str(item['reorder_point']),
                status
            ])
        return rows
    
    def _generate_recommendations(self, data: Dict) -> List[str]:
        """Generate recommendations based on data."""
        recommendations = []
        
        growth = data.get('growth_percent', 0)
        if growth < 0:
            recommendations.append("Sales are declining - consider promotional campaigns to boost revenue")
        elif growth > 20:
            recommendations.append("Strong growth - ensure inventory levels can support increased demand")
        
        if 'low_performers' in data and len(data['low_performers']) > 0:
            recommendations.append("Several products showing poor performance - review pricing and marketing strategy")
        
        recommendations.append("Continue monitoring daily sales trends for early issue detection")
        recommendations.append("Review top-performing products to identify successful strategies for replication")
        
        return recommendations
    
    def _build_templates(self) -> Dict:
        """Build report templates."""
        return {
            'sales': 'sales_performance',
            'inventory': 'inventory_status',
            'customer': 'customer_analysis',
            'financial': 'financial_performance'
        }
