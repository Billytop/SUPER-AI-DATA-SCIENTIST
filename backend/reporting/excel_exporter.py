"""
Excel Export Engine
Exports business data to Excel format with formatting and formulas.
"""

from typing import Dict, List, Optional
from datetime import datetime


class ExcelExporter:
    """
    Exports data to Excel-compatible format.
    Note: Framework for CSV/Excel export. In production would use openpyxl or xlsxwriter.
    """
   
    def __init__(self):
        self.default_formats = self._build_formats()
        
    def export_sales_data(self, data: List[Dict], filename: str = None) -> Dict:
        """
        Export sales data to Excel format.
        
        Args:
            data: List of sales records
            filename: Optional output filename
            
        Returns:
            Dict with export metadata and CSV content
        """
        if not filename:
            filename = f"sales_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        headers = ['Date', 'Invoice No', 'Customer', 'Amount', 'Payment Status', 'Category']
        rows = []
        
        for record in data:
            rows.append([
                record.get('date', ''),
                record.get('invoice_no', ''),
                record.get('customer', ''),
                f"{record.get('amount', 0):.2f}",
                record.get('payment_status', ''),
                record.get('category', '')
            ])
        
        csv_content = self._generate_csv(headers, rows)
        
        return {
            'filename': filename,
            'format': 'csv',
            'rows': len(rows),
            'content': csv_content,
            'size_kb': len(csv_content) / 1024
        }
    
    def export_inventory_data(self, data: List[Dict], filename: str = None) -> Dict:
        """Export inventory data to Excel."""
        if not filename:
            filename = f"inventory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        headers = ['Product Code', 'Product Name', 'Category', 'Stock Qty', 'Unit Price', 'Total Value', 'Reorder Point']
        rows = []
        
        for record in data:
            qty = record.get('quantity', 0)
            price = record.get('price', 0)
            total_value = qty * price
            
            rows.append([
                record.get('code', ''),
                record.get('name', ''),
                record.get('category', ''),
                str(qty),
                f"{price:.2f}",
                f"{total_value:.2f}",
                str(record.get('reorder_point', 0))
            ])
        
        csv_content = self._generate_csv(headers, rows)
        
        return {
            'filename': filename,
            'format': 'csv',
            'rows': len(rows),
            'content': csv_content
        }
    
    def export_customer_data(self, data: List[Dict], include_debt: bool = True, filename: str = None) -> Dict:
        """Export customer data to Excel."""
        if not filename:
            filename = f"customers_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        headers = ['Customer ID', 'Name', 'Email', 'Phone', 'Total Purchases', 'Last Purchase Date']
        if include_debt:
            headers.extend(['Outstanding Debt', 'Credit Limit'])
        
        rows = []
        for record in data:
            row = [
                record.get('id', ''),
                record.get('name', ''),
                record.get('email', ''),
                record.get('phone', ''),
                f"{record.get('total_purchases', 0):.2f}",
                record.get('last_purchase_date', '')
            ]
            
            if include_debt:
                row.extend([
                    f"{record.get('debt', 0):.2f}",
                    f"{record.get('credit_limit', 0):.2f}"
                ])
            
            rows.append(row)
        
        csv_content = self._generate_csv(headers, rows)
        
        return {
            'filename': filename,
            'format': 'csv',
            'rows': len(rows),
            'content': csv_content
        }
    
    def export_pivot_table(self, data: List[Dict], row_field: str, value_field: str, agg_function: str = 'sum') -> Dict:
        """
        Create pivot table export.
        
        Args:
            data: Source data
            row_field: Field to use for rows
            value_field: Field to aggregate
            agg_function: Aggregation function (sum, avg, count)
            
        Returns:
            Pivot table export
        """
        # Group data by row_field
        pivot = {}
        for record in data:
            key = record.get(row_field, 'Unknown')
            value = record.get(value_field, 0)
            
            if key not in pivot:
                pivot[key] = []
            pivot[key].append(value)
        
        # Apply aggregation
        pivot_results = []
        for key, values in pivot.items():
            if agg_function == 'sum':
                result = sum(values)
            elif agg_function == 'avg':
                result = sum(values) / len(values) if values else 0
            elif agg_function == 'count':
                result = len(values)
            else:
                result = sum(values)
            
            pivot_results.append([key, f"{result:.2f}" if agg_function != 'count' else str(int(result))])
        
        # Sort by value descending
        pivot_results.sort(key=lambda x: float(x[1].replace(',', '')), reverse=True)
        
        headers = [row_field.title(), f"{agg_function.title()}({value_field})"]
        csv_content = self._generate_csv(headers, pivot_results)
        
        return {
            'filename': f"pivot_{row_field}_{datetime.now().strftime('%Y%m%d')}.csv",
            'format': 'csv',
            'rows': len(pivot_results),
            'content': csv_content
        }
    
    def export_with_formulas(self, data: List[Dict], formula_columns: Dict[str, str]) -> Dict:
        """
        Export with Excel formulas.
        
        Args:
            data: Data to export
            formula_columns: Dict of column_name -> formula_template
            
        Returns:
            Export with formulas
        """
        # This would generate Excel formulas in production
        # For now, calculate formulas and export values
        headers = list(data[0].keys()) + list(formula_columns.keys())
        rows = []
        
        for i, record in enumerate(data):
            row = list(record.values())
            
            # Add formula columns (calculated, not actual Excel formulas in CSV)
            for formula_name, formula_template in formula_columns.items():
                # Simple formula evaluation (would be Excel formulas in real implementation)
                if formula_template == 'SUM':
                    result = sum(float(v) for v in record.values() if isinstance(v, (int, float)))
                elif formula_template == 'AVG':
                    numeric_values = [float(v) for v in record.values() if isinstance(v, (int, float))]
                    result = sum(numeric_values) / len(numeric_values) if numeric_values else 0
                else:
                    result = 0
                
                row.append(f"{result:.2f}")
            
            rows.append(row)
        
        csv_content = self._generate_csv(headers, rows)
        
        return {
            'filename': f"export_with_formulas_{datetime.now().strftime('%Y%m%d')}.csv",
            'format': 'csv',
            'rows': len(rows),
            'content': csv_content
        }
    
    def create_data_template(self, template_type: str) -> Dict:
        """
        Create importable Excel template.
        
        Args:
            template_type: Type of template (products, customers, employees)
            
        Returns:
            Template structure
        """
        templates = {
            'products': {
                'headers': ['Product Code', 'Name', 'Category', 'Price', 'Stock Quantity', 'Reorder Point'],
                'sample_rows': [
                    ['P001', 'Sample Product 1', 'Electronics', '50000', '100', '20'],
                    ['P002', 'Sample Product 2', 'Furniture', '120000', '50', '10']
                ]
            },
            'customers': {
                'headers': ['Customer ID', 'Name', 'Email', 'Phone', 'Credit Limit'],
                'sample_rows': [
                    ['C001', 'Sample Customer 1', 'customer1@example.com', '+255712345678', '500000'],
                    ['C002', 'Sample Customer 2', 'customer2@example.com', '+255723456789', '1000000']
                ]
            },
            'employees': {
                'headers': ['Employee ID', 'Name', 'Position', 'Department', 'Salary'],
                'sample_rows': [
                    ['E001', 'Sample Employee 1', 'Sales Manager', 'Sales', '800000'],
                    ['E002', 'Sample Employee 2', 'Cashier', 'Operations', '400000']
                ]
            }
        }
        
        template = templates.get(template_type, templates['products'])
        csv_content = self._generate_csv(template['headers'], template['sample_rows'])
        
        return {
            'filename': f"{template_type}_import_template.csv",
            'format': 'csv',
            'content': csv_content,
            'instructions': f"Fill in your {template_type} data following the sample format, then import this file."
        }
    
    def _generate_csv(self, headers: List[str], rows: List[List]) -> str:
        """Generate CSV content from headers and rows."""
        csv_lines = [','.join(f'"{h}"' for h in headers)]
        
        for row in rows:
            csv_lines.append(','.join(f'"{str(cell)}"' for cell in row))
        
        return '\n'.join(csv_lines)
    
    def _build_formats(self) -> Dict:
        """Build default Excel format styles."""
        return {
            'header': {
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': '#FFFFFF'
            },
            'currency': {
                'num_format': '#,##0.00" TZS"'
            },
            'percentage': {
                'num_format': '0.0%'
            },
            'date': {
                'num_format': 'yyyy-mm-dd'
            }
        }


class DashboardBuilder:
    """
    Builds automated dashboard structures.
    """
    
    def __init__(self):
        self.widget_templates = self._build_widget_templates()
        
    def create_sales_dashboard(self, data: Dict) -> Dict:
        """Create sales performance dashboard."""
        dashboard = {
            'title': 'Sales Performance Dashboard',
            'layout': 'grid',
            'widgets': []
        }
        
        # KPI Cards
        dashboard['widgets'].extend([
            {
                'type': 'kpi_card',
                'title': 'Total Sales',
                'value': data.get('total_sales', 0),
                'format': 'currency',
                'trend': data.get('sales_trend', 0),
                'size': 'small'
            },
            {
                'type': 'kpi_card',
                'title': 'Transactions',
                'value': data.get('transaction_count', 0),
                'format': 'number',
                'trend': data.get('transaction_trend', 0),
                'size': 'small'
            },
            {
                'type': 'kpi_card',
                'title': 'Avg Order Value',
                'value': data.get('avg_order', 0),
                'format': 'currency',
                'trend': data.get('avg_order_trend', 0),
                'size': 'small'
            }
        ])
        
        # Charts
        if 'daily_sales' in data:
            dashboard['widgets'].append({
                'type': 'line_chart',
                'title': 'Daily Sales Trend',
                'data': data['daily_sales'],
                'x_axis': 'date',
                'y_axis': 'sales',
                'size': 'large'
            })
        
        if 'category_breakdown' in data:
            dashboard['widgets'].append({
                'type': 'pie_chart',
                'title': 'Sales by Category',
                'data': data['category_breakdown'],
                'size': 'medium'
            })
        
        if 'top_products' in data:
            dashboard['widgets'].append({
                'type': 'bar_chart',
                'title': 'Top 10 Products',
                'data': data['top_products'],
                'size': 'medium'
            })
        
        return dashboard
    
    def create_inventory_dashboard(self, data: Dict) -> Dict:
        """Create inventory management dashboard."""
        dashboard = {
            'title': 'Inventory Dashboard',
            'layout': 'grid',
            'widgets': []
        }
        
        # KPI Cards
        dashboard['widgets'].extend([
            {
                'type': 'kpi_card',
                'title': 'Total Inventory Value',
                'value': data.get('total_value', 0),
                'format': 'currency',
                'size': 'small'
            },
            {
                'type': 'kpi_card',
                'title': 'Low Stock Items',
                'value': data.get('low_stock_count', 0),
                'format': 'number',
                'alert': data.get('low_stock_count', 0) > 0,
                'size': 'small'
            },
            {
                'type': 'kpi_card',
                'title': 'Out of Stock',
                'value': data.get('out_of_stock', 0),
                'format': 'number',
                'alert': data.get('out_of_stock', 0) > 0,
                'size': 'small'
            }
        ])
        
        # Inventory table
        if 'low_stock_items' in data:
            dashboard['widgets'].append({
                'type': 'data_table',
                'title': '⚠️ Low Stock Alert',
                'data': data['low_stock_items'],
                'size': 'large',
                'highlight': 'warning'
            })
        
        return dashboard
    
    def _build_widget_templates(self) -> Dict:
        """Build widget templates."""
        return {
            'kpi_card': {'width': 300, 'height': 150},
            'line_chart': {'width': 800, 'height': 400},
            'bar_chart': {'width': 600, 'height': 400},
            'pie_chart': {'width': 400, 'height': 400},
            'data_table': {'width': 800, 'height': 600}
        }
