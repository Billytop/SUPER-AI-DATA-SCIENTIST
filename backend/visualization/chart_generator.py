"""
Data Visualization and Chart Generation
Creates charts, graphs, and visual analytics for business data.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json


class ChartGenerator:
    """
    Generates chart configurations for various visualization libraries.
    """
    
    def __init__(self):
        self.chart_types = ['line', 'bar', 'pie', 'area', 'scatter', 'donut', 'gauge']
        
    def generate_line_chart(self, data: List[Dict], x_field: str, y_field: str, title: str = 'Line Chart') -> Dict:
        """Generate line chart configuration."""
        x_values = [item[x_field] for item in data]
        y_values = [item[y_field] for item in data]
        
        return {
            'type': 'line',
            'title': title,
            'data': {
                'labels': x_values,
                'datasets': [{
                    'label': y_field,
                    'data': y_values,
                    'borderColor': '#3498db',
                    'backgroundColor': 'rgba(52, 152, 219, 0.1)',
                    'tension': 0.4
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'legend': {'display': True},
                    'title': {'display': True, 'text': title}
                },
                'scales': {
                    'y': {'beginAtZero': True}
                }
            }
        }
    
    def generate_bar_chart(self, data: List[Dict], x_field: str, y_field: str, title: str = 'Bar Chart', horizontal: bool = False) -> Dict:
        """Generate bar chart configuration."""
        x_values = [item[x_field] for item in data]
        y_values = [item[y_field] for item in data]
        
        return {
            'type': 'bar' if not horizontal else 'horizontalBar',
            'title': title,
            'data': {
                'labels': x_values,
                'datasets': [{
                    'label': y_field,
                    'data': y_values,
                    'backgroundColor': '#3498db',
                    'borderColor': '#2980b9',
                    'borderWidth': 1
                }]
            },
            'options': {
                'indexAxis': 'y' if horizontal else 'x',
                'responsive': True,
                'plugins': {
                    'legend': {'display': False},
                    'title': {'display': True, 'text': title}
                }
            }
        }
    
    def generate_pie_chart(self, data: List[Dict], label_field: str, value_field: str, title: str = 'Pie Chart') -> Dict:
        """Generate pie chart configuration."""
        labels = [item[label_field] for item in data]
        values = [item[value_field] for item in data]
        
        # Generate colors
        colors = self._generate_colors(len(data))
        
        return {
            'type': 'pie',
            'title': title,
            'data': {
                'labels': labels,
                'datasets': [{
                    'data': values,
                    'backgroundColor': colors,
                    'borderWidth': 2
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'legend': {'position': 'right'},
                    'title': {'display': True, 'text': title}
                }
            }
        }
    
    def generate_donut_chart(self, data: List[Dict], label_field: str, value_field: str, title: str = 'Donut Chart') -> Dict:
        """Generate donut chart configuration."""
        pie_config = self.generate_pie_chart(data, label_field, value_field, title)
        pie_config['type'] = 'doughnut'
        pie_config['options']['cutout'] = '60%'
        return pie_config
    
    def generate_area_chart(self, data: List[Dict], x_field: str, y_fields: List[str], title: str = 'Area Chart') -> Dict:
        """Generate stacked area chart configuration."""
        x_values = [item[x_field] for item in data]
        
        datasets = []
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, y_field in enumerate(y_fields):
            y_values = [item.get(y_field, 0) for item in data]
            datasets.append({
                'label': y_field,
                'data': y_values,
                'borderColor': colors[i % len(colors)],
                'backgroundColor': self._add_opacity(colors[i % len(colors)], 0.3),
                'fill': True
            })
        
        return {
            'type': 'line',
            'title': title,
            'data': {
                'labels': x_values,
                'datasets': datasets
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'legend': {'display': True},
                    'title': {'display': True, 'text': title}
                },
                'scales': {
                    'y': {'stacked': True, 'beginAtZero': True}
                }
            }
        }
    
    def generate_multi_line_chart(self, data: List[Dict], x_field: str, y_fields: List[str], title: str = 'Multi-Line Chart') -> Dict:
        """Generate multi-line chart."""
        x_values = [item[x_field] for item in data]
        
        datasets = []
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, y_field in enumerate(y_fields):
            y_values = [item.get(y_field, 0) for item in data]
            datasets.append({
                'label': y_field,
                'data': y_values,
                'borderColor': colors[i % len(colors)],
                'backgroundColor': 'transparent',
                'tension': 0.4
            })
        
        return {
            'type': 'line',
            'title': title,
            'data': {
                'labels': x_values,
                'datasets': datasets
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'legend': {'display': True},
                    'title': {'display': True, 'text': title}
                }
            }
        }
    
    def generate_scatter_plot(self, data: List[Dict], x_field: str, y_field: str, title: str = 'Scatter Plot') -> Dict:
        """Generate scatter plot."""
        points = [{'x': item[x_field], 'y': item[y_field]} for item in data]
        
        return {
            'type': 'scatter',
            'title': title,
            'data': {
                'datasets': [{
                    'label': f'{y_field} vs {x_field}',
                    'data': points,
                    'backgroundColor': '#3498db'
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'legend': {'display': False},
                    'title': {'display': True, 'text': title}
                }
            }
        }
    
    def generate_gauge_chart(self, value: float, max_value: float, title: str = 'Gauge', thresholds: Dict = None) -> Dict:
        """Generate gauge/meter chart."""
        percentage = (value / max_value * 100) if max_value > 0 else 0
        
        # Determine color based on thresholds
        if thresholds:
            color = self._get_threshold_color(percentage, thresholds)
        else:
            color = '#3498db'
        
        return {
            'type': 'gauge',
            'title': title,
            'data': {
                'value': value,
                'max': max_value,
                'percentage': percentage,
                'color': color
            },
            'options': {
                'responsive': True,
                'title': {'display': True, 'text': title}
            }
        }
    
    def _generate_colors(self, count: int) -> List[str]:
        """Generate color palette."""
        base_colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e', '#95a5a6', '#16a085'
        ]
        
        # Repeat colors if needed
        colors = []
        for i in range(count):
            colors.append(base_colors[i % len(base_colors)])
        
        return colors
    
    def _add_opacity(self, hex_color: str, opacity: float) -> str:
        """Add opacity to hex color (return rgba string)."""
        # Convert hex to rgb
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        return f'rgba({r}, {g}, {b}, {opacity})'
    
    def _get_threshold_color(self, value: float, thresholds: Dict) -> str:
        """Get color based on threshold rules."""
        if value >= thresholds.get('excellent', 90):
            return '#2ecc71'  # Green
        elif value >= thresholds.get('good', 70):
            return '#f39c12'  # Yellow
        elif value >= thresholds.get('warning', 50):
            return '#e67e22'  # Orange
        else:
            return '#e74c3c'  # Red


class DashboardVisualizer:
    """
    Creates complete dashboard visualizations.
    """
    
    def __init__(self):
        self.chart_gen = ChartGenerator()
        
    def create_sales_dashboard(self, sales_data: Dict) -> Dict:
        """Create sales performance dashboard."""
        dashboard = {
            'title': 'Sales Performance Dashboard',
            'layout': 'grid',
            'widgets': []
        }
        
        # KPI metrics
        dashboard['widgets'].append({
            'type': 'kpi',
            'title': 'Total Sales',
            'value': sales_data.get('total_sales', 0),
            'format': 'currency',
            'trend': sales_data.get('sales_trend', 0),
            'size': 'small'
        })
        
        # Daily sales trend line chart
        if 'daily_sales' in sales_data:
            chart = self.chart_gen.generate_line_chart(
                sales_data['daily_sales'],
                'date',
                'amount',
                'Daily Sales Trend'
            )
            dashboard['widgets'].append({
                'type': 'chart',
                'chart': chart,
                'size': 'large'
            })
        
        # Category breakdown pie chart
        if 'category_sales' in sales_data:
            chart = self.chart_gen.generate_pie_chart(
                sales_data['category_sales'],
                'category',
                'amount',
                'Sales by Category'
            )
            dashboard['widgets'].append({
                'type': 'chart',
                'chart': chart,
                'size': 'medium'
            })
        
        # Top products bar chart
        if 'top_products' in sales_data:
            chart = self.chart_gen.generate_bar_chart(
                sales_data['top_products'][:10],
                'product',
                'sales',
                'Top 10 Products',
                horizontal=True
            )
            dashboard['widgets'].append({
                'type': 'chart',
                'chart': chart,
                'size': 'medium'
            })
        
        return dashboard
    
    def create_inventory_dashboard(self, inventory_data: Dict) -> Dict:
        """Create inventory management dashboard."""
        dashboard = {
            'title': 'Inventory Management Dashboard',
            'layout': 'grid',
            'widgets': []
        }
        
        # KPI: Total value
        dashboard['widgets'].append({
            'type': 'kpi',
            'title': 'Total Inventory Value',
            'value': inventory_data.get('total_value', 0),
            'format': 'currency',
            'size': 'small'
        })
        
        # Inventory by category
        if 'category_inventory' in inventory_data:
            chart = self.chart_gen.generate_donut_chart(
                inventory_data['category_inventory'],
                'category',
                'value',
                'Inventory Value by Category'
            )
            dashboard['widgets'].append({
                'type': 'chart',
                'chart': chart,
                'size': 'medium'
            })
        
        # Stock levels gauge
        if 'stock_health' in inventory_data:
            chart = self.chart_gen.generate_gauge_chart(
                inventory_data['stock_health']['score'],
                100,
                'Overall Stock Health',
                {'excellent': 90, 'good': 70, 'warning': 50}
            )
            dashboard['widgets'].append({
                'type': 'chart',
                'chart': chart,
                'size': 'small'
            })
        
        return dashboard
    
    def create_financial_dashboard(self, financial_data: Dict) -> Dict:
        """Create financial performance dashboard."""
        dashboard = {
            'title': 'Financial Performance Dashboard',
            'layout': 'grid',
            'widgets': []
        }
        
        # Revenue vs Expenses area chart
        if 'monthly_financials' in financial_data:
            chart = self.chart_gen.generate_area_chart(
                financial_data['monthly_financials'],
                'month',
                ['revenue', 'expenses', 'profit'],
                'Revenue, Expenses & Profit Trend'
            )
            dashboard['widgets'].append({
                'type': 'chart',
                'chart': chart,
                'size': 'large'
            })
        
        # Profit margin gauge
        if 'profit_margin' in financial_data:
            chart = self.chart_gen.generate_gauge_chart(
                financial_data['profit_margin'],
                100,
                'Profit Margin %',
                {'excellent': 30, 'good': 20, 'warning': 10}
            )
            dashboard['widgets'].append({
                'type': 'chart',
                'chart': chart,
                'size': 'small'
            })
        
        return dashboard


class DataVisualizationRecommender:
    """
    Recommends appropriate visualizations based on data characteristics.
    """
    
    def __init__(self):
        pass
        
    def recommend_chart_type(self, data_type: str, field_count: int, has_time_series: bool, data_distribution: str = 'normal') -> Dict:
        """
        Recommend best chart type for data.
        
        Args:
            data_type: Type of data (numeric, categorical, mixed)
            field_count: Number of fields to visualize
            has_time_series: Whether data includes time dimension
            data_distribution: Distribution pattern
            
        Returns:
            Chart recommendation
        """
        recommendations = []
        
        # Time series data
        if has_time_series:
            if field_count == 1:
                recommendations.append({
                    'chart_type': 'line',
                    'confidence': 0.9,
                    'reason': 'Line charts are ideal for showing trends over time'
                })
            else:
                recommendations.append({
                    'chart_type': 'multi_line',
                    'confidence': 0.85,
                    'reason': 'Multi-line chart shows multiple trends simultaneously'
                })
                recommendations.append({
                    'chart_type': 'area',
                    'confidence': 0.75,
                    'reason': 'Stacked area chart shows composition over time'
                })
        
        # Categorical data
        elif data_type == 'categorical':
            if field_count <= 10:
                recommendations.append({
                    'chart_type': 'pie',
                    'confidence': 0.85,
                    'reason': 'Pie chart effectively shows part-to-whole relationships'
                })
                recommendations.append({
                    'chart_type': 'bar',
                    'confidence': 0.8,
                    'reason': 'Bar chart allows easy comparison between categories'
                })
            else:
                recommendations.append({
                    'chart_type': 'bar',
                    'confidence': 0.9,
                    'reason': 'Bar chart handles many categories better than pie'
                })
        
        # Numeric correlation
        elif data_type == 'numeric' and field_count == 2:
            recommendations.append({
                'chart_type': 'scatter',
                'confidence': 0.9,
                'reason': 'Scatter plot shows correlation between two numeric variables'
            })
        
        # Single metric
        elif field_count == 1 and data_type == 'numeric':
            recommendations.append({
                'chart_type': 'gauge',
                'confidence': 0.8,
                'reason': 'Gauge effectively displays single KPI value'
            })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'primary_recommendation': recommendations[0] if recommendations else None,
            'alternatives': recommendations[1:3] if len(recommendations) > 1 else []
        }
    
    def suggest_color_scheme(self, purpose: str) -> Dict:
        """Suggest color scheme for visualization."""
        schemes = {
            'financial': {
                'primary': '#2ecc71',  # Green for profit
                'secondary': '#e74c3c',  # Red for loss
                'neutral': '#3498db',
                'description': 'Green/red scheme for financial data'
            },
            'performance': {
                'primary': '#3498db',  # Blue
                'secondary': '#9b59b6',  # Purple
                'neutral': '#95a5a6',
                'description': 'Professional blue/purple scheme'
            },
            'alert': {
                'primary': '#e74c3c',  # Red
                'secondary': '#f39c12',  # Orange
                'neutral': '#e67e22',
                'description': 'Warning colors for alerts'
            },
            'success': {
                'primary': '#2ecc71',  # Green
                'secondary': '#1abc9c',  # Teal
                'neutral': '#16a085',
                'description': 'Positive/success colors'
            }
        }
        
        return schemes.get(purpose, schemes['performance'])
