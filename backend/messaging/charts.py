import os
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ChartService:
    """Generate beautiful charts from data"""
    
    @staticmethod
    def create_chart(data, chart_type="bar", title="Chart", x_label="X", y_label="Y", output_path=None):
        """
        Create a chart from data
        
        Args:
            data: dict or DataFrame with data to plot
            chart_type: 'bar', 'pie', 'line', 'scatter', 'area', 'horizontal_bar'
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            output_path: Where to save (default: Downloads folder)
        
        Returns:
            Path to saved chart image
        """
        try:
            # Set style
            sns.set_style("whitegrid")
            plt.figure(figsize=(12, 7))
            
            # Convert data to plottable format
            if hasattr(data, 'iloc'):  # DataFrame
                if len(data.columns) == 2:
                    labels = data.iloc[:, 0].tolist()
                    values = data.iloc[:, 1].tolist()
                elif len(data.columns) == 1:
                    labels = data.index.tolist()
                    values = data.iloc[:, 0].tolist()
                else:
                    # Use first 2 columns
                    labels = data.iloc[:, 0].tolist()
                    values = data.iloc[:, 1].tolist()
            elif isinstance(data, dict):
                labels = list(data.keys())
                values = list(data.values())
            else:
                raise ValueError("Data must be a DataFrame or dict")
            
            # Limit to top 20 for readability
            if len(labels) > 20:
                labels = labels[:20]
                values = values[:20]
            
            # Generate chart based on type
            if chart_type.lower() in ["bar", "bar_chart", "bars"]:
                plt.bar(range(len(labels)), values, color=sns.color_palette("viridis", len(labels)))
                plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
                plt.ylabel(y_label)
                plt.xlabel(x_label)
                
            elif chart_type.lower() in ["pie", "pie_chart"]:
                plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3", len(labels)))
                plt.axis('equal')
                
            elif chart_type.lower() in ["line", "line_chart", "trend"]:
                plt.plot(range(len(labels)), values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
                plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
                plt.ylabel(y_label)
                plt.xlabel(x_label)
                plt.grid(True, alpha=0.3)
                
            elif chart_type.lower() in ["scatter", "scatter_plot"]:
                plt.scatter(range(len(labels)), values, s=100, alpha=0.6, c=values, cmap='viridis')
                plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
                plt.ylabel(y_label)
                plt.xlabel(x_label)
                
            elif chart_type.lower() in ["area", "area_chart"]:
                plt.fill_between(range(len(labels)), values, alpha=0.5, color='#A23B72')
                plt.plot(range(len(labels)), values, linewidth=2, color='#7A1E54')
                plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
                plt.ylabel(y_label)
                plt.xlabel(x_label)
                
            elif chart_type.lower() in ["horizontal_bar", "barh", "horizontal"]:
                plt.barh(range(len(labels)), values, color=sns.color_palette("rocket", len(labels)))
                plt.yticks(range(len(labels)), labels)
                plt.xlabel(y_label)
                plt.ylabel(x_label)
            
            else:
                # Default to bar
                plt.bar(range(len(labels)), values, color=sns.color_palette("mako", len(labels)))
                plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save to file
            if not output_path:
                downloads = os.path.join(os.path.expanduser("~"), "Downloads")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(downloads, f"Chart_{chart_type}_{timestamp}.png")
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Chart Generation Error: {e}")
            return None
    
    @staticmethod
    def detect_chart_type(query):
        """Detect chart type from user query"""
        q = query.lower()
        
        if any(x in q for x in ["pie", "pie chart", "pie graph"]):
            return "pie"
        elif any(x in q for x in ["line", "line chart", "trend", "time series"]):
            return "line"
        elif any(x in q for x in ["scatter", "scatter plot", "dots"]):
            return "scatter"
        elif any(x in q for x in ["area", "area chart", "filled"]):
            return "area"
        elif any(x in q for x in ["horizontal bar", "barh", "horizontal"]):
            return "horizontal_bar"
        elif any(x in q for x in ["bar", "bar chart", "bar graph"]):
            return "bar"
        else:
            return "bar"  # Default
    
    @staticmethod
    def extract_axis_labels(query, intent):
        """Extract X/Y axis labels from query or generate smart defaults"""
        q = query.lower()
        
        # Check for explicit axis specification
        x_match = None
        y_match = None
        
        # Pattern: "x axis [label]" or "x: [label]"
        import re
        x_pattern = re.search(r'x[\s:]+([a-z\s]+?)(?:y|$|and|\s{2,})', q)
        y_pattern = re.search(r'y[\s:]+([a-z\s]+?)(?:$|and|\s{2,})', q)
        
        if x_pattern:
            x_match = x_pattern.group(1).strip()
        if y_pattern:
            y_match = y_pattern.group(1).strip()
        
        # Smart defaults based on intent
        if intent == "SALES":
            return x_match or "Products / Period", y_match or "Revenue (TZS)"
        elif intent == "INVENTORY":
            return x_match or "Products", y_match or "Stock Value (TZS)"
        elif intent == "EXPENSES":
            return x_match or "Categories", y_match or "Amount (TZS)"
        elif intent == "EMPLOYEE_PERF":
            return x_match or "Employees", y_match or "Revenue (TZS)"
        else:
            return x_match or "Category", y_match or "Value"
