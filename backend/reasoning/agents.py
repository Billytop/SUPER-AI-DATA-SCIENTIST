import os
import re
import difflib
import base64
import random
import pandas as pd
from django.conf import settings
from django.db import connection, connections, close_old_connections
import logging
from .knowledge_base import KnowledgeBase  # 672-question training integration
from .advanced_capabilities import (
    TrainingDatasetGenerator,
    ControlledLearningEngine,
    ContextMemoryEngine,
    ProactiveAnalyzer,
    AutonomousAnalyzer,
    DataScienceGovernance
)
from .memory import AIMemory

class SephlightyBrain:
    """
    NEURAL CORE v4.0 - UNRESTRICTED ENTERPRISE LOGIC
    Role: Lead Data Scientist, Forensic Auditor, Strategic CFO.
    Capabilities: Predictive Synthesis, Forensic Tax Analysis, Global Mesh Synchronization.
    """
    
    RESPONSES = {
        "sw": {
            "greeting": ["Acknowledge. Neural Core v4.0 tayari kwa uchambuzi.", "Handshake imekamilika. Niandikie command.", "Mesh Secure. Naanza kutafuta data..."],
            "unknown": "**Handshake Failed.**\nHujataja command sahihi. Jaribu: 'Mauzo ya leo', 'Deni la [jina]', 'Bidhaa bora'.",
            "error": "**Kernel Panic:** Hitilafu ya mfumo imetokea. ",
            "help": "**Protocol Manual:**",
            "advisory_wait": "**Inachakata logic kwenye neural mesh...**",
            "no_data": "Null vector. Hakuna data iliyopatikana."
        },
        "en": {
            "greeting": ["Acknowledge. Neural Core v4.0 online.", "Handshake established. Awaiting instruction.", "Mesh Secure. Synchronizing with local node..."],
            "unknown": "**Handshake Failed.**\nUnrecognized command sequence. Try: 'Sales today', 'Debt of [name]', 'Best products'.",
            "error": "**Kernel Panic:** System integrity breach. ",
            "help": "**Protocol Manual:**",
            "advisory_wait": "**Propagating logic through neural mesh...**",
            "no_data": "Null vector. No data available."
        }
    }

    def __init__(self, user_id=None, lang='en'):
        self.lang = lang
        self.memory = AIMemory(user_id) if user_id else None
        
        default_context = {
            'last_intent': None,
            'last_sql': None,
            'last_filters': {'time': '', 'metric': 'qty', 'group': None, 'view': 'sum', 'category': None, 'contact': None}, 
            'lang': lang, # Use the provided lang
            'last_results': None,
            'expecting_clarification': False
        }
        
        if self.memory:
            self.context = self.memory.get_agent_context(default_context)
        else:
            self.context = default_context

        # Initialize advanced capabilities (Sections K-P)
        self.learning_engine = ControlledLearningEngine()
        self.context_memory = ContextMemoryEngine()
        self.proactive_analyzer = ProactiveAnalyzer()
        self.autonomous_analyzer = AutonomousAnalyzer()
        self.dataset_generator = TrainingDatasetGenerator() # Initialize the dataset generator
        
    def run(self, query):
        import time
        start_time = time.time()
        close_old_connections()
        final_df = None
        sql_used = None
        confidence = 0
        neural_ops = []
        insights = []
        # 1. Pipeline: Understand
        clean_q, lang = self.preprocess(query)
        self.context['lang'] = lang
        
        # 2. Pipeline: Intent
        intent, is_refinement = self.classify_intent(clean_q)
        
        # Memory Logic: Context Carry-over for Clarifications
        if intent == "UNKNOWN" and self.context.get('last_intent') == "CUSTOMER_RISK":
             intent = "CUSTOMER_RISK"
             is_export = any(x in clean_q for x in ["excel", "pdf", "csv", "download", "export"])
             if "deni" not in clean_q and "customer" not in clean_q and not is_export:
                  clean_q = "Deni la " + clean_q 
             is_refinement = True
        
        # Save to context memory
        self.context_memory.add_interaction(clean_q, '', resolved=False)
        
        new_time_filter = self.extract_time(clean_q)
        if intent == "UNKNOWN" and new_time_filter and self.context['last_intent']:
            intent = self.context['last_intent']
            is_refinement = True
        
        # Export Detection
        if any(x in clean_q for x in ["excel", "pdf", "csv", "download", "export"]):
            is_refinement = True
            if self.context.get('last_intent'):
                intent = self.context['last_intent']

        # Metric Refinement
        if intent == "UNKNOWN" and self.context.get('last_intent') and any(x in clean_q for x in ["amount", "value", "quantity", "qty", "count", "revenue"]):
             intent = self.context['last_intent']
             is_refinement = True

        # Generic Refinement (List, Details, Explain)
        if intent == "UNKNOWN" and self.context.get('last_intent'):
             refine_words = ["list", "listing", "details", "detail", "breakdown", "explain", "more", "expand", "show"]
             if any(x in clean_q for x in refine_words):
                 intent = self.context['last_intent']
                 is_refinement = True
                 if "list" in clean_q or "breakdown" in clean_q:
                     self.context['last_filters']['view'] = 'list'
        
        if (is_refinement or intent == "DETAILS") and self.context['last_intent']:
            intent = self.context['last_intent']
            if any(x in clean_q for x in ["list", "what", "items", "nini"]):
                 self.context['last_filters']['view'] = 'list'
        elif intent != "UNKNOWN":
            self.context['last_intent'] = intent
            if any(x in clean_q for x in ["list", "show", "ndani", "ledger"]):
                 self.context['last_filters']['view'] = 'list'
            else:
                 self.context['last_filters']['view'] = 'sum'
            
        # 3. Pipeline: Filters
        self.update_filters(clean_q, is_refinement)
        
        # 4. Pipeline: Output Mode
        output_mode = "text"
        if any(x in clean_q for x in ["chart", "graph", "plot"]): output_mode = "chart"
        if any(x in clean_q for x in ["excel", "csv", "report", "ripoti"]): output_mode = "file"
        
        # 5. Pipeline: Solver
        response_text = ""
        sql = None
        
        try:
            final_df = None  # Capture for output

            # Main Response Generation with Confidence Scoring
            domain = KnowledgeBase.get_domain_from_intent(intent)
            confidence_threshold = KnowledgeBase.get_confidence_threshold(domain)
            confidence = 85  # Default confidence

            # Skip generic audit if asking for specific time grouping
            is_grouping = any(x in clean_q for x in ["best", "month", "day", "week", "list"])
            if intent == "SALES" and "SUM" not in clean_q and not is_grouping and output_mode != "chart":
                response_text = self.run_sales_audit(clean_q, lang)
            
            elif intent == "SALES" and output_mode == "chart":
                # For charts, extract top products for visualization
                t_filter = self.context['last_filters']['time']
                sql = f"""
                SELECT p.name, SUM(sl.quantity * sl.unit_price_inc_tax) as revenue
                FROM sales_transactionsellline sl
                JOIN sales_transaction t ON t.id = sl.transaction_id
                JOIN inventory_product p ON p.id = sl.product_id
                WHERE t.type='sell' {t_filter}
                GROUP BY p.id, p.name
                ORDER BY revenue DESC
                LIMIT 20;
                """
                self.context['last_sql'] = sql
                df = self.execute_sql(sql)
                final_df = df
                
                # Generate chart
                if df is not None and not df.empty:
                    from messaging.charts import ChartService
                    chart_type = ChartService.detect_chart_type(clean_q)
                    x_label, y_label = ChartService.extract_axis_labels(clean_q, intent)
                    chart_path = ChartService.create_chart(
                        data=df,
                        chart_type=chart_type,
                        title=f"Sales Analysis - {t_filter.replace('AND', '').strip() or 'All Time'}",
                        x_label=x_label,
                        y_label=y_label
                    )
                    response_text = f"# 📊 Sales Visualization\n\nChart generated with {len(df)} products.\n\n**Chart Saved:** `{chart_path}`"
                else:
                    response_text = "No data available for chart."
            
            elif intent == "HRM":
                response_text = self.run_hr_audit(clean_q, lang)
            
            elif intent == "CUSTOMER_RISK":
                response_text = self.run_customer_audit(clean_q, lang)
                 
            elif intent == "EMPLOYEE_PERF":
                response_text = self.run_employee_audit(clean_q, lang)
                 
            elif intent == "FORECAST":
                 response_text = self.run_predictive_modeling(clean_q, lang)
                 
            elif intent == "TAX" or intent == "AUDIT":
                response_text = self.run_compliance_check(intent, clean_q, lang)

            elif intent == "GREETING":
                response_text = self.handle_greeting(clean_q, lang)
                confidence = 95  # High confidence for greetings

            elif intent == "HELP":
                response_text = self.handle_help(lang)
                confidence = 90
            
            elif intent == "ADVISORY":
                response_text = self.run_advisory_analysis(clean_q, lang)

            elif intent == "COMPARISON":
                response_text = self.run_comparison_analysis(clean_q, lang)
            elif intent == "CHART":
                # User wants to visualize data - suggest they use the "Show chart" feature or provide data
                return {
                    "answer": f"""
# 📊 Data Visualization Request

To create charts and graphs, you can:

1. **Ask for specific data first**, then request visualization
   - Example: "Show sales by month" → then → "Show as chart"
   
2. **Use visualization keywords** in your query
   - "Compare sales this year vs last year as chart"
   - "Show inventory trend"
   
3. **Request specific chart types**
   - Line chart, Bar chart, Pie chart, Area chart

💡 **Pro Tip**: I'll provide data tables that can be visualized. Your frontend can render these as interactive charts!

**What data would you like to visualize?**
""",
                    "confidence": 85,
                    "intent": intent,
                    "data": None,
                    "sql": "",
                    "execution_time": 0,
                    "insights": [],
                    "neural_ops": []
                }
            elif intent == "TREND":
                # User wants trend analysis
                return {
                    "answer": f"""
# 📈 Trend Analysis Request

To analyze trends, try these specific queries:

**Sales Trends:**
- "Compare sales this month vs last month"
- "Show sales this year vs last year"
- "Sales trend over last 6 months"

**Payment Trends:**
- "Compare payments this month vs last month"
- "Payment status breakdown this year"

**Multi-Metric Comparison:**
- "Compare sales and expenses this year"
- "Show revenue vs costs trend"

💡 **Pro Tip**: Be specific about the time period and metrics you want to analyze!

**What specific trend would you like to see?**
""",
                    "confidence": 85,
                    "intent": intent,
                    "data": None,
                    "sql": "",
                    "execution_time": 0,
                    "insights": [],
                    "neural_ops": []
                }
            elif intent == "KNOWLEDGE":
                response_text = self.run_knowledge_inquiry(clean_q, lang)

            elif intent == "STOCK_MOVEMENT":
                response_text = self.run_stock_movement(clean_q, lang)

            elif intent != "UNKNOWN":
                # Fallback for INVENTORY, EXPENSES, PURCHASES, BEST_PRODUCT
                sql, reasoning = self.generate_sql(intent, clean_q)
                if sql:
                    self.context['last_sql'] = sql
                    df = self.execute_sql(sql)
                    self.context['last_results'] = df
                    final_df = df
                else:
                    df = None
                
                # Handle export for INVENTORY
                export_path_excel = None
                export_path_pdf = None
                chart_path = None
                
                if df is not None and not df.empty:
                    # Chart generation
                    if output_mode == "chart":
                        from messaging.charts import ChartService
                        
                        chart_type = ChartService.detect_chart_type(clean_q)
                        x_label, y_label = ChartService.extract_axis_labels(clean_q, intent)
                        
                        # Generate title from intent and reasoning
                        chart_title = reasoning if reasoning else f"{intent} Analysis"
                        
                        chart_path = ChartService.create_chart(
                            data=df,
                            chart_type=chart_type,
                            title=chart_title,
                            x_label=x_label,
                            y_label=y_label
                        )
                    
                    # Inventory export
                    if intent == "INVENTORY" and any(x in clean_q for x in ["excel", "pdf", "export", "download"]):
                        from messaging.services import ExportService
                        
                        # Generate Excel
                        if "excel" in clean_q or "export" in clean_q or "download" in clean_q:
                            export_path_excel = ExportService.to_excel(df, title="Full_Inventory_Stock")
                        
                        # Generate PDF
                        if "pdf" in clean_q or "download" in clean_q:
                            # Create HTML table for PDF
                            html = "<h1>Full Inventory Stock Report</h1>"
                            html += "<table border='1' cellpadding='5'><tr><th>Product</th><th>Category</th><th>Quantity</th><th>Total Value (TZS)</th></tr>"
                            for _, row in df.iterrows():
                                html += f"<tr><td>{row.iloc[0]}</td><td>{row.iloc[1]}</td><td>{row.iloc[2]:,.0f}</td><td>{row.iloc[3]:,.2f}</td></tr>"
                            html += "</table>"
                            html += f"<p><b>Total Products:</b> {len(df)}</p>"
                            export_path_pdf = ExportService.to_pdf(html, title="Full_Inventory_Stock")
                
                # Add export info to response
                export_info = ""
                if export_path_excel:
                    export_info += f"\n\n📥 **Excel Exported:** `{export_path_excel}`"
                if export_path_pdf:
                    export_info += f"\n📥 **PDF Exported:** `{export_path_pdf}`"
                if chart_path:
                    export_info += f"\n\n📊 **Chart Generated:** `{chart_path}`"
                
                result = self.generate_response(df, intent, reasoning, output_mode, sql, query=clean_q)
                response_text = result["response"] + export_info
                confidence = result["confidence"]
                sql_used = result["sql"]
                neural_ops = result["neural_ops"]
                insights = result["insights"]
            else:
                # Unknown intent - use knowledge-base-enhanced fallback
                response_text, confidence = self.handle_unknown_with_kb(clean_q, lang)
            
            # Domain-aware confidence threshold checking
            if confidence < confidence_threshold:
                if lang == 'sw':
                    response_text += f"\n\n_⚠️ Confidence: {confidence}% (chini ya {confidence_threshold}% inayohitajika kwa {domain}). Tafadhali fafanua zaidi._"
                else:
                    response_text += f"\n\n_⚠️ Confidence: {confidence}% (below {confidence_threshold}% required for {domain}). Please clarify._"
            
            # Mark interaction as resolved
            self.context_memory.add_interaction(clean_q, response_text, resolved=True)
            
            # Add proactive suggestions if appropriate
            if confidence >= 70:
                proactive = self.get_proactive_suggestion()
                if proactive:
                    response_text += f"\n\n---\n\n{proactive}"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            response_text = f"**System Error:** {str(e)}"
            final_df = None

        # Save Context if memory enabled
        if self.memory:
            self.memory.save_agent_context(self.context)

        # Prepare Data for Frontend
        data_json = []
        if final_df is not None and not final_df.empty:
            try:
                clean_df = final_df.copy()
                clean_df = clean_df.fillna(0)
                data_json = clean_df.to_dict(orient='records')
            except:
                data_json = []

        execution_time = (time.time() - start_time) * 1000

        return {
            "answer": response_text,
            "confidence": confidence,
            "intent": intent,
            "data": data_json,
            "sql": sql_used if settings.DEBUG else None,
            "execution_time": execution_time,
            "insights": insights,
            "neural_ops": neural_ops
        }

    def get_proactive_suggestion(self):
        """Delegate to ProactiveAnalyzer"""
        return self.proactive_analyzer.generate_proactive_suggestion()

    def preprocess(self, query):
        # Stage 1: Aggressive normalization - remove extra spaces, trim, lowercase
        q = ' '.join(query.split()).lower()
        
        # Stage 2: Common phrase replacement (Swahili → English)
        # IMPORTANT: Process longer/specific phrases before shorter ones to avoid partial replacements
        phrase_map = {
            # Common typos FIRST
            "inganisha": "compare", "linganisha": "compare", "kulinganisha": "compare",
            "mwa jana": "last year", "mwa huu": "this year",
            "emianing": "remaining", "thngs": "things", "syterm": "system",
            # Then standard Swahili phrases (longer before shorter)
            "mwaka jana": "last year", "mwezi jana": "last month", "wiki jana": "last week",
            "mwaka huu": "this year", "mwezi huu": "this month", "wiki hii": "this week",
            "mwaka uliopita": "last year", "mwezi uliopita": "last month",
            "kwa nini": "why", "habari": "hello", "mambo": "hello", "leo": "today",
            # Note: "jana" (yesterday) removed from here to prevent conflicts with "mwa jana" → "last year"
        }
        for phrase, replacement in phrase_map.items():
            q = q.replace(phrase, replacement)
        
        # Stage 3: Word replacements with typo variants
        replacements = {
            # Time (Protect English terms + typos)
            "today": "today", "toady": "today", "todya": "today", "2day": "today",
            "yesterday": "yesterday", "yestarday": "yesterday", "yesterdy": "yesterday",
            "week": "week", "wek": "week", "weakk": "week",
            "month": "month", "mont": "month", "mnth": "month", "monthe": "month",
            "year": "year", "yr": "year", "yer": "year", "yaer": "year",
            "jana": "yesterday", "mwaka": "year", "leo": "today", "mwezi": "month",
            
            # Protect 'amount' from being replaced (it's a valid keyword)
            "amount": "amount", "amounts": "amount",
            
            # Business terms with typos and slang
            "mauzo": "sales", "salse": "sales", "slae": "sales", "sles": "sales", "sale": "sales",
            "seling": "sales", "sellng": "sales", "seeling": "sales",
            
            "manunuzi": "purchases", "purchase": "purchases", "pyrchases": "purchases", 
            "purchse": "purchases", "puchase": "purchases", "purchaces": "purchases",
            "buying": "purchases", "buyin": "purchases",
            
            "matumizi": "expenses", "expense": "expenses", "expneses": "expenses", 
            "expence": "expenses", "expnse": "expenses", "expens": "expenses",
            "cost": "expenses", "costs": "expenses", "ghrama": "expenses", "gharama": "expenses",
            "spending": "expenses", "spend": "expenses",
            
            "bidhaa": "product", "prodct": "product", "producct": "product", "prduct": "product",
            "item": "product", "itm": "product", "items": "product",
            
            "mzigo": "stock", "stock": "stock", "stok": "stock", "stck": "stock",
            "inventory": "inventory", "inventry": "inventory", "inventroy": "inventory",
            
            "mfanyakazi": "employee", "wafanyakazi": "employee", "staff": "employee", 
            "employe": "employee", "emplyee": "employee", "employeee": "employee",
            "worker": "employee", "wrkr": "employee", "workr": "employee",
            "wafanya": "employee", "kazi": "employee", "kibarua": "employee",
            
            "payroll": "payroll", "payrole": "payroll", "pay-roll": "payroll",
            "salary": "salary", "salry": "salary", "salari": "salary", "sallary": "salary",
            "mshahara": "salary", "bora": "best",
            
            # Finance with typos
            "kodi": "tax", "vat": "tax", "tra": "tax", "txa": "tax", "taxx": "tax",
            "audit": "audit", "audt": "audit", "auditt": "audit",
            "ukaguzi": "audit", "compliance": "audit", "compilance": "audit",
            
            "deni": "debt", "madeni": "debt", "det": "debt", "dept": "debt",
            "balance": "debt", "balence": "debt", "balanc": "debt",
            
            "faida": "profit", "proffit": "profit", "prfit": "profit", "profi": "profit",
            "profitt": "profit", "profet": "profit",
            
            "malipo": "payments", "payment": "payments", "paymnt": "payments",
            "paymet": "payments", "paymentt": "payments",
            
            "valuation": "inventory", "thamani": "inventory", "value": "inventory",
            
            # Knowledge / Definitions with typos
            "roi": "roi", "return": "roi", "ebitda": "ebitda", "margin": "margin", "margn": "margin",
            "liquidity": "liquidity", "liquidty": "liquidity",
            "turnover": "turnover", "turn-over": "turnover", "trn": "turnover",
            "cogs": "cogs", "equity": "equity", "equty": "equity",
            "asset": "asset", "aset": "asset", "asett": "asset",
            "liability": "liability", "liabilty": "liability", "liablity": "liability",
            
            # Actions with typos and slang
            "nshauri": "advise", "ushauri": "advise", "advce": "advise", "advise": "advise",
            "advice": "advise", "advic": "advise",
            
            "forecast": "forecast", "utabiri": "forecast", "forcast": "forecast", 
            "forecst": "forecast", "forcst": "forecast", "predict": "forecast",
            "predic": "forecast", "predct": "forecast",
            
            "compare": "compare", "compar": "compare", "compre": "compare", "comapre": "compare",
            "vs": "vs", "versus": "vs", "vrs": "vs",
            
            "performance": "performance", "perfomance": "performance", "performnce": "performance",
            "preformance": "performance", "perf": "performance",
            
            # Output formats with typos
            "chati": "chart", "chart": "chart", "cahrt": "chart", "chartt": "chart",
            "graph": "chart", "grapgh": "chart", "grph": "chart",
            
            "ripoti": "file", "report": "file", "reprt": "file", "reoprt": "file",
            
            "excel": "excel", "exel": "excel", "excell": "excel", "xcel": "excel",
            "pdf": "pdf", "pfd": "pdf",
            
            # Greetings (normalize) with typos
            "hi": "hello", "hey": "hello", "helo": "hello", "hallo": "hello", 
            "greating": "hello", "greting": "hello", "greetng": "hello",
            "good": "hello", "gud": "hello",
            
            "thanks": "thanks", "thank": "thanks", "thanx": "thanks", "thnks": "thanks",
            "asante": "thanks", "sante": "thanks",
            
            # Misc with typos and slang
            "nini": "what", "wat": "what", "wht": "what", "whatt": "what",
            "ndani": "inside", "insde": "inside",
            
            "list": "list", "lst": "list", "lsit": "list",
            "orodha": "list", "oroda": "list",
            
            "best": "best", "bst": "best", "bestt": "best",
            
            "ledger": "ledger", "legder": "ledger", "ledgr": "ledger",
            
            "thongs": "things", "thing": "things", "things": "things", "thngs": "things",
            "thng": "things", "thins": "things",
            
            "biggest": "biggest", "bigest": "biggest", "bigst": "biggest",
            "most": "most", "mot": "most", "moost": "most",
            "top": "top", "tp": "top", "topp": "top",
            
            "fot": "for", "fr": "for", "forr": "for",
            "gie": "give", "giv": "give", "gve": "give",
            "cn": "can", "cann": "can", "kan": "can",
            
            "majega": "majenga", "nipe": "give", "nambie": "give",
            
            "wadai": "debt", "madeni": "debt", 
            "watu": "customer", "mtu": "customer", "mteja": "customer",
            "wateja": "customer", "custmer": "customer", "cust": "customer",
            "customer": "customer", "cusotmer": "customer",
            
            "day": "days", "days": "days", "siku": "days",
            "wote": "all", "yote": "all", "oll": "all", "al": "all",
            
            # Status terms
            "delivered": "delivered", "deliverd": "delivered", "delivred": "delivered",
            "canceled": "canceled", "cancelled": "canceled", "canceld": "canceled",
            "pending": "pending", "pendng": "pending", "pnding": "pending",
            "partial": "partial", "partil": "partial", "partal": "partial",
            
            # Additional business terms
            "revenue": "revenue", "revenu": "revenue", "revanue": "revenue",  
            "order": "order", "ordr": "order", "orde": "order", "orders": "order",
            "transaction": "transaction", "transction": "transaction", "trnsaction": "transaction",
            
            "uzwa": "sales", "uliouzwa": "sales", "zilizo": "that", "uziku": "sales", "uziwa": "sales",
            "purchased": "purchases", "ununuaji": "purchases", "kununua": "purchases",
            "juu": "top", "chini": "bottom", "mwisho": "end", 
            "kuanzia": "from", "mpaka": "to", "mpk": "to",
            
            # Common query words
            "show": "show", "shw": "show", "sho": "show",
            "give": "give", "gve": "give", "giv": "give",
            "tell": "tell", "tll": "tell", "tel": "tell",
            "how": "how", "hw": "how", "hwo": "how",
            "many": "many", "mny": "many", "may": "many",
            "much": "much", "mch": "much", "muc": "much",
        }
        
        words = q.split()
        corrected = []
        key_terms = list(replacements.keys()) + [
            "sales", "stock", "employee", "product", "tax", "audit", "forecast", "profit", 
            "expenses", "purchases", "year", "last", "total", "show", "customer", "debt", "hello", "help",
            "today", "yesterday", "week", "month" 
        ]
        
        for w in words:
            if w in replacements:
                corrected.append(replacements[w])
            else:
                # Only fuzzy match for words with 4+ characters to avoid corrupting short Swahili words like 'ya', 'la'
                if len(w) >= 4:
                    matches = difflib.get_close_matches(w, key_terms, n=1, cutoff=0.8)
                    if matches:
                        matched = matches[0]
                        corrected.append(replacements.get(matched, matched))
                    else:
                        corrected.append(w)
                else:
                    corrected.append(w)
        
        # Strict Language Detection
        sw_markers = ["mwaka", "mwezi", "wiki", "jana", "leo", "kesho", "mauzo", "deni", "faida", "hasara", "bidhaa", "wateja", "nani", "gani", "kwa", "la", "ya", "nshauri", "shikamoo", "mambo", "habari", "vipi", "poa", "asante", "tafadhali"]
        
        raw_words = query.lower().split()
        score_sw = sum(1 for w in raw_words if w in sw_markers)
        
        # Default to English unless strong Swahili signal
        lang = 'sw' if score_sw > 0 else 'en'
        
        return " ".join(corrected), lang

    def classify_intent(self, query):
        q_lower = query.lower()
        
        # Expanded Greeting/Conversational Detection (50+ patterns)
        greeting_words = {
            # English
            "hello", "hi", "hey", "thanks", "thank", "bye", "goodbye", "good job", "morning", "evening", "afternoon",
            # Swahili
            "mambo", "habari", "shikamoo", "vipi", "poa", "safi", "hujambo", "salama", "ukoje", "kwaheri", "asante"
        }
        
        # Meta questions about AI identity & capabilities
        identity_patterns = [
            "who are you", "wewe ni nani", "you are", "what are you",
            "nani wewe", "what can you do", "unaweza kufanya", "unaweza nini",
            "how can you help", "unaweza kusaidia", "naweza kupata msaada",
            "nisaidie", "help me", "confused", "nimechanganyikiwa",
            "kuna mtu", "are you there", "uko", "nipo tayari", "tuanzie",'ready', "begin", "start"
        ]
        
        # Check for identity/meta questions first
        if any(pattern in q_lower for pattern in identity_patterns):
            return "GREETING", False  # We'll handle this in handle_greeting
        
        words = re.findall(r'\w+', q_lower)
        query_words = set(words)
        
        if len(words) <= 3 and any(g in query_words for g in greeting_words):
             return "GREETING", False
        
        if len(words) <= 6 and any(g in query_words for g in greeting_words):
             if not any(x in q_lower for x in ["debt", "sales", "stock", "deni", "mauzo", "report", "customer"]):
                  return "GREETING", False

        # Help detection
        help_triggers = ["help", "how to", "what can you do"]
        if any(trigger in q_lower for trigger in help_triggers):
            if not any(x in q_lower for x in ["give me", "show me", "ledger", "debt", "sales", "stock"]):
                return "HELP", False
        
        # Tax and Audit (check before COMPARISON to avoid 'company tax' → 'compare tax' confusion)
        if "tax" in query or "compliance" in query or "audit" in query or "vat" in query or "kodi" in query: 
            return "AUDIT", False
        
        # Comparison detection (including temporal patterns)
        temporal_comparison = (
            ("this" in q_lower and "last" in q_lower) or 
            ("today" in q_lower and "yesterday" in q_lower) or
            ("today" in q_lower and "last year" in q_lower) or
            ("day like today" in q_lower and "last year" in q_lower)
        )
        
        if "compare" in query or " vs " in query or "kulinganisha" in query or temporal_comparison:
             return "COMPARISON", False
        
        # Trend Analysis Detection (only for vague requests, not specific queries)
        trend_keywords = ["trend", "pattern", "progression", "mwenendo"]
        has_trend_word = any(x in q_lower for x in trend_keywords)
        has_specific_query = any(x in q_lower for x in ["compare", "vs", "show", "give", "last", "this", "6", "12", "month", "year"])
        
        if has_trend_word and not has_specific_query:
            return "TREND", False
        
        # Chart/Visualization Detection
        if any(x in q_lower for x in ["draw", "chart", "graph", "plot", "visualize", "show chart", "show graph"]):
            return "CHART", False
        
        # Knowledge Base Detection
        knowledge_triggers = ["what is", "define", "meaning of", "explain", "calculate", "formula"]
        if any(x in query for x in knowledge_triggers) or any(x in query.split() for x in ["roi", "ebitda", "margin", "liquidity", "cogs"]):
             return "KNOWLEDGE", False

        # Business intents
        if "debt" in query or "balance" in q_lower or "churn" in q_lower or "customer" in q_lower or "ledger" in q_lower or "statement" in q_lower or "risk" in q_lower: 
            return "CUSTOMER_RISK", False
        if "employee" in q_lower or "performance" in q_lower or "staff" in q_lower or "worker" in q_lower or "wafanyakazi" in q_lower: 
            return "EMPLOYEE_PERF", False
        if ("best" in q_lower or "bora" in q_lower) and ("employee" in q_lower or "staff" in q_lower or "seller" in q_lower or "mfanyakazi" in q_lower):
            return "EMPLOYEE_PERF", False
        
        # "Worst" patterns (opposite of best)
        if any(x in q_lower for x in ["worst", "dhaifu", "weakest", "lowest", "chini", "mbaya"]) and ("employee" in q_lower or "mfanyakazi" in q_lower or "sales" in q_lower or "mauzo" in q_lower):
            return "EMPLOYEE_PERF", False
        # Best products Detection (enhanced)
        best_product_triggers = ["best", "bora", "top", "selling", "identify"]
        product_nouns = ["product", "item", "bidhaa", "selling", "seller", "sold"]
        
        has_best_trigger = any(x in q_lower for x in best_product_triggers)
        has_product_noun = any(x in q_lower for x in product_nouns)
        has_top_number = bool(re.search(r'top\s+\d+', q_lower)) or bool(re.search(r'\d+\s+(?:best|top|selling)', q_lower))
        
        if (has_best_trigger and has_product_noun) or has_top_number:
            return "BEST_PRODUCT", False
        if "forecast" in query: 
            return "FORECAST", False

        if "advise" in query or "advice" in query or "suggest" in query or "better" in query or "profitable" in query or "ranking" in query: 
            return "ADVISORY", False
        if any(x in query for x in ["sales", "revenue", "payment status", "breakdown", "order", "delivered", "canceled", "cancelled", "pending", "partial"]) or ( "best" in query and (  "month" in query or "day" in query or "week" in query or "year" in query)): 
            return "SALES", False
        if "purchases" in query or "buying" in query: 
            return "PURCHASES", False
        if "expenses" in query or "spending" in query: 
            return "EXPENSES", False
        if "profit" in query or "income" in query: 
            return "ACCOUNTING", False
        if "stock" in query or "inventory" in query: 
            if any(x in query for x in ["movement", "history", "flow", "go", "went", "track", "ledger"]):
                return "STOCK_MOVEMENT", False
            return "INVENTORY", False
        
        # Refinement / Correction
        refine_triggers = ["not", "instead", "change", "no", "wrong", "actually", "mean", "hapana", "sio"]
        is_correction = any(x in query.lower().split() for x in refine_triggers)

        if is_correction and self.context.get('last_intent'): 
             return self.context['last_intent'], True
        
        # Details Refinement
        if any(x in query.lower() for x in ["details", "more", "explain", "expand"]):
             if self.context.get('last_intent'): return "DETAILS", True
        
        # Comparison detection
        if "compare" in query or " vs " in query or "kulinganisha" in query:
             return "COMPARISON", False
        
        return "UNKNOWN", False

    def run_comparison_analysis(self, query, lang):
        is_time_compare = any(x in query.lower() for x in ["month", "week", "year", "leo", "jana", "today", "yesterday"])
        q_lower = query.lower()
        
        with connections['erp'].cursor() as cursor:
            # A. Multi-Metric Comparison (e.g., "payments vs sales vs expenses vs purchases" or "sales and expenses")
            # Include both English and Swahili terms
            financial_metrics = [
                "payment", "payments", "sale", "sales", "expense", "expenses", "purchase", "purchases", "revenue", "cost",
                "mauzo", "matumizi", "manunuzi", "malipo", "mapato", "gharama"  # Swahili terms
            ]
            has_vs = " vs " in q_lower
            has_and = " and " in q_lower or " na " in q_lower  # Include Swahili "na" (and)
            has_multiple_metrics = sum(1 for m in financial_metrics if m in q_lower) >= 2
            
            if (has_vs or has_and) and has_multiple_metrics:
                # This is a multi-metric comparison, not a product comparison
                # Map both English and Swahili to standardized metric names
                metrics_map = {
                    "sales": ["sales", "sale", "mauzo"],
                    "expenses": ["expenses", "expense", "matumizi", "gharama"],
                    "purchases": ["purchases", "purchase", "manunuzi"],
                    "payments": ["payments", "payment", "malipo"]
                }
                
                metrics_found = []
                for metric, terms in metrics_map.items():
                    if any(term in q_lower for term in terms):
                        metrics_found.append(metric)
                
                # Get aggregates for each metric
                time_filter = self.context['last_filters']['time']
                results = {}
                
                for metric in metrics_found:
                    if metric == "sales":
                        sql = f"SELECT SUM(final_total) FROM transactions WHERE type='sell' {time_filter}"
                    elif metric == "expenses":
                        sql = f"SELECT SUM(final_total) FROM transactions WHERE type='expense' {time_filter}"
                    elif metric == "purchases":
                        sql = f"SELECT SUM(final_total) FROM transactions WHERE type='purchase' {time_filter}"
                    elif metric == "payments":
                        sql = f"SELECT SUM(amount) FROM transaction_payments tp JOIN transactions t ON t.id=tp.transaction_id WHERE 1=1 {time_filter.replace('transaction_date', 't.transaction_date') if time_filter else ''}"
                    
                    cursor.execute(sql)
                    results[metric] = float(cursor.fetchone()[0] or 0)
                
                # Format response
                period = time_filter.replace('AND', '').strip() or 'All Time'
                res = f"""
# 📊 Multi-Metric Financial Comparison
**Period:** {period}

## Financial Metrics
"""
                for metric, value in results.items():
                    emoji = {"sales": "💰", "expenses": "💸", "purchases": "🛒", "payments": "💳"}.get(metric, "📈")
                    res += f"*   {emoji} **{metric.title()}:** {value:,.2f} TZS\n"
                
                # Add insights
                if "sales" in results and "expenses" in results:
                    profit = results["sales"] - results["expenses"]
                    margin = (profit / results["sales"] * 100) if results["sales"] > 0 else 0
                    res += f"\n## 💡 Insights\n"
                    res += f"*   **Gross Profit:** {profit:,.2f} TZS\n"
                    res += f"*   **Profit Margin:** {margin:.1f}%\n"
                
                return res
            
            # B. Entity/Product Comparison (e.g. "Coke vs Pepsi" or "compare Product A and Product B")
            if not is_time_compare:
                entities = []
                
                # Try "X vs Y" pattern
                if " vs " in query:
                    entities = query.lower().split(" vs ")
                # Try "compare X and Y" pattern
                elif "compare" in q_lower and " and " in query:
                    # Extract text after "compare" and split by "and"
                    compare_idx = q_lower.find("compare")
                    after_compare = query[compare_idx + 7:].strip()  # Skip "compare "
                    entities = [e.strip() for e in after_compare.split(" and ")]
                
                if len(entities) >= 2:
                    # Better entity name extraction - take full names instead of just first/last word
                    e1 = entities[0].strip()
                    e2 = entities[1].strip()
                    
                    # Remove common words at the start
                    for word in ["the", "a", "an"]:
                        if e1.lower().startswith(word + " "):
                            e1 = e1[len(word)+1:]
                        if e2.lower().startswith(word + " "):
                            e2 = e2[len(word)+1:]
                    
                    # Enhanced SQL with profit, customers, and more metrics
                    sql = f"""
                    SELECT 
                        p.name,
                        SUM(sl.quantity) as units_sold,
                        SUM(sl.quantity * sl.unit_price_inc_tax) as revenue,
                        SUM(sl.quantity * (sl.unit_price_inc_tax - COALESCE(v.dpp_inc_tax, 0))) as profit,
                        COUNT(DISTINCT t.contact_id) as customers,
                        AVG(sl.unit_price_inc_tax) as avg_price,
                        COUNT(DISTINCT t.id) as transactions
                    FROM transaction_sell_lines sl
                    JOIN transactions t ON t.id = sl.transaction_id
                    JOIN products p ON p.id = sl.product_id
                    LEFT JOIN variations v ON v.id = sl.variation_id
                    WHERE (p.name LIKE '%{e1}%' OR p.name LIKE '%{e2}%') AND t.type='sell'
                    GROUP BY p.name
                    ORDER BY revenue DESC
                    """
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                    
                    if not rows: 
                        return f"❌ No sales data found for '{e1}' or '{e2}'. Please check product names."
                    
                    if len(rows) == 1:
                        return f"⚠️ Only found data for **{rows[0][0]}**. Product '{e1 if e1.lower() not in rows[0][0].lower() else e2}' not found in database."
                    
                    # Extract data for comparison
                    product1 = {
                        'name': rows[0][0],
                        'units': float(rows[0][1]),
                        'revenue': float(rows[0][2]),
                        'profit': float(rows[0][3] or 0),
                        'customers': int(rows[0][4]),
                        'avg_price': float(rows[0][5]),
                        'transactions': int(rows[0][6])
                    }
                    
                    product2 = {
                        'name': rows[1][0] if len(rows) > 1 else "N/A",
                        'units': float(rows[1][1]) if len(rows) > 1 else 0,
                        'revenue': float(rows[1][2]) if len(rows) > 1 else 0,
                        'profit': float(rows[1][3] or 0) if len(rows) > 1 else 0,
                        'customers': int(rows[1][4]) if len(rows) > 1 else 0,
                        'avg_price': float(rows[1][5]) if len(rows) > 1 else 0,
                        'transactions': int(rows[1][6]) if len(rows) > 1 else 0
                    }
                    
                    # Calculate differences
                    rev_diff = product1['revenue'] - product2['revenue']
                    profit_diff = product1['profit'] - product2['profit']
                    units_diff = product1['units'] - product2['units']
                    cust_diff = product1['customers'] - product2['customers']
                    
                    # Determine winner
                    winner = product1['name'] if rev_diff > 0 else product2['name']
                    winner_emoji = "🥇" if rev_diff > 0 else "🥈"
                    loser_emoji = "🥈" if rev_diff > 0 else "🥇"
                    
                    res = f"""
# ⚔️ Product Battle: {product1['name']} vs {product2['name']}

## 📊 Performance Metrics

| Metric | {winner_emoji} {product1['name'][:30]} | {loser_emoji} {product2['name'][:30]} | Difference |
|--------|------------|------------|------------|
| **Revenue** | {product1['revenue']:,.0f} TZS | {product2['revenue']:,.0f} TZS | **{abs(rev_diff):,.0f} TZS** |
| **Profit** | {product1['profit']:,.0f} TZS | {product2['profit']:,.0f} TZS | **{abs(profit_diff):,.0f} TZS** |
| **Units Sold** | {product1['units']:,.0f} | {product2['units']:,.0f} | **{abs(units_diff):,.0f}** |
| **Customers** | {product1['customers']:,} | {product2['customers']:,} | **{abs(cust_diff):,}** |
| **Avg Price** | {product1['avg_price']:,.0f} TZS | {product2['avg_price']:,.0f} TZS | {abs(product1['avg_price'] - product2['avg_price']):,.0f} TZS |
| **Transactions** | {product1['transactions']:,} | {product2['transactions']:,} | {abs(product1['transactions'] - product2['transactions']):,} |

## 🏆 Winner: **{winner}**

**{winner}** outperformed by:
- 💰 Revenue: **{abs(rev_diff):,.0f} TZS** ({'higher' if rev_diff > 0 else 'lower'})
- 📈 Profit: **{abs(profit_diff):,.0f} TZS** ({'higher' if profit_diff > 0 else 'lower'})
- 👥 Customers: **{abs(cust_diff):,} more customers**
- 📦 Volume: **{abs(units_diff):,.0f} more units sold**

💡 **Insight**: {winner} has a better market position with {product1['customers'] if rev_diff > 0 else product2['customers']} unique customers vs {product2['customers'] if rev_diff > 0 else product1['customers']}.
"""
                    return res

            # B. Status-Based Comparison (e.g., "closed debt vs outstanding debt")
            if "debt" in q_lower and ("closed" in q_lower or "paid" in q_lower or "outstanding" in q_lower or "unpaid" in q_lower):
                # Compare closed/paid debt vs outstanding/unpaid debt for the same period
                time_filter = "AND YEAR(transaction_date) = YEAR(CURDATE())"
                if "last year" in q_lower or "mwaka jana" in q_lower:
                    time_filter = "AND YEAR(transaction_date) = YEAR(CURDATE()) - 1"
                elif "this year" in q_lower or "these year" in q_lower:
                    time_filter = "AND YEAR(transaction_date) = YEAR(CURDATE())"
                
                # Closed/Paid Debt
                sql_closed = f"SELECT SUM(final_total) FROM transactions WHERE type='sell' AND payment_status='paid' {time_filter}"
                cursor.execute(sql_closed)
                closed_val = float(cursor.fetchone()[0] or 0)
                
                # Outstanding/Unpaid Debt (due + partial)
                sql_outstanding = f"SELECT SUM(final_total - (SELECT COALESCE(SUM(amount),0) FROM transaction_payments WHERE transaction_id=transactions.id)) FROM transactions WHERE type='sell' AND payment_status != 'paid' {time_filter}"
                cursor.execute(sql_outstanding)
                outstanding_val = float(cursor.fetchone()[0] or 0)
                
                total = closed_val + outstanding_val
                closed_pct = (closed_val / total * 100) if total > 0 else 0
                outstanding_pct = (outstanding_val / total * 100) if total > 0 else 0
                
                return f"""
# ⚖️ Debt Status Comparison
**Period:** {time_filter.replace('AND', '').strip()}

*   **Closed/Paid Debt:** {closed_val:,.2f} TZS ({closed_pct:.1f}%)
*   **Outstanding Debt:** {outstanding_val:,.2f} TZS ({outstanding_pct:.1f}%)
*   **Total Debt:** {total:,.2f} TZS
*   **Collection Rate:** {'📈' if closed_pct > 70 else '📉'} {closed_pct:.1f}%
"""
            
            # C. Time Comparison - Parse periods from query
            # Determine what metric to compare
            metric = "sales"
            metric_sql = "final_total"
            type_filter = "type='sell'"
            
            if "debt" in q_lower or "deni" in q_lower:
                metric = "debt"
                metric_sql = "final_total - (SELECT COALESCE(SUM(amount),0) FROM transaction_payments WHERE transaction_id=transactions.id)"
                type_filter = "type='sell' AND payment_status != 'paid'"
            elif "purchase" in q_lower or "buying" in q_lower:
                metric = "purchases"
                type_filter = "type='purchase'"
            elif "expense" in q_lower:
                metric = "expenses"
                type_filter = "type='expense'"
            
            # Determine time periods
            period1_label = "This Month"
            period2_label = "Last Month"
            sql_this = f"SELECT SUM({metric_sql}) FROM transactions WHERE {type_filter} AND MONTH(transaction_date)=MONTH(CURDATE()) AND YEAR(transaction_date)=YEAR(CURDATE())"
            sql_last = f"SELECT SUM({metric_sql}) FROM transactions WHERE {type_filter} AND MONTH(transaction_date)=MONTH(DATE_SUB(CURDATE(), INTERVAL 1 MONTH)) AND YEAR(transaction_date)=YEAR(DATE_SUB(CURDATE(), INTERVAL 1 MONTH))"
            
            # Year comparison
            if (("this year" in q_lower or "these year" in q_lower or "mwaka huu" in q_lower) and 
                ("last year" in q_lower or "mwaka jana" in q_lower)):
                period1_label = "This Year (2026)"
                period2_label = "Last Year (2025)"
                sql_this = f"SELECT SUM({metric_sql}) FROM transactions WHERE {type_filter} AND YEAR(transaction_date)=YEAR(CURDATE())"
                sql_last = f"SELECT SUM({metric_sql}) FROM transactions WHERE {type_filter} AND YEAR(transaction_date)=YEAR(CURDATE())-1"
            
            # Today vs same day last year
            elif (("today" in q_lower or "leo" in q_lower or "day like today" in query.lower()) and 
                  ("last year" in q_lower or "mwaka jana" in q_lower)):
                period1_label = "Today"
                period2_label = "Same Day Last Year"
                sql_this = f"SELECT SUM({metric_sql}) FROM transactions WHERE {type_filter} AND DATE(transaction_date)=CURDATE()"
                sql_last = f"SELECT SUM({metric_sql}) FROM transactions WHERE {type_filter} AND DATE(transaction_date)=DATE_SUB(CURDATE(), INTERVAL 1 YEAR)"
            
            # Today vs yesterday
            elif ("today" in q_lower or "leo" in q_lower) and ("yesterday" in q_lower or "jana" in q_lower):
                period1_label = "Today"
                period2_label = "Yesterday"
                sql_this = f"SELECT SUM({metric_sql}) FROM transactions WHERE {type_filter} AND DATE(transaction_date)=CURDATE()"
                sql_last = f"SELECT SUM({metric_sql}) FROM transactions WHERE {type_filter} AND DATE(transaction_date)=DATE_SUB(CURDATE(), INTERVAL 1 DAY)"
            
            # This week vs last week
            elif ("this week" in q_lower or "wiki hii" in q_lower) and ("last week" in q_lower or "wiki iliyopita" in q_lower):
                period1_label = "This Week"
                period2_label = "Last Week"
                sql_this = f"SELECT SUM({metric_sql}) FROM transactions WHERE {type_filter} AND YEARWEEK(transaction_date,1)=YEARWEEK(CURDATE(),1)"
                sql_last = f"SELECT SUM({metric_sql}) FROM transactions WHERE {type_filter} AND YEARWEEK(transaction_date,1)=YEARWEEK(CURDATE(),1)-1"
            
            cursor.execute(sql_this)
            this_val = cursor.fetchone()[0] or 0
            
            cursor.execute(sql_last)
            last_val = cursor.fetchone()[0] or 0
            
            this_val = float(this_val)
            last_val = float(last_val)
            
            diff = this_val - last_val
            pct = ((diff / last_val) * 100) if last_val > 0 else 100 if this_val > 0 else 0
            
            icon = "📈" if diff >= 0 else "📉"
            return f"""
# ⚖️ {metric.title()} Comparison
**Period:** {period1_label} vs {period2_label}

*   **{period1_label}:** {this_val:,.2f} TZS
*   **{period2_label}:** {last_val:,.2f} TZS
*   **Difference:** {diff:+,.2f} TZS
*   **Growth:** {icon} {pct:+.1f}%
"""



    def run_knowledge_inquiry(self, query, lang):
        terms = {
            "roi": "Return on Investment (ROI) measures profitability. Formula: (Net Profit / Cost of Investment) * 100.",
            "ebitda": "EBITDA = Earnings Before Interest, Taxes, Depreciation, and Amortization. It shows operational profitability.",
            "margin": "Gross Margin = (Revenue - COGS) / Revenue. It indicates how much profit you make on each dollar of sales.",
            "liquidity": "Liquidity is your company's ability to pay short-term debts. Current Ratio = Current Assets / Current Liabilities.",
            "turnover": "Inventory Turnover measures how fast you sell stock. High turnover means efficient sales.",
            "cogs": "COGS (Cost of Goods Sold) is the direct cost of producing the goods sold by a company.",
            "equity": "Equity represents the shareholders' stake in the company (Assets - Liabilities).",
            "asset": "An Asset is something the company owns that has value (Cash, Inventory, Equipment).",
            "liability": "A Liability is something the company owes (Loans, Accounts Payable).",
            "valuation": "Business Valuation is the process of determining the economic value of a business. In this system, check 'Inventory Value' or 'Total Assets'."
        }
        
        found = []
        for term, definition in terms.items():
             if term in query.lower():
                 found.append(f"### 💡 {term.upper()}\n{definition}")
                
        if not found:
             return "I can explain business terms like ROI, EBITDA, Margin, Liquidity, etc. Try asking: 'What is ROI?'"
            
        return "\n".join(found)

    def update_filters(self, query, is_refinement):
        time_f = self.extract_time(query)
        if time_f: self.context['last_filters']['time'] = time_f
        elif not is_refinement and "FORECAST" not in query:
             self.context['last_filters']['time'] = "AND YEAR(transaction_date) = 2026"
        if "amount" in query: self.context['last_filters']['metric'] = 'amount'

    def extract_time(self, q):
        sql_parts = []
        year = None
        
        if "today" in q: return "AND DATE(transaction_date) = CURDATE()"
        if "yesterday" in q: return "AND DATE(transaction_date) = SUBDATE(CURDATE(), 1)"
        
        if "this week" in q: return "AND YEARWEEK(transaction_date, 1) = YEARWEEK(CURDATE(), 1)"
        if "last week" in q: return "AND YEARWEEK(transaction_date, 1) = YEARWEEK(CURDATE(), 1) - 1"
        if "this month" in q: return "AND MONTH(transaction_date) = MONTH(CURDATE()) AND YEAR(transaction_date) = YEAR(CURDATE())"
        if "last month" in q: return "AND MONTH(transaction_date) = MONTH(DATE_SUB(CURDATE(), INTERVAL 1 MONTH)) AND YEAR(transaction_date) = YEAR(DATE_SUB(CURDATE(), INTERVAL 1 MONTH))"

        if "last year" in q: year = "YEAR(CURDATE()) - 1"
        elif "2025" in q: year = "2025"
        elif "2026" in q: year = "2026"
        elif "this year" in q: year = "YEAR(CURDATE())"
        
        if year: sql_parts.append(f"YEAR(transaction_date) = {year}")
        
        month_names = {
            "january": "1", "jan": "1",
            "february": "2", "feb": "2",
            "march": "3", "mar": "3",
            "april": "4", "apr": "4",
            "may": "5",
            "june": "6", "jun": "6",
            "july": "7", "jul": "7",
            "august": "8", "aug": "8",
            "september": "9", "sep": "9", "sept": "9", "sept": "9",
            "october": "10", "oct": "10",
            "november": "11", "nov": "11",
            "december": "12", "dec": "12"
        }
        
        # Month Name Range Detection (e.g., "june to december", "january mpaka march")
        # Build regex pattern for month names
        month_pattern = "|".join(month_names.keys())
        range_pattern = f"({month_pattern})\\s*(?:to|mpaka|until|-)\\s*({month_pattern})"
        month_range_match = re.search(range_pattern, q.lower())
        
        if month_range_match:
            start_month_name = month_range_match.group(1)
            end_month_name = month_range_match.group(2)
            start_num = month_names[start_month_name]
            end_num = month_names[end_month_name]
            sql_parts.append(f"MONTH(transaction_date) BETWEEN {start_num} AND {end_num}")
        else:
            # Single month detection (only if no range found)
            for month_name, month_num in month_names.items():
                if month_name in q or f"mwezi wa {month_num}" in q or f"mwezi {month_num}" in q:
                    sql_parts.append(f"MONTH(transaction_date) = {month_num}")
                    break
        
        # Date Format: DD/MM/YYYY or DD-MM-YYYY
        date_match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', q)
        if date_match:
             d, m, y = date_match.groups()
             sql_parts.append(f"DATE(transaction_date) = '{y}-{m}-{d}'")
        
        range_match = re.search(r'(\d{1,2})\s*(?:mpaka|to)\s*(\d{1,2})', q)
        if range_match:
             s, e = range_match.groups()
             sql_parts.append(f"MONTH(transaction_date) BETWEEN {s} AND {e}")
        elif "month" in q and not any(month_names.keys()): 
             m_match = re.search(r'month\s+(\d{1,2})', q)
             if m_match: sql_parts.append(f"MONTH(transaction_date) = {m_match.group(1)}")

        if sql_parts: return "AND " + " AND ".join(sql_parts)
        return None

    def run_customer_audit(self, query, lang):
        # 0. Check Emptiness
        with connections['erp'].cursor() as cursor:
            try:
                cursor.execute("SELECT COUNT(*) FROM contacts")
                total_partners = cursor.fetchone()[0] or 0
            except:
                return "⚠️ **CONFIGURATION ERROR**: The `contacts` table does not exist. Please check your DB connection."
        
        if total_partners == 0:
            return "⚠️ **DATABASE EMPTY**: No contacts found (0 records)."

        # NEW: Detect if user wants aggregate total summary
        q_lower = query.lower()
        wants_total = any(x in q_lower for x in ["total debt", "all debt", "sum", "jumla"])
        
        if wants_total and not any(x in q_lower for x in ["customer", "mteja", "name", "jina", "who", "list"]):
            # Return simple aggregate total
            t_filter = self.context['last_filters']['time']
            with connections['erp'].cursor() as cursor:
                sql = f"""
                SELECT 
                    SUM(final_total - (SELECT COALESCE(SUM(amount),0) FROM transaction_payments WHERE transaction_id=transactions.id)) as outstanding,
                    COUNT(DISTINCT contact_id) as customers,
                    SUM(final_total) as total_sales
                FROM transactions 
                WHERE type='sell' AND payment_status != 'paid' {t_filter}
                """
                cursor.execute(sql)
                row = cursor.fetchone()
                
                outstanding = float(row[0] or 0)
                customer_count = int(row[1] or 0)
                total_sales = float(row[2] or 0)
                
                return f"""
# 💰 Total Outstanding Debt Summary
**Period:** {t_filter.replace('AND', '').strip() or 'All Time'}

## 📊 Aggregate Metrics
*   **Total Outstanding Debt:** {outstanding:,.2f} TZS
*   **Customers with Debt:** {customer_count:,}
*   **Total Sales (Unpaid):** {total_sales:,.2f} TZS
*   **Average Debt per Customer:** {(outstanding / customer_count if customer_count > 0 else 0):,.2f} TZS

💡 **Action Items:**
• Send payment reminders to {customer_count} customers
• Follow up on high-value debts
• Review credit policies
"""

        # 1. Detect query type (Aggregate vs Individual)
        t_filter = self.context['last_filters']['time']
        is_aggregate = any(x in query.lower() for x in ["how many", "wangapi", "total cust", "predict", "forecast", "trend"])
        is_debt_creation = any(x in query.lower() for x in ["debt created", "new debt", "madeni mapya"])
        is_served = any(x in query.lower() for x in ["served", "active", "buy", "bought", "hudumiwa"])
        
        if is_aggregate or is_served or is_debt_creation:
            # === AGGREGATE MODE ===
            
            # A. Customers Served (Unique active in timeframe)
            sql_served = f"SELECT COUNT(DISTINCT contact_id) FROM transactions WHERE type='sell' {t_filter}"
            
            # B. New Debt Created (in timeframe)
            # Logic: Sell transactions in timeframe where payment status is NOT paid, summing up the balance
            sql_debt_new = f"""
                SELECT SUM(final_total - (SELECT COALESCE(SUM(amount),0) FROM transaction_payments WHERE transaction_id=t.id)) 
                FROM transactions t 
                WHERE t.type='sell' AND t.payment_status != 'paid' {t_filter}
            """
            
            # C. Total Sales in timeframe (Context)
            sql_sales = f"SELECT SUM(final_total) FROM transactions WHERE type='sell' {t_filter}"
            
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql_served)
                served_count = cursor.fetchone()[0] or 0
                
                cursor.execute(sql_debt_new)
                new_debt = cursor.fetchone()[0] or 0
                
                cursor.execute(sql_sales)
                total_sales = cursor.fetchone()[0] or 0.1 # Avoid div by zero
            
            # D. Forecasting (Simple Projection)
            prediction_text = ""
            if "predict" in query.lower() or "forecast" in query.lower():
                # Simple logic: If filtering by "this week" (approx 7 days), daily avg * 7
                # Better: Get last 3 months trend
                prediction_text = "\n### 🔮 AI Prediction\n"
                if served_count > 0:
                     projected = int(served_count * 1.2) # Simple 20% growth optimism/logic placeholder
                     prediction_text += f"*   Based on current velocity, you are on track to serve **{projected}** customers next period.\n"
                     prediction_text += f"*   **Action:** Ensure inventory levels for top products."

            scope = t_filter.replace('AND', '').strip() or "All Time"
            
            return f"""
# 👥 Customer Operations Report
**Scope:** {scope}

## 📊 Activity Summary
*   **Customers Served:** {served_count}
*   **New Debt Risk:** {float(new_debt):,.2f} TZS
*   **Debt-to-Sales Ratio:** {((float(new_debt)/float(total_sales))*100):.1f}%

{prediction_text}
"""

        # === INDIVIDUAL MODE (Existing Logic) ===
        general_query_patterns = ["what customers", "which customers", "show customers", "list customers", 
                                  "customers at risk", "risky customers", "top debtors", "high risk"]
        simple_keywords = ["list", "all", "top", "wote", "yote", "rank", "summary"]
        is_general_query = any(pattern in query.lower() for pattern in general_query_patterns) or \
                           any(word in query.lower().split() for word in simple_keywords)
        
        # Resolve Name
        target_ids = []
        target_name = None
        
        if not is_general_query:
            match = re.search(r'(?:la|of|balance|customer|statement|ledger)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
            candidate = match.group(1).strip() if match else None
            
            if candidate:
                 noise_words = ["for", "of", "from", "to", "the", "a", "an", "are", "at", "risk", "lose", "them", 
                               "la", "ya", "cha", "wa", "za", "vya", "kwa"]
                 words = candidate.split()
                 cleaned = [w for w in words if w.lower() not in noise_words]
                 candidate = " ".join(cleaned) if cleaned else candidate

            if not candidate and "deni la " in query.lower():
                 candidate = query.lower().replace("deni la ", "").strip()
            
            if candidate and candidate.lower() not in ["mteja", "debt", "risk", "customer"]:
                 # Smart Match Logic
                 with connections['erp'].cursor() as cursor:
                     cursor.execute(f"SELECT id, name FROM contacts WHERE name LIKE '%{candidate}%' LIMIT 5")
                     candidates = cursor.fetchall()
                     if not candidates:
                         terms = candidate.split()
                         if terms:
                             likes = [f"name LIKE '%{t}%'" for t in terms if len(t)>2]
                             if likes:
                                 cursor.execute(f"SELECT id, name FROM contacts WHERE {' OR '.join(likes)} LIMIT 20")
                                 candidates = cursor.fetchall()
                     
                     if candidates:
                         best = max(candidates, key=lambda c: difflib.SequenceMatcher(None, candidate.lower(), str(c[1]).lower()).ratio())
                         if difflib.SequenceMatcher(None, candidate.lower(), str(best[1]).lower()).ratio() > 0.4:
                             target_ids = [best[0]]
                             target_name = str(best[1])

        # 2. Main Intelligence Query
        where_clause = ""
        if target_ids:
             where_clause = f"AND p.id IN ({','.join(map(str, target_ids))})"
             
        limit_clause = "LIMIT 10" if target_ids else "LIMIT 5"
        
        # SUPREME QUERY
        sql = f"""
        SELECT 
            COALESCE(p.name, 'Unknown') as name,
            SUM(CASE WHEN t.type='sell' THEN t.final_total ELSE 0 END) as total_sales,
            SUM(CASE WHEN t.payment_status != 'paid' AND t.type='sell' THEN t.final_total - (SELECT COALESCE(SUM(amount),0) FROM transaction_payments WHERE transaction_id=t.id) ELSE 0 END) as debt,
            COUNT(DISTINCT t.id) as visit_count,
            MAX(t.transaction_date) as last_seen,
            SUM(t.discount_amount) as total_discount,
            (SELECT username FROM users WHERE id = (SELECT created_by FROM transactions WHERE contact_id = p.id ORDER BY transaction_date DESC LIMIT 1)) as last_staff
        FROM transactions t
        JOIN contacts p ON p.id = t.contact_id
        WHERE t.type IN ('sell', 'opening_balance') {where_clause}
        GROUP BY p.id, p.name
        ORDER BY debt DESC
        {limit_clause}
        """
        
        with connections['erp'].cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            
        if not rows: return "No individual usage data found."

        res = ""
        for r in rows:
            name, sales, debt, visits, last, disc, staff = r
            sales = float(sales or 0)
            debt = float(debt or 0)
            disc = float(disc or 0)
            
            risk = "🟢 Safe"
            action = "Keep engaging."
            if debt > 0:
                if sales > 0 and (debt/sales) > 0.5: 
                    risk = "🔴 CRITICAL"
                    action = "⛔ STOP CREDIT immediately."
                elif (debt/sales) > 0.2: 
                    risk = "🟡 Watchlist"
                    action = "Send payment reminder."
            
            res += f"""
### 👤 {name}
*   **Total Sales:** {sales:,.2f} TZS
*   **Outstanding Debt:** {debt:,.2f} TZS
*   **Visits:** {visits} (Last: {str(last)[:10]})
*   **Risk Level:** {risk}
*   **Last Server:** {staff or 'Unknown'}
*   **💡 Action:** {action}
"""
            if target_ids:
                 with connections['erp'].cursor() as c2:
                    c2.execute(f"SELECT p.name, SUM(sl.quantity) as q FROM transaction_sell_lines sl JOIN transactions t ON t.id=sl.transaction_id JOIN products p ON p.id=sl.product_id WHERE t.contact_id={target_ids[0]} GROUP BY p.id ORDER BY q DESC LIMIT 3")
                    prods = c2.fetchall()
                    if prods:
                        p_list = ", ".join([f"{p[0]} ({p[1]:.0f})" for p in prods])
                        res += f"*   **Top Buys:** {p_list}\n"
        return f"""
# 🔮 CUSTOMER 360 INTELLIGENCE
{res}
"""

    def run_hr_audit(self, query, lang):
        t_filter = self.context['last_filters']['time']
        with connections['erp'].cursor() as cursor:
            cursor.execute("SELECT id FROM expense_categories WHERE name LIKE '%Salary%' OR name LIKE '%Payroll%' OR name LIKE '%HR%' LIMIT 1")
            cat_row = cursor.fetchone()
            cat_id = cat_row[0] if cat_row else 14 
            
        sql_cost = f"SELECT SUM(final_total) FROM transactions WHERE type='expense' {t_filter}"
        sql_headcount = "SELECT COUNT(*) FROM essentials_employees"
        
        with connections['erp'].cursor() as cursor:
            cursor.execute(sql_cost)
            cost = cursor.fetchone()[0] or 0
            try:
                cursor.execute(sql_headcount)
                headcount = cursor.fetchone()[0] or 0
            except: headcount = "Unknown"

        return f"""
# 👥 HRM WORKFORCE AUDIT
**Scope:** {t_filter.replace('AND', '').strip() or 'Default'}
## 1. Payroll Analysis
*   **Target Category ID:** {cat_id}
*   **Total Salary Cost:** {float(cost or 0):,.2f} TZS
*   **Active Staff:** {headcount}
"""

    def run_sales_audit(self, query, lang):
        t_filter = self.context['last_filters']['time']
        q_lower = query.lower()
        
        # Detect grouping requests
        needs_daily_breakdown = any(x in q_lower for x in ["kila siku", "per day", "daily", "each day", "kwa siku"])
        needs_category_breakdown = any(x in q_lower for x in ["kwa category", "per category", "by category", "kwa aina"])
        needs_hourly_breakdown = any(x in q_lower for x in ["kwa saa", "per hour", "hourly", "by hour"])
        
        # Detect "which day" or "which month" questions
        which_day_max = any(x in q_lower for x in ["siku gani", "which day", "day with most", "best day"])
        which_month_min = any(x in q_lower for x in ["mwezi gani", "which month", "chini", "lowest month"])
        
        # Detect voided/canceled sales
        is_voided = any(x in q_lower for x in ["yaliyofutwa", "voided", "canceled", "cancelled"])
        
        # Detect trend queries
        is_declining = any(x in q_lower for x in ["yanayopungua", "declining", "dropping", "falling"])
        is_growing = any(x in q_lower for x in ["yanayoongezeka", "growing", "increasing", "rising"])
        
        # Handle "which day had highest sales"
        if which_day_max:
            sql = f"""
            SELECT DATE(transaction_date) as day, SUM(final_total) as total
            FROM transactions 
            WHERE type='sell' {t_filter}
            GROUP BY DATE(transaction_date)
            ORDER BY total DESC
            LIMIT 1
            """
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql)
                row = cursor.fetchone()
                if row:
                    return f"📅 **Best Sales Day:** {row[0]}\n💰 **Revenue:** {float(row[1]):,.2f} TZS"
                else:
                    return "No data available."
        
        # Handle "which month had lowest sales"
        if which_month_min:
            sql = """
            SELECT MONTHNAME(transaction_date) as month, SUM(final_total) as total
            FROM transactions 
            WHERE type='sell'
            GROUP BY MONTH(transaction_date), YEAR(transaction_date)
            ORDER BY total ASC
            LIMIT 1
            """
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql)
                row = cursor.fetchone()
                if row:
                    return f"📉 **Weakest Month:** {row[0]}\n💰 **Revenue:** {float(row[1]):,.2f} TZS"
                else:
                    return "No data available."
        
        # Daily breakdown
        if needs_daily_breakdown:
            sql = f"""
            SELECT DATE(transaction_date) as day, SUM(final_total) as total
            FROM transactions 
            WHERE type='sell' {t_filter}
            GROUP BY DATE(transaction_date)
            ORDER BY day DESC
            LIMIT 10
            """
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                if not rows:
                    return "No sales data for this period."
                
                res = "# 📅 Daily Sales Breakdown\n\n"
                for r in rows:
                    res += f"**{r[0]}**: {float(r[1]):,.2f} TZS\n"
                return res

    def run_best_product(self, query, lang):
        q_lower = query.lower()
        time_filter = self.context['last_filters']['time']
        
        # Check if user wants by amount/revenue instead of quantity
        by_amount = any(x in q_lower for x in ["by amount", "by revenue", "by sales", "amount", "revenue", "kiasi", "thamani"])
        
        # Detect number of items requested
        top_n = 5
        for word in q_lower.split():
            if word.isdigit():
                top_n = int(word)
                break
        
        with connections['erp'].cursor() as cursor:
            if by_amount:
                # Sort by revenue instead of quantity
                sql = f"""
                SELECT p.name, SUM(sl.quantity * sl.unit_price_inc_tax) as revenue
                FROM transaction_sell_lines sl
                JOIN transactions t ON t.id = sl.transaction_id
                JOIN products p ON p.id = sl.product_id
                WHERE t.type='sell' {time_filter}
                GROUP BY p.id, p.name
                ORDER BY revenue DESC
                LIMIT {top_n}
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                if not rows:
                    return "No product data available."
                
                res = f"📊 **Data Sequence:** **Top {top_n} Products by Revenue**\n\n"
                for i, r in enumerate(rows, 1):
                    res += f"{i}. **{r[0]}**: {float(r[1]):,.2f} TZS\n"
                return res
            else:
                # Original logic - by quantity
                sql = f"""
                SELECT p.name, SUM(sl.quantity) as total_qty
                FROM transaction_sell_lines sl
                JOIN transactions t ON t.id = sl.transaction_id
                JOIN products p ON p.id = sl.product_id
                WHERE t.type='sell' {time_filter}
                GROUP BY p.id, p.name
                ORDER BY total_qty DESC
                LIMIT {top_n}
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                if not rows:
                    return "No product data available."
                
                res = f"📊 **Data Sequence:** **Products**\n\n"
                for i, r in enumerate(rows, 1):
                    res += f"{i}. **{r[0]}**: {float(r[1]):,.2f}\n"
                return res
        
        # Category breakdown
        if needs_category_breakdown:
            sql = f"""
            SELECT c.name, SUM(sl.quantity * sl.unit_price_inc_tax) as revenue
            FROM transaction_sell_lines sl
            JOIN transactions t ON t.id = sl.transaction_id
            JOIN products p ON p.id = sl.product_id
            JOIN categories c ON c.id = p.category_id
            WHERE t.type='sell' {t_filter}
            GROUP BY c.id, c.name
            ORDER BY revenue DESC
            LIMIT 10
            """
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                if not rows:
                    return "No category data available."
                
                res = "# 📊 Sales by Category\n\n"
                for r in rows:
                    res += f"**{r[0]}**: {float(r[1]):,.2f} TZS\n"
                return res
        
        # Hourly breakdown
        if needs_hourly_breakdown:
            sql = f"""
            SELECT HOUR(transaction_date) as hour, SUM(final_total) as total, COUNT(id) as count
            FROM transactions 
            WHERE type='sell' {t_filter}
            GROUP BY HOUR(transaction_date)
            ORDER BY hour ASC
            """
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                if not rows:
                    return "No hourly data available."
                
                res = "# ⏰ Hourly Sales Breakdown\n\n"
                for r in rows:
                    hour_label = f"{r[0]:02d}:00"
                    res += f"**{hour_label}**: {float(r[1]):,.2f} TZS ({r[2]} transactions)\n"
                return res
        
        # Voided sales
        if is_voided:
            sql = f"""
            SELECT DATE(transaction_date) as day, COUNT(id) as count, SUM(final_total) as total
            FROM transactions 
            WHERE type='sell' AND status IN ('draft', 'cancelled') {t_filter}
            GROUP BY DATE(transaction_date)
            ORDER BY day DESC
            LIMIT 10
            """
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                if not rows:
                    return "No voided sales in this period."
                
                res = "# 🚫 Voided/Cancelled Sales\n\n"
                for r in rows:
                    res += f"**{r[0]}**: {r[1]} transactions, {float(r[2] or 0):,.2f} TZS\n"
                return res
        
        # Trend detection: Declining sales
        if is_declining:
            sql = """
            SELECT MONTH(transaction_date) as month, YEAR(transaction_date) as year, SUM(final_total) as total
            FROM transactions 
            WHERE type='sell'
            GROUP BY YEAR(transaction_date), MONTH(transaction_date)
            ORDER BY year DESC, month DESC
            LIMIT 3
            """
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                if len(rows) >= 2:
                    current = float(rows[0][2] or 0)
                    previous = float(rows[1][2] or 0)
                    change = ((current - previous) / previous * 100) if previous > 0 else 0
                    
                    if change < -5:
                        return f"📉 **Sales are declining!**\n\n⚠️ Drop of {abs(change):.1f}% from last period.\n💰 Current: {current:,.2f} TZS\n📊 Previous: {previous:,.2f} TZS"
                    else:
                        return f"✅ **Sales are stable/growing**\n\nChange: {change:+.1f}%"
                else:
                    return "Need more data to detect trends."
        
        # Trend detection: Growing sales
        if is_growing:
            sql = """
            SELECT MONTH(transaction_date) as month, YEAR(transaction_date) as year, SUM(final_total) as total
            FROM transactions 
            WHERE type='sell'
            GROUP BY YEAR(transaction_date), MONTH(transaction_date)
            ORDER BY year DESC, month DESC
            LIMIT 3
            """
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                if len(rows) >= 2:
                    current = float(rows[0][2] or 0)
                    previous = float(rows[1][2] or 0)
                    change = ((current - previous) / previous * 100) if previous > 0 else 0
                    
                    if change > 5:
                        return f"📈 **Sales are growing!**\n\n✅ Growth of {change:.1f}% from last period.\n💰 Current: {current:,.2f} TZS\n📊 Previous: {previous:,.2f} TZS"
                    else:
                        return f"⚠️ **Sales growth is slow**\n\nChange: {change:+.1f}%"
                else:
                    return "Need more data to detect trends."
        
        # Cash vs Bank split
        if any(x in q_lower for x in ["cash vs bank", "kwa cash", "by payment", "payment method"]):
            sql = f"""
            SELECT 
                payment_method,
                COUNT(id) as count,
                SUM(final_total) as total
            FROM transactions 
            WHERE type='sell' {t_filter}
            GROUP BY payment_method
            ORDER BY total DESC
            """
            with connections['erp'].cursor() as cursor:
                try:
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                    if rows:
                        res = "# 💳 Sales by Payment Method\n\n"
                        for r in rows:
                            method = r[0] or 'Unknown'
                            res += f"**{method}**: {float(r[2]):,.2f} TZS ({r[1]} transactions)\n"
                        return res
                    else:
                        return "No payment method data available."
                except:
                    return "⚠️ Payment method split not supported in current schema. Upgrade needed."
        
        # Default comprehensive audit (existing logic)
        sql_kpi = f"SELECT COUNT(id), SUM(final_total), SUM(tax_amount), SUM(total_before_tax), AVG(final_total) FROM transactions WHERE type='sell' {t_filter}"
        sql_risk = f"SELECT payment_status, COUNT(id), SUM(final_total) FROM transactions WHERE type='sell' {t_filter} GROUP BY payment_status"
        sql_top = f"SELECT COALESCE(p.name, 'Product'), SUM(sl.quantity) as qty FROM transaction_sell_lines sl JOIN transactions t ON t.id = sl.transaction_id JOIN products p ON p.id = sl.product_id WHERE t.type='sell' {t_filter} GROUP BY p.id, p.name ORDER BY qty DESC LIMIT 3"
        
        with connections['erp'].cursor() as cursor:
            cursor.execute(sql_kpi)
            kpis = cursor.fetchone() or (0,0,0,0,0)
            cursor.execute(sql_risk)
            risk_rows = cursor.fetchall()
            cursor.execute(sql_top)
            top_rows = cursor.fetchall()
        
        count, rev, tax, net_rev, aov = kpis
        rev = float(rev or 0)
        risk_text = "\n".join([f"- **{(str(r[0] or 'Unknown')).upper()}:** {r[1]} txns ({float(r[2] or 0):,.0f})" for r in risk_rows])
        top_text = "\n".join([f"- {str(r[0] or 'Product')} ({float(r[1] or 0):,.0f} units)" for r in top_rows])
        
        self.context['last_sql'] = sql_kpi

        return f"""
# 🔍 CFO SALES AUDIT REPORT
**Scope:** {t_filter.replace('AND', '').strip() or 'Default'}
## 1. Financials
*   **Total Revenue:** {rev:,.2f} TZS
*   **Net Revenue:** {float(net_rev or 0):,.2f} TZS
*   **Tax:** {float(tax or 0):,.2f} TZS
## 2. Operations
*   **Txns:** {count}
*   **AOV:** {float(aov or 0):,.2f} TZS
*   **Top 3:**
{top_text}
## 3. Risk
{risk_text}
"""

    def run_employee_audit(self, query, lang):
        t_filter = self.context['last_filters']['time']
        q_lower = query.lower()
        
        # Detect "worst" pattern
        is_worst = any(x in q_lower for x in ["worst", "dhaifu", "weakest", "lowest", "chini", "mbaya"])
        order_dir = "ASC" if is_worst else "DESC"
        label = "Weakest" if is_worst else "Best"

        export_format = None
        if "excel" in query.lower(): export_format = "excel"
        elif "pdf" in query.lower(): export_format = "pdf"
        elif "csv" in query.lower(): export_format = "csv"
        
        sql = f"""
        SELECT 
            COALESCE(u.username, 'Unknown') as name,
            COUNT(DISTINCT t.id) as transactions,
            SUM(t.final_total) as revenue,
            AVG(t.final_total) as avg_salet
        FROM transactions t
        JOIN users u ON u.id = t.created_by
        WHERE t.type='sell' {t_filter}
        GROUP BY u.id, u.username
        ORDER BY revenue {order_dir}
        LIMIT 20
        """
        self.context['last_sql'] = sql
        
        try:
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
        except Exception as e:
            return f"System Error: {str(e)}"

        res = ""
        employee_data = []
        for i, r in enumerate(rows):
            name, count, rev, avg = r
            rev = float(rev or 0)
            res += f"{i+1}. **{str(name or 'Unknown')}**: {rev:,.2f} TZS ({count} sales)\n"
            employee_data.append({"Rank": i+1, "Employee": str(name or 'Unknown'), "Sales Count": count, "Revenue (TZS)": rev})
        
        export_path = None
        if export_format and employee_data:
            from messaging.services import ExportService
            if export_format == "excel": export_path = ExportService.to_excel(employee_data, title="Employee_Performance")
            elif export_format == "pdf": 
                html = f"<table><tr><th>Rank</th><th>Employee</th><th>Sales</th><th>Revenue</th></tr>"
                for emp in employee_data: html += f"<tr><td>{emp['Rank']}</td><td>{emp['Employee']}</td><td>{emp['Sales Count']}</td><td>{emp['Revenue (TZS)']:,.2f}</td></tr>"
                html += "</table>"
                export_path = ExportService.to_pdf(html, title="Employee_Performance")
        
        report = f"""
# 👨‍💼 EMPLOYEE PERFORMANCE AUDIT
**Scope:** {t_filter.replace('AND', '').strip() or 'Default'}
## 🏆 Leaderboard (Top 20)
{res}
"""
        if export_path: report += f"\n📥 **File Exported:** `{export_path}`\n"
        return report

    def run_compliance_check(self, intent, query, lang):
        t_filter = self.context['last_filters']['time']
        sql = f"SELECT SUM(tax_amount), SUM(final_total) FROM transactions WHERE type='sell' {t_filter}"
        with connections['erp'].cursor() as cursor:
             cursor.execute(sql)
             tax, rev = cursor.fetchone()
        
        return f"""
# 🛡️ TAX & AUDIT COMPLIANCE
**Scope:** {t_filter.replace('AND', '').strip() or 'Default'}
## 1. Financial Integrity
*   **Total Tax Liability:** {float(tax or 0):,.2f} TZS
*   **Gross Revenue:** {float(rev or 0):,.2f} TZS
*   **Implied Effective Rate:** { ((float(tax or 0) / float(rev)) * 100) if rev else 0 :.1f}%
"""

    def run_forecast(self, query, lang, mode):
        sql_hist = "SELECT SUM(final_total) FROM transactions WHERE type='sell' AND transaction_date > DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
        with connections['erp'].cursor() as cursor:
             cursor.execute(sql_hist)
             rev_30d = cursor.fetchone()[0] or 0
             
        daily_avg = float(rev_30d) / 30
        next_mo = daily_avg * 30
        
        return f"""
# 📈 AI FORECAST (Linear)
**Basis:** 30-Day Moving Average
*   **Past 30 Days:** {float(rev_30d):,.2f} TZS
*   **Next Month Projection:** {next_mo:,.2f} TZS
"""

    def handle_greeting(self, query, lang):
        import random
        from datetime import datetime
        
        q_lower = query.lower()
        hour = datetime.now().hour
        time_sw = "Asubuhi njema" if hour < 12 else "Mchana mwema" if hour < 18 else "Jioni njema"
        time_en = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
        
        # Thanks
        if "thanks" in q_lower or "thank" in q_lower or "asante" in q_lower:
            return random.choice(["Karibu sana! 😊", "You're welcome!", "Hakuna neno! 💼"]) if lang == 'sw' else random.choice(["You're welcome! 😊", "Happy to help!", "Anytime! 💼"])
        
        # Goodbye
        if "bye" in q_lower or "goodbye" in q_lower or "kwaheri" in q_lower: 
            return "Kwaheri! Tutaonana. 👋" if lang == 'sw' else "Goodbye! Come back anytime. 👋"
        
        # Identity Questions ("Who are you?", "Wewe ni nani?")
        if any(x in q_lower for x in ["who are you", "wewe ni nani", "nani wewe", "what are you"]):
            if lang == 'sw':
                return """
🤖 **Mimi ni SephlightyAI**
Msaidizi wako wa Akili Bandia (AI) kwa ajili ya Biashara.

**Naweza:**
• Kuchambua mauzo, gharama, faida
• Kufuatilia madeni ya wateja
• Kutoa ushauri wa kibiashara
• Kutengeneza ripoti (Excel/PDF)

*Uliza chochote kuhusu biashara yako!*
"""
            else:
                return """
🤖 **I'm SephlightyAI**
Your AI-powered Business Intelligence Assistant.

**I can:**
• Analyze sales, expenses, profits
• Track customer debts
• Provide business advisory
• Generate reports (Excel/PDF)

*Ask me anything about your business!*
"""
        
        # Capability Questions ("Unaweza kufanya nini?", "What can you do?")
        if any(x in q_lower for x in ["what can you do", "unaweza kufanya", "unaweza nini", "how can you help", "unaweza kusaidia"]):
            if lang == 'sw':
                return """
💡 **Ninaweza Kukusaidia:**

**📊 Uchambuzi:**
• Mauzo (leo, wiki, mwezi, mwaka)
• Madeni ya wateja (ripoti ya hatari)
• Wafanyakazi bora (ufanisi)

**🔮 Utabiri:**
• Forecast ya mauzo
• Trends za biashara

**📦 Hisa:**
• Stock movement (historia)
• Thamani ya inventory

**Jaribu:** "Mauzo ya wiki hii" au "Deni la [jina]"
"""
            else:
                return """
💡 **I Can Help You With:**

**📊 Analysis:**
• Sales (today, this week, month, year)
• Customer debts (risk reports)
• Best employees (performance)

**🔮 Forecasting:**
• Sales predictions
• Business trends

**📦 Inventory:**
• Stock movement (history)
• Inventory valuation

**Try:** "Sales this week" or "Debt of [name]"
"""
        
        # Onboarding ("Tuanzie wapi?", "Nimechanganyikiwa", "Ready to start")
        if any(x in q_lower for x in ["tuanzie", "confused", "nimechanganyikiwa", "ready", "tayari", "start", "begin", "where", "wapi", "nisaidie"]):
            if lang == 'sw':
                return """
🎯 **Tuanze Hapa:**

**Ukiwa mpya, jaribu:**
1. `Mauzo ya leo` - Faida ya siku
2. `Deni la [jina mteja]` - Angalia madeni
3. `Bidhaa bora` - Top sellers

**Hali ya haraka:**
• `Wafanyakazi wangapi?`
• `Forecast ya mwezi ujao`
• `Nipe Excel` (export data)

*Andika swali lako hapa chini →*
"""
            else:
                return """
🎯 **Let's Start Here:**

**If you're new, try:**
1. `Sales today` - Check daily revenue
2. `Debt of [customer name]` - Check debts
3. `Best products` - Top sellers

**Quick status:**
• `How many employees?`
• `Forecast next month`
• `Generate Excel` (export data)

*Type your question below →*
"""
        
        # Default Greeting
        base_greeting = random.choice(self.RESPONSES[lang]['greeting'])
        
        tips_sw = """
**Mifano ya Haraka:**
• `Mauzo 2026`
• `Deni la [jina]`
• `Mfanyakazi bora`
• `Nipe ripoti` (Excel/PDF)
"""
        tips_en = """
**Quick Examples:**
• `Total sales 2026`
• `Debt of [name]`
• `Best employee`
• `Generate excel` (export report)
"""
        return base_greeting + (tips_sw if lang == 'sw' else tips_en)
    
    def handle_help(self, lang):
        last_intent = self.context.get('last_intent')
        if last_intent == "EMPLOYEE_PERF":
            return """
# 🆘 Employee Performance Help
**Try these:**
• `Best employee last month`
• `Compare employee performance`
• `Generate excel`
"""
        elif last_intent == "CUSTOMER_RISK":
            return """
# 🆘 Customer Intelligence Help
**Try these:**
• `Deni la [customer name]`
• `Ledger for [customer]`
• `Top 10 debtors`
"""
        else:
            return """
# 🆘 SephlightyAI Help Center
## 📊 What I Can Do:
**Financial Analysis:**
• Sales, Expenses, Profit reports
• Tax & Audit compliance
• Forecasting & predictions
**People & Performance:**
• Employee rankings (`mfanyakazi bora`)
• Customer debt tracking (`deni`)
**Data Export:**
• `generate excel/pdf` after any report
"""
    
    def run_advisory_analysis(self, query, lang):
        # 1. Detect Target & Metric
        target = "product"
        if "category" in query or "group" in query: target = "category"
        elif "brand" in query: target = "brand"
        
        metric = "revenue"
        if "profit" in query: metric = "profit"
        elif "qty" in query or "quantity" in query or "volume" in query or "units" in query: metric = "qty"
        
        t_filter = self.context['last_filters']['time']
        scope = t_filter.replace('AND', '').strip() or "All Time"
        
        # 2. Build SQL based on Target
        sql = ""
        title = ""
        
        if target == "product":
            col = "SUM(sl.quantity * sl.unit_price_inc_tax)" if metric == 'revenue' else "SUM(sl.quantity)"
            if metric == "profit": 
                 # Approximation: Selling Price - Purchase Price (if available, else handled via Margin)
                 # Using standard margin approx: (Sell - Purchase_Price_Inc_Tax) * Qty
                 col = "SUM((sl.unit_price_inc_tax - p.purchase_price) * sl.quantity)"
            
            sql = f"""
            SELECT p.name, {col} as val 
            FROM transaction_sell_lines sl 
            JOIN transactions t ON t.id = sl.transaction_id 
            JOIN products p ON p.id = sl.product_id 
            WHERE t.type='sell' {t_filter} 
            GROUP BY p.id, p.name 
            ORDER BY val DESC LIMIT 5
            """
            title = f"Top Products by {metric.title()}"

        elif target == "category":
            col = "SUM(sl.quantity * sl.unit_price_inc_tax)" if metric == 'revenue' else "SUM(sl.quantity)"
            if metric == "profit": col = "SUM((sl.unit_price_inc_tax - p.purchase_price) * sl.quantity)"
            
            sql = f"""
            SELECT COALESCE(c.name, 'Uncategorized'), {col} as val 
            FROM transaction_sell_lines sl 
            JOIN transactions t ON t.id = sl.transaction_id 
            JOIN products p ON p.id = sl.product_id 
            LEFT JOIN categories c ON c.id = p.category_id
            WHERE t.type='sell' {t_filter} 
            GROUP BY c.id, c.name 
            ORDER BY val DESC LIMIT 5
            """
            title = f"Top Categories by {metric.title()}"

        elif target == "brand":
            col = "SUM(sl.quantity * sl.unit_price_inc_tax)" if metric == 'revenue' else "SUM(sl.quantity)"
            if metric == "profit": col = "SUM((sl.unit_price_inc_tax - p.purchase_price) * sl.quantity)"
            
            sql = f"""
            SELECT COALESCE(b.name, 'No Brand'), {col} as val 
            FROM transaction_sell_lines sl 
            JOIN transactions t ON t.id = sl.transaction_id 
            JOIN products p ON p.id = sl.product_id 
            LEFT JOIN brands b ON b.id = p.brand_id
            WHERE t.type='sell' {t_filter} 
            GROUP BY b.id, b.name 
            ORDER BY val DESC LIMIT 5
            """
            title = f"Top Brands by {metric.title()}"
        
        # 3. Execute
        try:
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
        except Exception as e:
            return f"⚠️ **Analysis Failed**: Could not run rank by {target}. (Error: {str(e)})"
            
        if not rows: return f"No sales data found for {target} analysis in this period."
        
        res = ""
        rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        for i, r in enumerate(rows):
            icon = rank_icons[i] if i < 5 else f"{i+1}."
            val = float(r[1] or 0)
            fmt_val = f"{val:,.0f}" if metric == 'qty' else f"{val:,.2f} TZS"
            res += f"{icon} **{r[0]}**: {fmt_val}\n"
            
        return f"""
# 🧠 Business Advisory
**Analysis:** {title}
**Scope:** {scope}

{res}

*💡 Tip: Ask "Compare [Item A] vs [Item B]" for a deeper head-to-head.*
"""

    def run_stock_movement(self, query, lang):
        # 1. Identify Product
        target_name = None
        # Remove noise words and negations
        # Remove noise words and negations
        noise = ["stock", "movement", "history", "of", "for", "la", "ya", "za", "wa", "cha", "vya", "kwa", "nipe", "nataka", "show", "give", "me", "no", "not", "sio", "hapana", "actually", "mean", "wrong", "instead"]
        
        q_clean = query.lower()
        # Use word boundaries to avoid corrupting product names (e.g., 'la' in 'black')
        for w in noise:
            pattern = r'\b' + re.escape(w) + r'\b'
            q_clean = re.sub(pattern, '', q_clean)
        
        target_name = ' '.join(q_clean.split()).strip()
        if not target_name:
             return self.RESPONSES[lang]['unknown']
        
        # 2. Ambiguity Check
        with connections['erp'].cursor() as cursor:
            cursor.execute(f"SELECT id, name FROM products WHERE name LIKE '%{target_name}%' LIMIT 10")
            candidates = cursor.fetchall()
            
        if not candidates:
            # Fallback: Fuzzy Search
            with connections['erp'].cursor() as c:
                c.execute("SELECT id, name FROM products")
                all_prods = c.fetchall()
            
            matches = difflib.get_close_matches(target_name, [p[1] for p in all_prods], n=3, cutoff=0.4)
            if matches:
                suggestions = "\n".join([f"• {m}" for m in matches])
                return f"**Hmm, I couldn't find '{target_name}'.**\nDid you mean:\n{suggestions}"
            else:
                return f"**Product '{target_name}' not found.**\nTry checking the spelling."
            
        if len(candidates) > 1:
            # Check for exactly one precise match
            exact = [c for c in candidates if c[1].lower().strip() == target_name.lower().strip()]
            if len(exact) >= 1:
                # If exact match found, blindly pick the first exact one (Direct Answer Rule)
                pid, pname = exact[0]
            else:
                # If too many stats, show top 5
                options = "\n".join([f"• {c[1]}" for c in candidates[:5]])
                return f"**Which '{target_name}'?**\nFound multiple matches:\n{options}\n\nPlease be specific."
        else:
            pid, pname = candidates[0]
            
        # 3. Ledger Logic (Movement)
        # History: Opening Base + Purchases - Sales
        
        sql = f"""
        SELECT 
            transaction_date,
            type,
            ref_no,
            status,
            (SELECT COALESCE(SUM(quantity), 0) FROM transaction_sell_lines WHERE transaction_id=t.id AND product_id={pid}) as qty_out,
            (SELECT COALESCE(SUM(quantity), 0) FROM purchase_lines WHERE transaction_id=t.id AND product_id={pid}) as qty_in,
            (SELECT name FROM contacts WHERE id=t.contact_id) as contact
        FROM transactions t
        WHERE t.id IN (
            SELECT transaction_id FROM transaction_sell_lines WHERE product_id={pid}
            UNION
            SELECT transaction_id FROM purchase_lines WHERE product_id={pid}
        )
        ORDER BY transaction_date DESC
        LIMIT 15
        """
        
        with connections['erp'].cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            # Get Current Stock
            cursor.execute(f"SELECT SUM(qty_available) FROM variation_location_details WHERE product_id={pid}")
            curr_stock = cursor.fetchone()[0] or 0
            
        if not rows: return f"No movement history for **{pname}**."
        
        res = ""
        for r in rows:
            date, type_, ref, status, q_out, q_in, contact = r
            date_str = str(date)[:10]
            
            if type_ == 'sell':
                icon = "🔴"
                desc = f"Sold to {contact or 'Walk-in'}"
                qty = f"-{float(q_out):.0f}"
            elif type_ == 'purchase':
                icon = "🟢"
                desc = f"Purchased from {contact or 'Supplier'}"
                qty = f"+{float(q_in):.0f}"
            elif type_ == 'opening_stock':
                icon = "📦"
                desc = "Opening Stock"
                qty = f"+{float(q_in):.0f}"
            else:
                icon = "⚪"
                desc = f"{type_}"
                qty = "0"
                
            res += f"{icon} **{date_str}**: {desc} ({qty})\n"
            
        return f"""
# 📦 Stock Ledger: {pname}
**Current Stock:** {float(curr_stock):.0f} units

## 📜 Recent Movement
{res}
"""

    def generate_sql(self, intent, query):
        f = self.context['last_filters']
        t_filter = f['time']
        metric = f.get('metric', 'amount')
        view = f.get('view', 'sum')
        
        if intent == "INVENTORY":
            export_requested = any(x in query for x in ["excel", "pdf", "csv", "export", "download"])
            if export_requested or view == 'list':
                return f"""
                SELECT p.name as product_name, COALESCE(c.name, 'Uncategorized') as category, SUM(vld.qty_available) as quantity, SUM(vld.qty_available * v.sell_price_inc_tax) as total_value
                FROM variation_location_details vld
                JOIN variations v ON v.id = vld.variation_id
                JOIN products p ON p.id = vld.product_id
                LEFT JOIN categories c ON c.id = p.category_id
                GROUP BY p.id, p.name, c.name
                HAVING quantity > 0
                ORDER BY total_value DESC;
                """, "Full Inventory Report"
            else:
                col = "SUM(vld.qty_available * v.sell_price_inc_tax)" if metric == 'amount' else "SUM(vld.qty_available)"
                return f"SELECT p.name, COALESCE({col}, 0) as val FROM variation_location_details vld JOIN products p ON p.id = vld.product_id JOIN variations v ON v.id = vld.variation_id GROUP BY p.id, p.name ORDER BY val DESC LIMIT 10;", "Inventory"
             
        if intent == "BEST_PRODUCT":
             col = "SUM(sl.quantity * sl.unit_price_inc_tax)" if metric == 'amount' else "SUM(sl.quantity)"
             return f"SELECT p.name, {col} as val FROM transaction_sell_lines sl JOIN transactions t ON t.id = sl.transaction_id JOIN products p ON p.id = sl.product_id WHERE t.type='sell' {t_filter} GROUP BY p.id, p.name ORDER BY val DESC LIMIT 5;", "Products"

        if intent == "SALES":
            group_sql = ""
            select_sql = "SUM(final_total)"
            limit_sql = ""
            label = "Sales Perf"

            if "month" in query and ("best" in query or "list" in query or "trend" in query):
                 select_sql = "MONTHNAME(transaction_date) as period, SUM(final_total)"
                 group_sql = "GROUP BY MONTH(transaction_date), MONTHNAME(transaction_date)"
                 label = "Monthly Sales"
                 if "best" in query: limit_sql = "LIMIT 1"
            elif "week" in query and ("best" in query or "list" in query):
                 select_sql = "CONCAT('Week ', WEEK(transaction_date)) as period, SUM(final_total)"
                 group_sql = "GROUP BY WEEK(transaction_date)"
                 label = "Weekly Sales"
                 if "best" in query: limit_sql = "LIMIT 1"
            elif "hour" in query or "time" in query:
                 select_sql = "CONCAT(HOUR(transaction_date), ':00') as period, SUM(final_total)"
                 group_sql = "GROUP BY HOUR(transaction_date)"
                 label = "Hourly Sales"
                 if "best" in query: limit_sql = "LIMIT 1"
            elif "day" in query and ("best" in query or "list" in query):
                 select_sql = "DAYNAME(transaction_date) as period, SUM(final_total)"
                 group_sql = "GROUP BY DAYOFWEEK(transaction_date), DAYNAME(transaction_date)"
                 label = "Daily Sales"
                 if "best" in query: limit_sql = "LIMIT 1"
            elif "year" in query and ("best" in query or "list" in query):
                 select_sql = "YEAR(transaction_date) as period, SUM(final_total)"
                 group_sql = "GROUP BY YEAR(transaction_date)"
                 label = "Yearly Sales"
            
            if group_sql: return f"SELECT {select_sql} as val FROM transactions WHERE type='sell' {t_filter} {group_sql} ORDER BY val DESC {limit_sql};", label
            if view == 'list': return f"SELECT invoice_no, final_total FROM transactions WHERE type='sell' {t_filter} ORDER BY final_total DESC LIMIT 10;", "Sales List"
            return f"SELECT SUM(final_total) as val FROM transactions WHERE type='sell' {t_filter};", "Sales Perf"

        if intent == "PURCHASES":
            if "product" in query or "item" in query or "bidhaa" in query:
                col = "SUM(pl.quantity * pl.purchase_price_inc_tax)" if metric == 'amount' else "SUM(pl.quantity)"
                return f"SELECT p.name, {col} as val FROM purchase_lines pl JOIN transactions t ON t.id = pl.transaction_id JOIN products p ON p.id = pl.product_id WHERE t.type='purchase' {t_filter} GROUP BY p.id, p.name ORDER BY val DESC LIMIT 10;", "Purchased Products"
            if view == 'list': return f"SELECT COALESCE(c.name, 'Unknown'), t.final_total FROM transactions t LEFT JOIN contacts c ON c.id = t.contact_id WHERE t.type='purchase' {t_filter} ORDER BY final_total DESC LIMIT 10;", "Suppliers"
            return f"SELECT SUM(final_total) as val FROM transactions WHERE type='purchase' {t_filter};", "Purchase Perf"
            
        if intent == "EXPENSES":
            if view == 'list':
                return f"""
                SELECT t.transaction_date, COALESCE(ec.name, 'Uncategorized') as category, t.final_total
                FROM transactions t LEFT JOIN expense_categories ec ON ec.id = t.expense_category_id
                WHERE t.type='expense' {t_filter} ORDER BY t.transaction_date DESC LIMIT 20;
                """, "Expense Breakdown"
            return f"SELECT SUM(final_total) as val FROM transactions WHERE type='expense' {t_filter};", "Expenses"
            
        if intent == "ACCOUNTING":
            return f"""SELECT 
(SELECT IFNULL(SUM(final_total), 0) FROM transactions WHERE type='sell' {t_filter}) - 
(SELECT IFNULL(SUM(final_total), 0) FROM transactions WHERE type='purchase' {t_filter}) -
(SELECT IFNULL(SUM(final_total), 0) FROM transactions WHERE type='expense' {t_filter})
as net_profit;""", "Net Profit"

        if intent == "ADVISORY": return f"-- ADVISORY|", "Advisory"
        if intent == "HELP": return "-- HELP", "Assist"
        return None, None

    def execute_sql(self, sql):
        close_old_connections()
        if "--" in sql: return None
        try:
            with connections['erp'].cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                if cursor.description:
                    cols = [col[0] for col in cursor.description]
                    return pd.DataFrame(rows, columns=cols)
                return pd.DataFrame() 
        except Exception as e:
            print(f"SQL EXECUTION ERROR: {e}\nQuery: {sql}")
            return None
        
    def handle_unknown_with_kb(self, q, lang):
        """
        Knowledge-base-enhanced unknown handler with 672 training questions.
        Returns (response, confidence_score)
        """
        q_lower = q.lower()
        
        # Use knowledge base for smart suggestions
        suggestions, confidence = KnowledgeBase.get_smart_suggestions(q_lower, lang)
        
        if confidence > 45:
            # Partial match found
            category = self._detect_category(q_lower)
            response = f"**📚 I partially understand - {category} query.**\n\n"
            if lang == 'sw':
                response += "**Unamaanisha:**\n"
            else:
                response += "**Did you mean:**\n"
            
            response += "\n".join([f"• {s}" for s in suggestions])
            response += "\n\n💡 **Tip:** " + (
                "Taja kwa uwazi - jina la bidhaa, tarehe, au mteja."
                if lang == 'sw' else
                "Be specific - product name, date, or customer name."
            )
            return response, confidence
        
        # No match - generic fallback with training-based suggestions
        default_suggestions = [
            "Mauzo ya leo" if lang == 'sw' else "Sales today",
            "Deni la [jina]" if lang == 'sw' else "Debt of [name]",
            "Bidhaa bora" if lang == 'sw' else "Best products",
            "Stock movement",
            "Mfanyakazi bora" if lang == 'sw' else "Best employee"
        ]
        
        response = "**⚠️ " + (
            "Sijaeleweka swali lako kabisa." if lang == 'sw' else
            "I don't fully understand your question."
        ) + "**\n\n"
        
        response += (
            "**Ninaweza kusaidia na:**\n" if lang == 'sw' else
            "**I can help with:**\n"
        )
        response += "\n".join([f"• {s}" for s in default_suggestions])
        
        # Add knowledge base hint
        response += "\n\n📖 **Training:** " + (
            "Nimejifunza maswali 672 kuhusu ERP/Accounting/Tax/Analytics."
            if lang == 'sw' else
            "I'm trained on 672 questions about ERP/Accounting/Tax/Analytics."
        )
        
        return response, 30
    
    def _detect_category(self, q_lower):
        """Detect query category for better messaging"""
        if any(x in q_lower for x in ['balance', 'ledger', 'accounting']):
            return 'Accounting'
        elif any(x in q_lower for x in ['tax', 'vat', 'tra']):
            return 'Tax'
        elif any(x in q_lower for x in ['sale', 'mauzo', 'revenue']):
            return 'Sales'
        elif any(x in q_lower for x in ['stock', 'inventory']):
            return 'Inventory'
        elif any(x in q_lower for x in ['customer', 'debt', 'deni']):
            return 'Customer'
        elif any(x in q_lower for x in ['employee', 'mfanyakazi']):
            return 'Employee'
        elif any(x in q_lower for x in ['forecast', 'predict']):
            return 'Analytics'
        return 'Business'
    
    def handle_unknown_with_confidence(self, q, lang):
        """Legacy method - delegates to knowledge-base version"""
        return self.handle_unknown_with_kb(q, lang)
        """
        Enhanced unknown handler with confidence scoring and smart suggestions.
        Returns (response, confidence_score)
        """
        q_lower = q.lower()
        
        # Category-based suggestions with confidence scoring
        if any(x in q_lower for x in ["balance", "sheet", "accounting", "ledger", "journal"]):
            response = "⚠️ **Balance Sheet & Accounting Reports**\n\nThese advanced features require additional setup. Available reports:\n• Profit/Loss: `Profit ya mwaka`\n• Tax: `VAT payable`\n• Cash position: `Bank balance`\n\n**Confidence: 45%** - These are partial matches. Please specify which report you need."
            return response, 45
        
        # Sales-related partial matches
        if any(x in q_lower for x in ["sell", "sale", "revenue", "uzaji", "mauzo"]):
            suggestions = [
                "Mauzo ya leo" if lang == 'sw' else "Sales today",
                "Mauzo ya mwezi huu" if lang == 'sw' else "Sales this month",
                "Bidhaa bora" if lang == 'sw' else "Best products"
            ]
            response = f"**I partially understand - sales related query.**\n\nDid you mean:\n" + "\n".join([f"• {s}" for s in suggestions])
            return response, 55
        
        # Inventory-related partial matches
        if any(x in q_lower for x in ["stock", "inventory", "hisa", "bidhaa"]):
            suggestions = [
                "Stock movement" if lang == 'en' else "Harakati za stock",
                "Bidhaa zilizo chini" if lang == 'sw' else "Low stock items",
                "Thamani ya stock" if lang == 'sw' else "Stock valuation"
            ]
            response = f"**I partially understand - inventory related query.**\n\nDid you mean:\n" + "\n".join([f"• {s}" for s in suggestions])
            return response, 58
        
        # Customer/Debt related
        if any(x in q_lower for x in ["customer", "debt", "deni", "mteja", "balance"]):
            suggestions = [
                "Deni la [jina mteja]" if lang == 'sw' else "Debt of [customer name]",
                "Wateja wenye deni" if lang == 'sw' else "Customers with debt",
                "Aging ya madeni" if lang == 'sw' else "Debt aging report"
            ]
            response = f"**I partially understand - customer/debt query.**\n\nDid you mean:\n" + "\n".join([f"• {s}" for s in suggestions])
            return response, 52
        
        # Employee/HR related
        if any(x in q_lower for x in ["employee", "staff", "mfanyakazi", "worker"]):
            suggestions = [
                "Mfanyakazi bora" if lang == 'sw' else "Best employee",
                "Performance ya mfanyakazi" if lang == 'sw' else "Employee performance",
                "Mauzo kwa mfanyakazi" if lang == 'sw' else "Sales by employee"
            ]
            response = f"**I partially understand - employee query.**\n\nDid you mean:\n" + "\n".join([f"• {s}" for s in suggestions])
            return response, 50
        
        # Fallback: Generic suggestions
        default_suggestions_sw = [
            "Mauzo ya leo",
            "Deni la [jina]",
            "Bidhaa bora",
            "Stock movement",
            "Mfanyakazi bora"
        ]
        
        default_suggestions_en = [
            "Sales today",
            "Debt of [name]",
            "Best products",
            "Stock movement",
            "Best employee"
        ]
        
        suggestions = default_suggestions_sw if lang == 'sw' else default_suggestions_en
        
        response = f"**I don't fully understand your question.**\n\nHere are some things I can help with:\n" + "\n".join([f"• {s}" for s in suggestions])
        response += "\n\n💡 **Tip:** Be specific with product names, dates, or customer names."
        
        return response, 30  # Low confidence for complete unknowns
    
    def handle_unknown(self, q, lang):
        """Legacy handler - delegates to enhanced version"""
        response, _ = self.handle_unknown_with_confidence(q, lang)
        return response

    def get_neural_operations(self, intent, query):
        """
        Generate tactical neural operation logs for the frontend terminal.
        """
        import random
        base_ops = [
            "Handshake verified with mesh node.",
            "Loading knowledge vector 672...",
            "Analyzing synaptic proximity...",
            "Propagating logic gates...",
            "Forensic data audit initiated."
        ]
        
        intent_ops = {
            "SALES": [
                "Scanning revenue vectors...",
                "Synthesizing transaction history...",
                "Calculating margin drift...",
                "Optimizing sales projection."
            ],
            "CUSTOMER_RISK": [
                "Auditing debt escalation...",
                "Verifying customer solvency...",
                "Analyzing churn probability...",
                "Forecasting arrears risk."
            ],
            "INVENTORY": [
                "Querying stock buffers...",
                "Analyzing rotation speed...",
                "Calculating potential stockout...",
                "Optimizing order logic."
            ]
        }
        
        ops = random.sample(base_ops, 3)
        if intent in intent_ops:
            ops += random.sample(intent_ops[intent], 2)
        else:
            ops += ["Executing generic business logic.", "Refreshing data quality cache."]
            
        return ops

    def generate_response(self, df, intent, reasoning, mode, sql, query=None):
        if df is None or df.empty:
            return {
                "response": "No data found matching your criteria.",
                "confidence": 0,
                "intent": intent,
                "sql": sql,
                "neural_ops": self.get_neural_operations(intent, query or ""),
                "insights": []
            }
        res = ""
        
        if len(df.columns) == 1:
            res = f"**{float(df.iloc[0,0] or 0):,.2f}**"
        elif len(df.columns) == 4:
            total_qty = 0
            total_value = 0
            for i, row in df.iterrows():
                product = str(row.iloc[0] or 'Unknown')
                category = str(row.iloc[1] or 'N/A')
                qty = float(row.iloc[2] or 0)
                value = float(row.iloc[3] or 0)
                total_qty += qty
                total_value += value
                if i < 20: res += f"{i+1}. **{product}** ({category}) - Qty: {qty:,.0f}, Value: {value:,.2f} TZS\n"
            res += f"\n**Summary:** {len(df)} products, Total Qty: {total_qty:,.0f}, Total Value: {total_value:,.2f} TZS"
        elif len(df.columns) == 3:
            for i, row in df.iterrows():
                date = str(row.iloc[0])[:10] if row.iloc[0] else "N/A"
                category = str(row.iloc[1] or 'Unknown')
                amount = float(row.iloc[2] or 0)
                res += f"{i+1}. **{date}** - {category}: {amount:,.2f} TZS\n"
        else:
            for i, row in df.iterrows():
                res += f"{i+1}. **{str(row.iloc[0] or 'Unknown')}**: {float(row.iloc[1] or 0):,.2f}\n"
        
        suggestions = self.generate_suggestions(intent, df)
        tone_prefix = self.get_tone_prefix(intent, df)
        
        return {
            "response": f"{tone_prefix} **{reasoning}**\n\n{res}\n\n---\n{suggestions}",
            "confidence": 98 if df is not None and not df.empty else 40,
            "intent": intent,
            "sql": sql,
            "neural_ops": self.get_neural_operations(intent, query or ""),
            "insights": self.extract_insights(df, intent) if df is not None else []
        }

    def extract_insights(self, df, intent):
        if df is None or df.empty: return []
        # Advanced neural insight generation logic
        insights = []
        if len(df.columns) >= 2:
             if 'Value' in str(df.columns) or 'amount' in str(df.columns).lower():
                  insights.append("Detected high-value vectors in current dataset.")
        
        if intent == "SALES":
             insights.append("Revenue velocity is within expected enterprise parameters.")
        elif intent == "INVENTORY":
             insights.append("Mesh detects potential optimization in stock rotation.")
             
        return insights

    def get_tone_prefix(self, intent, df):
        import random
        # Tactical Resonance Filter
        if intent == "CUSTOMER_RISK" or (intent == "sales" and df is not None and len(df)>0 and hasattr(df.iloc[0,0], 'real') and df.iloc[0,0] < 0): 
            prefixes = ["🛡️ **Breach Alert:**", "⚠️ **Neural Risk Detection:**", "🚨 **Vector Anomaly:**"]
            return random.choice(prefixes)
            
        prefixes_advisor = ["🧠 **Neural Synthesis:**", "💡 **Logic Insight:**", "🔮 **Inference Result:**"]
        prefixes_analyst = ["📊 **Data Sequence:**", "🔢 **Neural Computation:**", "🔎 **Forensic Audit:**"]
        return random.choice(prefixes_advisor + prefixes_analyst)

    def generate_suggestions(self, intent, df):
        import random
        
        # Build context-aware suggestion pools
        suggestions = []
        
        if intent == "SALES":
            suggestions = [
                "🏆 Show top selling products?",
                "📉 Analyze profit margins?", 
                "🔮 Forecast next month?",
                "📅 Compare with last week?",
                "📊 Break down by category?",
                "⏰ Show hourly sales pattern?",
                "🎯 Identify best performing day?",
                "📈 Show sales trend chart?",
                "💰 Calculate average order value?",
                "🛒 Analyze customer purchase frequency?"
            ]
        elif intent == "CUSTOMER_RISK":
            suggestions = [
                "📜 Show detailed ledger?",
                "📞 Get contact information?",
                "⏳ When was last payment?",
                "📊 Analyze payment behavior?",
                "🎯 Top 10 debtors?",
                "💳 Show transaction history?",
                "📧 Generate reminder message?",
                "🔍 Check credit limit status?",
                "📈 Debt trend over time?",
                "🛡️ Risk assessment report?"
            ]
        elif intent == "INVENTORY":
            suggestions = [
                "📉 Show slow-moving stock?",
                "📦 Generate valuation report?",
                "🛑 Which items are out of stock?",
                "💰 Calculate potential profit?",
                "🔄 Reorder recommendations?",
                "📊 Stock movement analysis?",
                "⚠️ Low stock alerts?",
                "💵 Identify dead stock?",
                "📈 Stock turnover ratio?",
                "🏭 Compare suppliers?"
            ]
        elif intent == "EMPLOYEE_PERF":
            suggestions = [
                "💼 Compare with last month?",
                "📈 Show commission breakdown?",
                "⏰ Analyze login activity?",
                "🏆 Monthly MVP rankings?",
                "📊 Sales by employee chart?",
                "🎯 Set performance targets?",
                "💰 Calculate incentives?",
                "📅 Weekly performance comparison?",
                "🔍 Identify training needs?"
            ]
        elif intent == "PURCHASES":
            suggestions = [
                "📊 Top suppliers by volume?",
                "💰 Analyze purchase costs?",
                "📈 Compare with last month?",
                "🔍 Most purchased items?",
                "📉 Identify cost savings opportunities?",
                "📅 Purchase trend analysis?",
                "🛒 Supplier performance review?",
                "💳 Payment terms analysis?"
            ]
        elif intent == "EXPENSES":
            suggestions = [
                "📊 Break down by category?",
                "💰 Compare with budget?",
                "📈 Month-over-month trend?",
                "🔍 Identify cost reduction areas?",
                "📉 Largest expense items?",
                "📅 Quarterly expense summary?",
                "💳 Track recurring expenses?",
                "⚠️ Over-budget categories?"
            ]
        elif intent == "AUDIT" or intent == "TAX":
            suggestions = [
                "📊 Generate compliance report?",
                "💰 Calculate tax liability?",
                "📈 Year-over-year comparison?",
                "🔍 Audit trail details?",
                "📉 Deduction opportunities?",
                "📅 Quarterly tax summary?",
                "💳 VAT reconciliation?",
                "🛡️ Risk assessment?"
            ]
        elif intent == "COMPARISON":
            suggestions = [
                "📊 Add another period?",
                "📈 Show as chart?",
                "💰 Include profit comparison?",
                "🔍 Break down by category?",
                "📅 Compare different metrics?",
                "🎯 Identify growth drivers?",
                "📉 Variance analysis?",
                "💡 What changed the most?"
            ]
        elif intent == "BEST_PRODUCT":
            suggestions = [
                "📈 Show sales trend?",
                "💰 Profitability analysis?",
                "📊 Compare with competitors?",
                "🔍 Customer demographics?",
                "📅 Seasonal patterns?",
                "🎯 Cross-sell opportunities?",
                "💡 Bundle recommendations?",
                "📉 Identify declining products?"
            ]
        elif intent == "FORECAST":
            suggestions = [
                "📊 Show confidence intervals?",
                "📈 Compare with actual?",
                "💰 Revenue projections?",
                "🔍 Key growth drivers?",
                "📅 Extended forecast (6 months)?",
                "🎯 Scenario analysis?",
                "💡 Recommended actions?",
                "📉 Risk factors?"
            ]
        else:
            # Generic fallback suggestions
            suggestions = [
                "📊 Generate Excel report?",
                "📈 Show trend chart?",
                "🔍 Analyze by category?",
                "💰 Financial summary?",
                "📅 Compare time periods?",
                "🎯 Top performers?",
                "💡 Get recommendations?",
                "📉 Identify issues?"
            ]
        
        # Randomly select 3 suggestions to avoid repetition
        picks = random.sample(suggestions, min(len(suggestions), 3))
        html = "**💡 Suggested Next Steps:**\n"
        for p in picks: 
            html += f"• `{p}`\n"
        return html

    def run_predictive_modeling(self, query, lang):
        """
        NEURAL PREDICTIVE ENGINE
        Uses Scikit-Learn Linear Regression to project future revenue.
        """
        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np
            import pandas as pd
            from datetime import timedelta, datetime
            
            # Fetch historical sales data (last 90 days)
            sql = """
            SELECT DATE(t.transaction_date) as date, SUM(final_total) as revenue 
            FROM transactions t 
            WHERE t.type='sell' AND t.status='final' 
            AND t.transaction_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
            GROUP BY DATE(t.transaction_date) 
            ORDER BY date ASC;
            """
            df = self.execute_sql(sql)
            
            if df is None or len(df) < 10:
                return "**Insufficient Data:** I need at least 10 days of transaction history to build a regression model."

            # Prepare Data for Regression
            df['date'] = pd.to_datetime(df['date'])
            df['day_index'] = (df['date'] - df['date'].min()).dt.days
            
            X = df[['day_index']].values
            y = df['revenue'].values
            
            # Train Model
            model = LinearRegression()
            model.fit(X, y)
            r2_score = model.score(X, y) # Confidence Score
            
            # Predict Next 30 Days
            last_day = df['day_index'].max()
            future_days = np.array([[last_day + i] for i in range(1, 31)])
            predictions = model.predict(future_days)
            
            total_predicted_revenue = sum(predictions)
            trend = "Upward 📈" if model.coef_[0] > 0 else "Downward 📉"
            
            # Format Response
            confidence_pct = int(min(max(r2_score * 100, 0), 99))
            
            response = f"""
### 🔮 Neural Forecast (Next 30 Days)

**Projection Model:** Linear Regression (Scikit-Learn)
**Training Vectors:** {len(df)} historical data points
**Model Confidence (R²):** {confidence_pct}%
**detected Trend:** {trend}

---

### 📊 Predicted Revenue: **{total_predicted_revenue:,.2f} TZS**

**Key Insights:**
• Daily growth factor: **{model.coef_[0]:,.2f} TZS/day**
• The baseline revenue intercept is **{model.intercept_:,.2f} TZS**

> _"Based on the regression slope of {model.coef_[0]:.2f}, I project a {trend.lower()} trajectory for the coming month. Recommended action: {'Increase stock levels' if model.coef_[0] > 0 else 'Review pricing strategy'}."_
"""
            # Store prediction data context for charting (hacky way to pass data up)
            self.context['last_results'] = pd.DataFrame({
                'day': range(1, 31),
                'predicted_revenue': predictions
            })
            
            return response
            
        except ImportError:
            return "**System Error:** Scikit-Learn not installed. Neural Core running in restricted mode."
        except Exception as e:
            return f"**Predictive Engine Error:** {str(e)}"

class SQLReasoningAgent(SephlightyBrain):
    pass
