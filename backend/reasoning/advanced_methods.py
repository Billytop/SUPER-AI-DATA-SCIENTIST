    def run_autonomous_analysis(self, lang='en'):
        \"\"\"
        Section O: Safe autonomous business analysis.
        Triggered by: 'analyze my business', 'check everything'
        \"\"\"
        try:
            # Scan all modules
            modules = ['sales', 'inventory', 'customers', 'employees', 'accounting', 'tax']
            scan_results = self.autonomous_analyzer.full_business_scan(modules)
            
            # Generate user-friendly report
            report = self.autonomous_analyzer.generate_summary_report(scan_results)
            
            return report
        except Exception as e:
            return f\"❌ Analysis error: {str(e)}\"
    
    def export_training_dataset(self, format='json', path=None):
        \"\"\"
        Section K: Export 672-question training dataset.
        Formats: json, csv, parquet, sql
        \"\"\"
        try:
            result = TrainingDatasetGenerator.generate_dataset(format, path)
            
            if isinstance(result, str):
                return f\"✅ {result}\"
            else:
                return \"✅ Dataset generated successfully. Use path parameter to export to file.\"
        except Exception as e:
            return f\"❌ Export error: {str(e)}\"
    
    def get_proactive_suggestion(self):
        \"\"\"
        Section N: Generate proactive business suggestions.
        Returns contextual suggestions based on detected patterns.
        \"\"\"
        try:
            # Mock business data - would connect to real data
            business_data = {
                'overdue_debt': 50000,
                'revenue': 100000,
                'negative_cashflow_days': 0
            }
            
            # Detect risks
            risks = self.proactive_analyzer.detect_risks(business_data)
            
            if risks:
                return self.proactive_analyzer.generate_proactive_suggestion()
            
            return None
        except:
            return None
    
    def validate_forecast_quality(self, forecast_data, threshold=70):
        \"\"\"
        Section P: Data science governance - validate forecasts.
        \"\"\"
        is_valid, message = DataScienceGovernance.validate_forecast(
            forecast_data, 
            threshold
        )
        return is_valid, message
    
    def get_context_summary(self):
        \"\"\"
        Section M: Get conversation context summary.
        \"\"\"
        return self.context_memory.get_context_summary()
    
    def propose_learning(self, user_query, ai_response, user_feedback):
        \"\"\"
        Section L: Controlled learning - propose new knowledge.
        Requires user approval before activation.
        \"\"\"
        return self.learning_engine.observe_interaction(
            user_query, 
            ai_response, 
            user_feedback
        )
    
    def approve_learning_candidate(self, candidate_id):
        \"\"\"
        Section L: Approve and activate learning.
        \"\"\"
        return self.learning_engine.approve_learning(candidate_id)
