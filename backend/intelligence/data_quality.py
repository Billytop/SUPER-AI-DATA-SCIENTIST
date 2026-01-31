"""
Data Quality Checker and Intelligence Module
Validates data quality, detects duplicates, anomalies, and provides data insights.
"""

from typing import Dict, List, Optional, Set
from datetime import datetime
import re
from collections import defaultdict


class DataQualityChecker:
    """
    Comprehensive data quality validation and scoring.
    """
    
    def __init__(self):
        self.quality_rules = self._init_quality_rules()
        self.validation_history = []
        
    def validate_record(self, record: Dict, record_type: str) -> Dict:
        """
        Validate a single data record.
        
        Args:
            record: Data record to validate
            record_type: Type of record (customer, product, transaction)
            
        Returns:
            Validation result with issues and score
        """
        issues = []
        warnings = []
        
        rules = self.quality_rules.get(record_type, {})
        
        # Check required fields
        for field in rules.get('required_fields', []):
            if field not in record or not record[field]:
                issues.append(f'Missing required field: {field}')
        
        # Check data formats
        for field, format_type in rules.get('formats', {}).items():
            if field in record and record[field]:
                if not self._validate_format(record[field], format_type):
                    issues.append(f'Invalid format for {field}: expected {format_type}')
        
        # Check value ranges
        for field, range_spec in rules.get('ranges', {}).items():
            if field in record and record[field] is not None:
                value = record[field]
                if 'min' in range_spec and value < range_spec['min']:
                    issues.append(f'{field} below minimum: {value} < {range_spec["min"]}')
                if 'max' in range_spec and value > range_spec['max']:
                    warnings.append(f'{field} above maximum: {value} > {range_spec["max"]}')
        
        # Check logical constraints
        if record_type == 'transaction':
            if 'total' in record and 'subtotal' in record:
                if record['total'] < record['subtotal']:
                    issues.append('Total cannot be less than subtotal')
        
        # Calculate quality score
        total_checks = (
            len(rules.get('required_fields', [])) +
            len(rules.get('formats', {})) +
            len(rules.get('ranges', {}))
        )
        
        failed_checks = len(issues) + (len(warnings) * 0.5)
        quality_score = ((total_checks - failed_checks) / total_checks * 100) if total_checks > 0 else 100
        
        result = {
            'valid': len(issues) == 0,
            'quality_score': max(0, quality_score),
            'issues': issues,
            'warnings': warnings,
            'checked_at': datetime.now().isoformat()
        }
        
        self.validation_history.append(result)
        
        return result
    
    def validate_dataset(self, records: List[Dict], record_type: str) -> Dict:
        """Validate entire dataset."""
        results = []
        
        for record in records:
            result = self.validate_record(record, record_type)
            results.append(result)
        
        # Calculate overall statistics
        valid_count = sum(1 for r in results if r['valid'])
        avg_quality = sum(r['quality_score'] for r in results) / len(results) if results else 0
        
        all_issues = []
        for r in results:
            all_issues.extend(r['issues'])
        
        # Count issue types
        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_counts[issue] += 1
        
        return {
            'total_records': len(records),
            'valid_records': valid_count,
            'invalid_records': len(records) - valid_count,
            'validation_rate': (valid_count / len(records) * 100) if records else 0,
            'avg_quality_score': avg_quality,
            'common_issues': sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'results': results
        }
    
    def get_data_completeness(self, records: List[Dict], expected_fields: List[str]) -> Dict:
        """Check data completeness across records."""
        if not records:
            return {'completeness': 0}
        
        field_completeness = {}
        
        for field in expected_fields:
            non_empty = sum(1 for r in records if r.get(field))
            completeness = (non_empty / len(records) * 100)
            field_completeness[field] = completeness
        
        overall_completeness = sum(field_completeness.values()) / len(expected_fields) if expected_fields else 0
        
        return {
            'overall_completeness': overall_completeness,
            'field_completeness': field_completeness,
            'incomplete_fields': [f for f, c in field_completeness.items() if c < 90]
        }
    
    def detect_data_anomalies(self, records: List[Dict], field: str) -> List[Dict]:
        """Detect statistical anomalies in numeric field."""
        values = [r[field] for r in records if field in r and isinstance(r[field], (int, float))]
        
        if len(values) < 3:
            return []
        
        # Calculate statistics
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        # Find outliers (values beyond 3 standard deviations)
        anomalies = []
        for i, record in enumerate(records):
            if field in record and isinstance(record[field], (int, float)):
                value = record[field]
                z_score = abs((value - mean) / std_dev) if std_dev > 0 else 0
                
                if z_score > 3:
                    anomalies.append({
                        'record_index': i,
                        'value': value,
                        'z_score': z_score,
                        'deviation_from_mean': value - mean,
                        'severity': 'high' if z_score > 5 else 'medium'
                    })
        
        return anomalies
    
    def _validate_format(self, value: any, format_type: str) -> bool:
        """Validate value against format type."""
        if format_type == 'email':
            return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str(value)))
        elif format_type == 'phone':
            return bool(re.match(r'^\+?[0-9]{10,15}$', str(value).replace(' ', '').replace('-', '')))
        elif format_type == 'date':
            try:
                datetime.fromisoformat(str(value))
                return True
            except:
                return False
        elif format_type == 'number':
            return isinstance(value, (int, float))
        elif format_type == 'text':
            return isinstance(value, str) and len(value) > 0
        else:
            return True
    
    def _init_quality_rules(self) -> Dict:
        """Initialize data quality rules."""
        return {
            'customer': {
                'required_fields': ['name', 'contact'],
                'formats': {
                    'email': 'email',
                    'phone': 'phone'
                },
                'ranges': {
                    'credit_limit': {'min': 0, 'max': 100000000}
                }
            },
            'product': {
                'required_fields': ['name', 'price'],
                'formats': {
                    'price': 'number',
                    'sku': 'text'
                },
                'ranges': {
                    'price': {'min': 0},
                    'stock_quantity': {'min': 0}
                }
            },
            'transaction': {
                'required_fields': ['date', 'total', 'customer_id'],
                'formats': {
                    'date': 'date',
                    'total': 'number'
                },
                'ranges': {
                    'total': {'min': 0}
                }
            }
        }


class DuplicateDetector:
    """
    Detects duplicate and similar records.
    """
    
    def __init__(self):
        self.similarity_threshold = 0.85
        
    def find_exact_duplicates(self, records: List[Dict], key_fields: List[str]) -> List[List[int]]:
        """Find exact duplicate records based on key fields."""
        signature_map = defaultdict(list)
        
        for i, record in enumerate(records):
            # Create signature from key fields
            signature = self._create_signature(record, key_fields)
            signature_map[signature].append(i)
        
        # Return groups with duplicates
        duplicates = [indices for indices in signature_map.values() if len(indices) > 1]
        
        return duplicates
    
    def find_similar_records(self, records: List[Dict], compare_fields: List[str]) -> List[Dict]:
        """Find similar but not identical records."""
        similar_pairs = []
        
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                similarity = self._calculate_similarity(records[i], records[j], compare_fields)
                
                if self.similarity_threshold <= similarity < 1.0:
                    similar_pairs.append({
                        'index1': i,
                        'index2': j,
                        'similarity': similarity,
                        'record1': records[i],
                        'record2': records[j]
                    })
        
        return sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)
    
    def find_fuzzy_customer_duplicates(self, customers: List[Dict]) -> List[Dict]:
        """Find potential duplicate customers using fuzzy matching."""
        duplicates = []
        
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                name_similarity = self._string_similarity(
                    customers[i].get('name', ''),
                    customers[j].get('name', '')
                )
                
                phone_match = (
                    customers[i].get('phone') == customers[j].get('phone')
                    if customers[i].get('phone') and customers[j].get('phone')
                    else False
                )
                
                email_match = (
                    customers[i].get('email') == customers[j].get('email')
                    if customers[i].get('email') and customers[j].get('email')
                    else False
                )
                
                # Duplicate if name is very similar AND (phone OR email matches)
                if name_similarity > 0.9 and (phone_match or email_match):
                    duplicates.append({
                        'customer1': customers[i],
                        'customer2': customers[j],
                        'name_similarity': name_similarity,
                        'phone_match': phone_match,
                        'email_match': email_match,
                        'confidence': 0.95 if (phone_match and email_match) else 0.85
                    })
        
        return duplicates
    
    def suggest_merge_strategy(self, duplicate_records: List[Dict]) -> Dict:
        """Suggest how to merge duplicate records."""
        if len(duplicate_records) < 2:
            return {'error': 'Need at least 2 records to merge'}
        
        # Choose most complete record as base
        completeness_scores = []
        for record in duplicate_records:
            non_empty = sum(1 for v in record.values() if v)
            completeness_scores.append(non_empty)
        
        base_index = completeness_scores.index(max(completeness_scores))
        base_record = duplicate_records[base_index].copy()
        
        # Merge data from other records
        merged_record = base_record.copy()
        
        for i, record in enumerate(duplicate_records):
            if i == base_index:
                continue
            
            for key, value in record.items():
                # Fill in missing values
                if key not in merged_record or not merged_record[key]:
                    merged_record[key] = value
        
        return {
            'base_record_index': base_index,
            'merged_record': merged_record,
            'records_to_delete': [i for i in range(len(duplicate_records)) if i != base_index]
        }
    
    def _create_signature(self, record: Dict, key_fields: List[str]) -> str:
        """Create unique signature from key fields."""
        values = []
        for field in key_fields:
            value = str(record.get(field, '')).lower().strip()
            values.append(value)
        return ':::'.join(values)
    
    def _calculate_similarity(self, record1: Dict, record2: Dict, compare_fields: List[str]) -> float:
        """Calculate similarity score between two records."""
        if not compare_fields:
            return 0.0
        
        total_similarity = 0.0
        
        for field in compare_fields:
            if field in record1 and field in record2:
                val1 = record1[field]
                val2 = record2[field]
                
                if isinstance(val1, str) and isinstance(val2, str):
                    similarity = self._string_similarity(val1, val2)
                else:
                    similarity = 1.0 if val1 == val2 else 0.0
                
                total_similarity += similarity
        
        return total_similarity / len(compare_fields)
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Levenshtein-like algorithm."""
        str1 = str1.lower().strip()
        str2 = str2.lower().strip()
        
        if str1 == str2:
            return 1.0
        
        if not str1 or not str2:
            return 0.0
        
        # Simple character-based similarity
        longer = max(len(str1), len(str2))
        
        # Count matching characters
        matches = sum(1 for a, b in zip(str1, str2) if a == b)
        
        return matches / longer


class DataProfiler:
    """
    Profiles datasets to understand structure and characteristics.
    """
    
    def __init__(self):
        pass
        
    def profile_dataset(self, records: List[Dict]) -> Dict:
        """Generate comprehensive data profile."""
        if not records:
            return {'error': 'Empty dataset'}
        
        # Get all fields
        all_fields = set()
        for record in records:
            all_fields.update(record.keys())
        
        field_profiles = {}
        
        for field in all_fields:
            field_profiles[field] = self._profile_field(records, field)
        
        return {
            'total_records': len(records),
            'total_fields': len(all_fields),
            'fields': list(all_fields),
            'field_profiles': field_profiles,
            'data_types': self._infer_data_types(records),
            'profiled_at': datetime.now().isoformat()
        }
    
    def _profile_field(self, records: List[Dict], field: str) -> Dict:
        """Profile a single field."""
        values = [r[field] for r in records if field in r]
        
        profile = {
            'present_count': len(values),
            'missing_count': len(records) - len(values),
            'completeness': (len(values) / len(records) * 100) if records else 0,
            'unique_values': len(set(str(v) for v in values)),
            'sample_values': list(set(str(v) for v in values))[:5]
        }
        
        # Numeric statistics
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if numeric_values:
            profile['numeric_stats'] = {
                'min': min(numeric_values),
                'max': max(numeric_values),
                'avg': sum(numeric_values) / len(numeric_values),
                'sum': sum(numeric_values)
            }
        
        # String statistics
        string_values = [v for v in values if isinstance(v, str)]
        if string_values:
            lengths = [len(s) for s in string_values]
            profile['string_stats'] = {
                'min_length': min(lengths),
                'max_length': max(lengths),
                'avg_length': sum(lengths) / len(lengths)
            }
        
        return profile
    
    def _infer_data_types(self, records: List[Dict]) -> Dict:
        """Infer data type for each field."""
        if not records:
            return {}
        
        field_types = {}
        
        # Sample first record to get fields
        fields = records[0].keys()
        
        for field in fields:
            types = set()
            for record in records[:100]:  # Sample first 100
                if field in record and record[field] is not None:
                    types.add(type(record[field]).__name__)
            
            # Determine primary type
            if len(types) == 1:
                field_types[field] = list(types)[0]
            elif 'int' in types and 'float' in types:
                field_types[field] = 'numeric'
            else:
                field_types[field] = 'mixed'
        
        return field_types


class DataEnrichment:
    """
    Enriches data with derived fields and calculated values.
    """
    
    def __init__(self):
        self.enrichment_rules = {}
        
    def enrich_customer_data(self, customer: Dict, transactions: List[Dict]) -> Dict:
        """Enrich customer record with transaction insights."""
        enriched = customer.copy()
        
        if transactions:
            # Calculate customer metrics
            enriched['total_purchases'] = sum(t.get('total', 0) for t in transactions)
            enriched['transaction_count'] = len(transactions)
            enriched['avg_purchase_value'] = enriched['total_purchases'] / len(transactions)
            
            # Get date of first and last purchase
            dates = sorted([t.get('date') for t in transactions if t.get('date')])
            if dates:
                enriched['first_purchase_date'] = dates[0]
                enriched['last_purchase_date'] = dates[-1]
                
                # Calculate customer lifetime (days)
                try:
                    first = datetime.fromisoformat(str(dates[0]))
                    last = datetime.fromisoformat(str(dates[-1]))
                    enriched['customer_lifetime_days'] = (last - first).days
                except:
                    pass
            
            # Customer segment
            if enriched['total_purchases'] > 5000000:
                enriched['segment'] = 'VIP'
            elif enriched['total_purchases'] > 1000000:
                enriched['segment'] = 'Premium'
            else:
                enriched['segment'] = 'Standard'
        
        return enriched
    
    def enrich_product_data(self, product: Dict, sales_data: List[Dict]) -> Dict:
        """Enrich product data with sales insights."""
        enriched = product.copy()
        
        if sales_data:
            enriched['total_units_sold'] = sum(s.get('quantity', 0) for s in sales_data)
            enriched['total_revenue'] = sum(s.get('amount', 0) for s in sales_data)
            
            # Average selling price
            if enriched['total_units_sold'] > 0:
                enriched['avg_selling_price'] = enriched['total_revenue'] / enriched['total_units_sold']
            
            # Product velocity (units per day)
            dates = [s.get('date') for s in sales_data if s.get('date')]
            if dates:
                try:
                    first = datetime.fromisoformat(str(min(dates)))
                    last = datetime.fromisoformat(str(max(dates)))
                    days = (last - first).days + 1
                    enriched['sales_velocity'] = enriched['total_units_sold'] / days
                except:
                    pass
        
        return enriched
