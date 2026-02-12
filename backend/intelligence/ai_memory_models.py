"""
AI Memory Models — SephlightyAI
Structured memory for the AI brain: conversations, facts, decisions, profiles.

MEMORY PRINCIPLES:
  - Facts are IMMUTABLE — never overwritten
  - Interpretations EVOLVE — versioned reasoning
  - Decisions are VERSIONED — full audit trail
  - All memory is searchable and recallable
"""

import uuid
from django.db import models
from django.contrib.auth.models import User
from core.models import Business


class AIConversationMemory(models.Model):
    """
    Long-term conversation summaries.
    After a conversation ends, the AI distills key learnings into this table.
    Used for cross-conversation context recall.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='ai_memories')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ai_memories', null=True)

    # Conversation reference
    conversation_id = models.UUIDField(null=True, blank=True,
        help_text="Links to chat.Conversation")

    # Distilled content
    summary = models.TextField(help_text="AI-generated summary of the conversation")
    key_topics = models.JSONField(default=list,
        help_text="['sales analysis', 'customer debt', 'inventory']")
    key_entities = models.JSONField(default=list,
        help_text="['Product: Cement', 'Customer: John', 'Period: Jan 2026']")
    key_insights = models.JSONField(default=list,
        help_text="Important business insights discovered")

    # Search support
    embedding_hash = models.CharField(max_length=64, null=True, blank=True,
        help_text="Hash of the embedding vector for quick lookup")

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    importance_score = models.FloatField(default=0.5,
        help_text="0.0-1.0, how important this memory is")

    class Meta:
        db_table = 'ai_conversation_memory'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['business', '-created_at']),
            models.Index(fields=['business', '-importance_score']),
        ]

    def __str__(self):
        return f"Memory: {self.summary[:60]}..."


class AIMemoryChunk(models.Model):
    """
    Append-only business facts.
    IMMUTABLE — once written, never modified. New facts are appended.
    This is the ground truth the AI reasons from.
    """
    CHUNK_TYPES = [
        ('fact', 'Business Fact'),
        ('metric', 'Computed Metric'),
        ('pattern', 'Detected Pattern'),
        ('correction', 'User Correction'),
        ('learning', 'AI Learning'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='ai_facts')

    # Content
    chunk_type = models.CharField(max_length=20, choices=CHUNK_TYPES, default='fact')
    content = models.TextField(help_text="The fact/metric/pattern in natural language")
    structured_data = models.JSONField(default=dict,
        help_text="Machine-readable data: {'total_sales': 5000000, 'period': '2026-01'}")

    # Domain tagging
    domain = models.CharField(max_length=50, default='general',
        help_text="sales, purchases, customers, expenses, inventory, health")

    # Source tracking
    source = models.CharField(max_length=100, default='computed',
        help_text="How was this fact derived: computed, user_stated, inferred")
    source_query = models.TextField(null=True, blank=True,
        help_text="The SQL or query that produced this fact")

    # Validity
    confidence = models.FloatField(default=1.0,
        help_text="1.0 for DB facts, lower for inferences")
    is_active = models.BooleanField(default=True,
        help_text="Soft-delete: superseded facts can be marked inactive")
    superseded_by = models.ForeignKey('self', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='supersedes')

    # Search support
    embedding_hash = models.CharField(max_length=64, null=True, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    fact_date = models.DateField(null=True, blank=True,
        help_text="The date this fact refers to (not when it was computed)")

    class Meta:
        db_table = 'ai_memory_chunks'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['business', 'domain', '-created_at']),
            models.Index(fields=['business', 'chunk_type', 'is_active']),
            models.Index(fields=['business', '-created_at']),
        ]

    def __str__(self):
        return f"[{self.chunk_type}] {self.content[:60]}"


class AIDecisionLog(models.Model):
    """
    Versioned AI decisions with full reasoning trail.
    Every significant AI recommendation is logged with its reasoning.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='ai_decisions')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    # Decision content
    domain = models.CharField(max_length=50, default='general')
    decision = models.TextField(help_text="The recommendation/decision made")
    reasoning = models.TextField(help_text="Why the AI made this decision")

    # Input context
    input_query = models.TextField(null=True, blank=True)
    input_data = models.JSONField(default=dict,
        help_text="Snapshot of data used for this decision")

    # Confidence and validation
    confidence = models.FloatField(default=0.8)
    was_accepted = models.BooleanField(null=True,
        help_text="Did the user accept? null=pending, true=accepted, false=rejected")
    user_feedback = models.TextField(null=True, blank=True)

    # Versioning
    version = models.IntegerField(default=1)
    previous_version = models.ForeignKey('self', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='next_versions')

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'ai_decision_log'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['business', 'domain', '-created_at']),
        ]

    def __str__(self):
        return f"Decision v{self.version}: {self.decision[:60]}"


class AICustomerProfile(models.Model):
    """
    AI-maintained per-customer intelligence profile.
    Updated incrementally as new data comes in.
    """
    RISK_LEVELS = [
        ('low', 'Low Risk'),
        ('medium', 'Medium Risk'),
        ('high', 'High Risk'),
        ('critical', 'Critical Risk'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='ai_customer_profiles')
    contact_id = models.IntegerField(help_text="FK to contacts table (ERP)")

    # Profile data
    customer_name = models.CharField(max_length=255, null=True, blank=True)
    risk_level = models.CharField(max_length=20, choices=RISK_LEVELS, default='low')
    risk_score = models.FloatField(default=0.0, help_text="0.0 (safe) to 1.0 (critical)")

    # Payment behavior
    avg_payment_days = models.FloatField(default=0.0,
        help_text="Average days to pay from invoice date")
    payment_reliability = models.FloatField(default=1.0,
        help_text="0.0-1.0 reliability score")
    total_lifetime_value = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    total_outstanding = models.DecimalField(max_digits=22, decimal_places=4, default=0)

    # Purchase patterns
    preferred_products = models.JSONField(default=list)
    purchase_frequency = models.CharField(max_length=50, default='unknown',
        help_text="daily, weekly, monthly, sporadic")
    last_purchase_date = models.DateField(null=True, blank=True)

    # AI insights
    churn_probability = models.FloatField(default=0.0,
        help_text="Probability of customer leaving")
    upsell_potential = models.FloatField(default=0.0)
    ai_notes = models.TextField(null=True, blank=True,
        help_text="AI-generated notes about this customer")

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'ai_customer_profiles'
        unique_together = ['business', 'contact_id']
        indexes = [
            models.Index(fields=['business', 'risk_level']),
            models.Index(fields=['business', '-risk_score']),
        ]

    def __str__(self):
        return f"Profile: {self.customer_name or self.contact_id} ({self.risk_level})"


class AIProductProfile(models.Model):
    """
    AI-maintained per-product intelligence profile.
    Tracks margin, velocity, dead-stock risk, and demand patterns.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='ai_product_profiles')
    product_id = models.IntegerField(help_text="FK to products table (ERP)")

    # Product identity
    product_name = models.CharField(max_length=255, null=True, blank=True)
    category = models.CharField(max_length=100, null=True, blank=True)

    # Performance metrics
    avg_margin_pct = models.FloatField(default=0.0,
        help_text="Average profit margin percentage")
    sales_velocity = models.FloatField(default=0.0,
        help_text="Units sold per month average")
    revenue_contribution = models.FloatField(default=0.0,
        help_text="Percentage of total revenue")

    # Inventory intelligence
    is_dead_stock = models.BooleanField(default=False)
    days_since_last_sale = models.IntegerField(default=0)
    optimal_reorder_point = models.IntegerField(default=0)
    optimal_reorder_qty = models.IntegerField(default=0)

    # Demand patterns
    demand_trend = models.CharField(max_length=20, default='stable',
        help_text="rising, stable, declining, seasonal")
    seasonality_pattern = models.JSONField(default=dict,
        help_text="Monthly demand pattern: {'Jan': 100, 'Feb': 120, ...}")

    # Cross-sell
    frequently_bought_with = models.JSONField(default=list,
        help_text="Product IDs commonly purchased together")

    # AI insights
    ai_notes = models.TextField(null=True, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'ai_product_profiles'
        unique_together = ['business', 'product_id']
        indexes = [
            models.Index(fields=['business', 'is_dead_stock']),
            models.Index(fields=['business', '-sales_velocity']),
        ]

    def __str__(self):
        return f"Product: {self.product_name or self.product_id} ({self.demand_trend})"
