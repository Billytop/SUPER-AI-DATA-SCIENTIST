<?php

namespace App\Models\AI;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * 🚀 SEPHLIGHTY BUSINESS SUPER‑AI 
 * MODEL: AIConversation
 */
class AIConversation extends Model
{
    protected $table = 'ai_conversations';
    
    protected $fillable = [
        'user_id',
        'session_id',
        'title',
        'metadata',
        'last_active_at'
    ];

    protected $casts = [
        'metadata' => 'array',
        'last_active_at' => 'datetime'
    ];

    public function chunks(): HasMany
    {
        return $this->hasMany(AIMemoryChunk::class, 'source_id')->where('source_type', 'conversation');
    }
}

/**
 * MODEL: AIMemoryChunk
 */
class AIMemoryChunk extends Model
{
    protected $table = 'ai_memory_chunks';

    protected $fillable = [
        'source_type',
        'source_id',
        'content',
        'importance_score',
        'tags'
    ];

    protected $casts = [
        'tags' => 'array'
    ];

    public function embedding(): BelongsTo
    {
        return $this->belongsTo(AIEmbedding::class, 'id', 'chunk_id');
    }
}

/**
 * MODEL: AIEmbedding
 */
class AIEmbedding extends Model
{
    protected $table = 'ai_embeddings';

    protected $fillable = [
        'chunk_id',
        'vector',
        'model'
    ];

    protected $casts = [
        'vector' => 'array'
    ];

    public function chunk(): BelongsTo
    {
        return $this->belongsTo(AIMemoryChunk::class, 'chunk_id');
    }
}

/**
 * MODEL: AIDecision
 */
class AIDecision extends Model
{
    protected $table = 'ai_decisions';

    protected $fillable = [
        'decision_type',
        'prompt_used',
        'recommendation',
        'justification',
        'confidence_score',
        'is_implemented',
        'impact_metrics'
    ];

    protected $casts = [
        'impact_metrics' => 'array',
        'is_implemented' => 'boolean',
        'confidence_score' => 'float'
    ];
}

/**
 * MODEL: AICustomerProfile
 */
class AICustomerProfile extends Model
{
    protected $table = 'ai_customer_profiles';

    protected $fillable = [
        'customer_id',
        'risk_score',
        'loyalty_score',
        'predicted_lifetime_value',
        'buying_patterns',
        'retention_strategy'
    ];

    protected $casts = [
        'risk_score' => 'float',
        'loyalty_score' => 'float',
        'predicted_lifetime_value' => 'float',
        'buying_patterns' => 'array'
    ];
}

/**
 * MODEL: AIProductProfile
 */
class AIProductProfile extends Model
{
    protected $table = 'ai_product_profiles';

    protected $fillable = [
        'product_id',
        'profit_margin_trend',
        'aging_days',
        'is_dead_stock',
        'predicted_demand',
        'seasonality_factors'
    ];

    protected $casts = [
        'profit_margin_trend' => 'float',
        'is_dead_stock' => 'boolean',
        'seasonality_factors' => 'array',
        'predicted_demand' => 'float'
    ];
}
