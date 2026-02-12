<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/**
 * 🚀 SEPHLIGHTY BUSINESS SUPER‑AI 
 * START CODE: AI MEMORY MIGRATIONS (Standard Laravel Blueprint)
 */

return new class extends Migration
{
    public function up(): void
    {
        // 1. AI CONVERSATIONS: Long-context session tracking
        Schema::create('ai_conversations', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('user_id')->index();
            $table->string('session_id')->unique();
            $table->string('title')->nullable();
            $table->json('metadata')->nullable(); // Stores emotional tone, active domain
            $table->timestamp('last_active_at');
            $table->timestamps();
        });

        // 2. AI MEMORY CHUNKS: Atomic knowledge storage for RAG
        Schema::create('ai_memory_chunks', function (Blueprint $table) {
            $table->id();
            $table->string('source_type'); // invoice, product, conversation, external
            $table->unsignedBigInteger('source_id')->index();
            $table->text('content');
            $table->string('importance_score')->default('medium'); // low, medium, high, critical
            $table->json('tags')->nullable();
            $table->timestamps();
        });

        // 3. AI EMBEDDINGS: Vector storage for semantic search
        Schema::create('ai_embeddings', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('chunk_id')->unique();
            $table->json('vector'); // Stored as JSON for SQLite/MySQL compat, or Binary
            $table->string('model')->default('text-embedding-3-small');
            $table->timestamps();
            
            $table->foreign('chunk_id')->references('id')->on('ai_memory_chunks')->onDelete('cascade');
        });

        // 4. AI DECISIONS: Tracking system recommendations and results
        Schema::create('ai_decisions', function (Blueprint $table) {
            $table->id();
            $table->string('decision_type'); // Pricing, Risk, Credit, Optimization
            $table->text('prompt_used');
            $table->text('recommendation');
            $table->text('justification');
            $table->decimal('confidence_score', 5, 2);
            $table->boolean('is_implemented')->default(false);
            $table->json('impact_metrics')->nullable();
            $table->timestamps();
        });

        // 5. AI CUSTOMER PROFILES: Advanced risk and behavior scoring
        Schema::create('ai_customer_profiles', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('customer_id')->unique();
            $table->decimal('risk_score', 5, 2)->default(0.00);
            $table->decimal('loyalty_score', 5, 2)->default(0.00);
            $table->decimal('predicted_lifetime_value', 15, 2)->default(0.00);
            $table->json('buying_patterns')->nullable();
            $table->text('retention_strategy')->nullable();
            $table->timestamps();
        });

        // 6. AI PRODUCT PROFILES: Aging, profit erosion, and demand forecasting
        Schema::create('ai_product_profiles', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('product_id')->unique();
            $table->decimal('profit_margin_trend', 5, 2)->default(0.00);
            $table->integer('aging_days')->default(0);
            $table->boolean('is_dead_stock')->default(false);
            $table->decimal('predicted_demand', 15, 2)->nullable();
            $table->json('seasonality_factors')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('ai_product_profiles');
        Schema::dropIfExists('ai_customer_profiles');
        Schema::dropIfExists('ai_decisions');
        Schema::dropIfExists('ai_embeddings');
        Schema::dropIfExists('ai_memory_chunks');
        Schema::dropIfExists('ai_conversations');
    }
};
