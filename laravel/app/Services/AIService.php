<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use App\Models\AI\AIConversation;
use App\Models\AI\AIDecision;
use App\Models\AI\AICustomerProfile;
use App\Models\AI\AIProductProfile;
use Illuminate\Support\Str;

/**
 * 🚀 SEPHLIGHTY BUSINESS SUPER‑AI 
 * SERVICE: AIService - The Bridge between Laravel and Python AI Engine
 */
class AIService
{
    protected string $aiUrl;

    public function __construct()
    {
        // Points to our FastAPI backend generated in previous phases
        $this->aiUrl = env('AI_ENGINE_URL', 'http://localhost:8000');
    }

    /**
     * Submit a query to the Sephlighty AI Brain
     */
    public function query(string $text, string $sessionId = null, int $userId = null)
    {
        $sessionId = $sessionId ?? Str::uuid();

        // 1. Send to Python Backend
        $response = Http::post("{$this->aiUrl}/query", [
            'query' => $text,
            'session_id' => $sessionId,
            'user_id' => $userId
        ]);

        if ($response->failed()) {
            return ['error' => 'AI Engine Unreachable', 'status' => 503];
        }

        $result = $response->json();

        // 2. Persistent AI Reasoning (Laravel Memory)
        $this->storeDecision($result);

        return $result;
    }

    /**
     * Store key AI decisions and metadata for long-term learning
     */
    protected function storeDecision(array $data)
    {
        if (!isset($data['analysis'])) return;

        $type = $data['analysis']['type'] ?? 'general';
        
        AIDecision::create([
            'decision_type' => $type,
            'prompt_used' => $data['original_query'] ?? '',
            'recommendation' => $data['response'] ?? '',
            'justification' => $data['reasoning'] ?? '',
            'confidence_score' => $data['confidence'] ?? 0.95,
            'impact_metrics' => $data['metrics'] ?? []
        ]);
    }

    /**
     * Update AI-driven behavior scores for customers
     */
    public function updateCustomerIQ(int $customerId, array $intelligence)
    {
        return AICustomerProfile::updateOrCreate(
            ['customer_id' => $customerId],
            [
                'risk_score' => $intelligence['risk'] ?? 0,
                'loyalty_score' => $intelligence['loyalty'] ?? 0,
                'buying_patterns' => $intelligence['patterns'] ?? []
            ]
        );
    }
}
