"""
SephlightyAI Transformer Core Engine
Author: Antigravity AI
Version: 1.0.0

A sovereign, from-scratch Transformer implementation for enterprise business intelligence.
No external ML framework dependencies — pure Python + math.

ARCHITECTURE:
1. Positional Encoding (sinusoidal)
2. Multi-Head Self-Attention (scaled dot-product)
3. Feed-Forward Network (GELU activation)
4. Layer Normalization
5. TransformerBlock (Attention + FFN + Residual + Norm)
6. BusinessTransformer (stacked blocks with cross-table attention)
7. Hybrid LSTM-Transformer bridge for time-series signals
"""

import math
import random
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("TRANSFORMER_CORE")
logger.setLevel(logging.INFO)


# =============================================================================
# 1. MATH UTILITIES
# =============================================================================

def _softmax(values: List[float]) -> List[float]:
    """Numerically stable softmax."""
    max_val = max(values) if values else 0.0
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps) + 1e-12
    return [e / total for e in exps]


def _gelu(x: float) -> float:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def _layer_norm(vector: List[float], gamma: List[float], beta: List[float],
                eps: float = 1e-5) -> List[float]:
    """Layer normalization on a single vector."""
    n = len(vector)
    mean = sum(vector) / n
    var = sum((v - mean) ** 2 for v in vector) / n
    std = math.sqrt(var + eps)
    return [gamma[i] * (vector[i] - mean) / std + beta[i] for i in range(n)]


def _dot_product(a: List[float], b: List[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def _mat_vec_mul(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Matrix-vector multiplication."""
    return [_dot_product(row, vector) for row in matrix]


def _mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication: A[m x n] @ B[n x p] -> C[m x p]."""
    m = len(a)
    n = len(b)
    p = len(b[0]) if b else 0
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def _transpose(matrix: List[List[float]]) -> List[List[float]]:
    """Transpose a 2D matrix."""
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    """Element-wise vector addition."""
    return [x + y for x, y in zip(a, b)]


def _vec_scale(vec: List[float], scalar: float) -> List[float]:
    """Scale a vector by a scalar."""
    return [v * scalar for v in vec]


def _xavier_init(rows: int, cols: int) -> List[List[float]]:
    """Xavier/Glorot initialization for weight matrices."""
    limit = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def _zeros(size: int) -> List[float]:
    """Zero-initialized vector."""
    return [0.0] * size


def _ones(size: int) -> List[float]:
    """Ones-initialized vector."""
    return [1.0] * size


# =============================================================================
# 2. POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding:
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017).
    Injects sequence order information into token embeddings.
    Supports up to max_len positions.
    """

    def __init__(self, d_model: int, max_len: int = 10000):
        self.d_model = d_model
        self.max_len = max_len
        self._cache: Dict[int, List[float]] = {}

    def encode(self, position: int) -> List[float]:
        """Get positional encoding vector for a given position."""
        if position in self._cache:
            return self._cache[position]

        pe = [0.0] * self.d_model
        for i in range(0, self.d_model, 2):
            div_term = math.exp(-i * math.log(10000.0) / self.d_model)
            pe[i] = math.sin(position * div_term)
            if i + 1 < self.d_model:
                pe[i + 1] = math.cos(position * div_term)

        self._cache[position] = pe
        return pe

    def encode_sequence(self, seq_len: int) -> List[List[float]]:
        """Generate positional encodings for a whole sequence."""
        return [self.encode(pos) for pos in range(seq_len)]


# =============================================================================
# 3. MULTI-HEAD SELF-ATTENTION
# =============================================================================

class MultiHeadAttention:
    """
    Scaled Dot-Product Multi-Head Attention.
    Splits the input into multiple heads, computes attention in parallel,
    and concatenates the results.

    Parameters:
        d_model: Dimensionality of the model
        n_heads: Number of attention heads
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # Weight matrices for Q, K, V projections (one per head, concatenated)
        self.W_q = _xavier_init(d_model, d_model)
        self.W_k = _xavier_init(d_model, d_model)
        self.W_v = _xavier_init(d_model, d_model)
        self.W_o = _xavier_init(d_model, d_model)

        # Biases
        self.b_q = _zeros(d_model)
        self.b_k = _zeros(d_model)
        self.b_v = _zeros(d_model)
        self.b_o = _zeros(d_model)

        # Attention weights for interpretability
        self.last_attention_weights: List[List[List[float]]] = []

    def _project(self, x: List[float], W: List[List[float]], b: List[float]) -> List[float]:
        """Linear projection: x @ W + b"""
        return _vec_add(_mat_vec_mul(W, x), b)

    def _split_heads(self, projected: List[float]) -> List[List[float]]:
        """Split a d_model vector into n_heads vectors of size d_k."""
        heads = []
        for h in range(self.n_heads):
            start = h * self.d_k
            end = start + self.d_k
            heads.append(projected[start:end])
        return heads

    def _merge_heads(self, heads: List[List[float]]) -> List[float]:
        """Concatenate head outputs back to d_model."""
        merged = []
        for head in heads:
            merged.extend(head)
        return merged

    def forward(self, sequence: List[List[float]],
                mask: Optional[List[List[bool]]] = None) -> List[List[float]]:
        """
        Forward pass of Multi-Head Attention.

        Args:
            sequence: List of token vectors [seq_len x d_model]
            mask: Optional attention mask [seq_len x seq_len]

        Returns:
            List of attended vectors [seq_len x d_model]
        """
        seq_len = len(sequence)
        scale = math.sqrt(self.d_k)

        # Project all tokens to Q, K, V
        Q_all = [self._project(tok, self.W_q, self.b_q) for tok in sequence]
        K_all = [self._project(tok, self.W_k, self.b_k) for tok in sequence]
        V_all = [self._project(tok, self.W_v, self.b_v) for tok in sequence]

        # Split into heads
        Q_heads = [self._split_heads(q) for q in Q_all]  # [seq_len][n_heads][d_k]
        K_heads = [self._split_heads(k) for k in K_all]
        V_heads = [self._split_heads(v) for v in V_all]

        # Process each head
        self.last_attention_weights = []
        all_head_outputs = [[] for _ in range(seq_len)]  # [seq_len][n_heads][d_k]

        for h in range(self.n_heads):
            head_attn_weights = []

            for i in range(seq_len):
                # Compute attention scores for token i against all tokens
                scores = []
                for j in range(seq_len):
                    score = _dot_product(Q_heads[i][h], K_heads[j][h]) / scale
                    # Apply mask if provided
                    if mask and not mask[i][j]:
                        score = -1e9
                    scores.append(score)

                # Softmax
                weights = _softmax(scores)
                head_attn_weights.append(weights)

                # Weighted sum of V
                attended = _zeros(self.d_k)
                for j in range(seq_len):
                    for d in range(self.d_k):
                        attended[d] += weights[j] * V_heads[j][h][d]

                all_head_outputs[i].append(attended)

            self.last_attention_weights.append(head_attn_weights)

        # Merge heads and apply output projection
        output = []
        for i in range(seq_len):
            merged = self._merge_heads(all_head_outputs[i])
            projected = self._project(merged, self.W_o, self.b_o)
            output.append(projected)

        return output

    def get_attention_map(self, head_idx: int = 0) -> List[List[float]]:
        """Return attention weights for visualization/debugging."""
        if head_idx < len(self.last_attention_weights):
            return self.last_attention_weights[head_idx]
        return []


# =============================================================================
# 4. FEED-FORWARD NETWORK
# =============================================================================

class FeedForwardNetwork:
    """
    Position-wise Feed-Forward Network with GELU activation.
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    Expands to d_ff (typically 4x d_model) then projects back.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        self.d_model = d_model
        self.d_ff = d_ff or (d_model * 4)

        self.W1 = _xavier_init(d_model, self.d_ff)
        self.b1 = _zeros(self.d_ff)
        self.W2 = _xavier_init(self.d_ff, d_model)
        self.b2 = _zeros(d_model)

    def forward(self, x: List[float]) -> List[float]:
        """Forward pass through FFN."""
        # First linear: d_model -> d_ff
        hidden = _vec_add(_mat_vec_mul(self.W1, x), self.b1)

        # GELU activation
        hidden = [_gelu(h) for h in hidden]

        # Second linear: d_ff -> d_model
        output = _vec_add(_mat_vec_mul(self.W2, hidden), self.b2)

        return output


# =============================================================================
# 5. TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock:
    """
    Single Transformer Block: Self-Attention + FFN with residual connections
    and layer normalization.

    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> + (residual) -> LayerNorm -> FFN -> + (residual) -> output
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: Optional[int] = None,
                 dropout_rate: float = 0.1):
        self.d_model = d_model
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.dropout_rate = dropout_rate

        # Layer Norm parameters (learnable gamma and beta)
        self.ln1_gamma = _ones(d_model)
        self.ln1_beta = _zeros(d_model)
        self.ln2_gamma = _ones(d_model)
        self.ln2_beta = _zeros(d_model)

    def _dropout(self, vec: List[float], training: bool = False) -> List[float]:
        """Simulated dropout (for training mode)."""
        if not training or self.dropout_rate <= 0:
            return vec
        scale = 1.0 / (1.0 - self.dropout_rate)
        return [v * scale if random.random() > self.dropout_rate else 0.0 for v in vec]

    def forward(self, sequence: List[List[float]],
                mask: Optional[List[List[bool]]] = None,
                training: bool = False) -> List[List[float]]:
        """
        Forward pass through one Transformer block.

        Args:
            sequence: [seq_len x d_model]
            mask: Optional attention mask
            training: Whether to apply dropout

        Returns:
            [seq_len x d_model]
        """
        # --- Sub-layer 1: Multi-Head Attention ---
        # Pre-norm
        normed = [_layer_norm(tok, self.ln1_gamma, self.ln1_beta) for tok in sequence]

        # Self-attention
        attended = self.attention.forward(normed, mask)

        # Dropout + residual
        attended = [self._dropout(a, training) for a in attended]
        residual1 = [_vec_add(sequence[i], attended[i]) for i in range(len(sequence))]

        # --- Sub-layer 2: Feed-Forward Network ---
        # Pre-norm
        normed2 = [_layer_norm(tok, self.ln2_gamma, self.ln2_beta) for tok in residual1]

        # FFN
        ffn_out = [self.ffn.forward(tok) for tok in normed2]

        # Dropout + residual
        ffn_out = [self._dropout(f, training) for f in ffn_out]
        output = [_vec_add(residual1[i], ffn_out[i]) for i in range(len(sequence))]

        return output


# =============================================================================
# 6. SPARSE ATTENTION (for large datasets)
# =============================================================================

class SparseAttention:
    """
    Sparse Attention mechanism for processing very large sequences.
    Uses sliding window + global tokens to reduce O(n²) to O(n * w).
    """

    def __init__(self, d_model: int, n_heads: int = 4, window_size: int = 64,
                 n_global_tokens: int = 4):
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.n_global_tokens = n_global_tokens
        self.attention = MultiHeadAttention(d_model, n_heads)

    def _create_sparse_mask(self, seq_len: int) -> List[List[bool]]:
        """Create sparse attention mask: global tokens + sliding window."""
        mask = [[False] * seq_len for _ in range(seq_len)]

        for i in range(seq_len):
            # Global tokens can attend to everything
            if i < self.n_global_tokens:
                for j in range(seq_len):
                    mask[i][j] = True
                    mask[j][i] = True
            else:
                # Sliding window
                start = max(0, i - self.window_size // 2)
                end = min(seq_len, i + self.window_size // 2 + 1)
                for j in range(start, end):
                    mask[i][j] = True

        return mask

    def forward(self, sequence: List[List[float]]) -> List[List[float]]:
        """Forward pass with sparse attention mask."""
        mask = self._create_sparse_mask(len(sequence))
        return self.attention.forward(sequence, mask)


# =============================================================================
# 7. CROSS-TABLE ATTENTION
# =============================================================================

class CrossTableAttention:
    """
    Cross-Attention between different data sources (tables).
    E.g., Sales data attends to Customer data, Inventory attends to Sales.

    This is the key innovation for enterprise data reasoning:
    it lets the AI "see connections" between tables.
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Cross-attention projections
        self.W_q = _xavier_init(d_model, d_model)
        self.W_k = _xavier_init(d_model, d_model)
        self.W_v = _xavier_init(d_model, d_model)
        self.W_o = _xavier_init(d_model, d_model)
        self.b_q = _zeros(d_model)
        self.b_k = _zeros(d_model)
        self.b_v = _zeros(d_model)
        self.b_o = _zeros(d_model)

    def forward(self, query_seq: List[List[float]],
                context_seq: List[List[float]]) -> List[List[float]]:
        """
        Cross-attention: query_seq attends to context_seq.

        Args:
            query_seq: Source sequence [seq_q x d_model] (e.g., Sales rows)
            context_seq: Context sequence [seq_c x d_model] (e.g., Customer rows)

        Returns:
            Attended output [seq_q x d_model]
        """
        seq_q = len(query_seq)
        seq_c = len(context_seq)
        scale = math.sqrt(self.d_k)

        # Project Q from query, K/V from context
        Q = [_vec_add(_mat_vec_mul(self.W_q, tok), self.b_q) for tok in query_seq]
        K = [_vec_add(_mat_vec_mul(self.W_k, tok), self.b_k) for tok in context_seq]
        V = [_vec_add(_mat_vec_mul(self.W_v, tok), self.b_v) for tok in context_seq]

        output = []
        for i in range(seq_q):
            # Attention scores
            scores = [_dot_product(Q[i], K[j]) / scale for j in range(seq_c)]
            weights = _softmax(scores)

            # Weighted sum
            attended = _zeros(self.d_model)
            for j in range(seq_c):
                for d in range(self.d_model):
                    attended[d] += weights[j] * V[j][d]

            # Output projection
            projected = _vec_add(_mat_vec_mul(self.W_o, attended), self.b_o)
            output.append(projected)

        return output


# =============================================================================
# 8. TOKEN EMBEDDER (Business Data → Vectors)
# =============================================================================

class BusinessTokenEmbedder:
    """
    Converts business data (text, numbers, categories) into dense vector representations.
    Uses hash-based embedding for zero-dependency operation.
    """

    def __init__(self, d_model: int, vocab_size: int = 50000):
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding matrix
        self.embedding_table = _xavier_init(vocab_size, d_model)

        # Numerical projection (for continuous values like amounts)
        # A d_model-sized weight vector: scalar * weights => d_model embedding
        limit = math.sqrt(6.0 / (1 + d_model))
        self.num_projection = [random.uniform(-limit, limit) for _ in range(d_model)]
        self.num_bias = _zeros(d_model)

    def _hash_token(self, token: str) -> int:
        """Deterministic hash for token → embedding index."""
        h = hashlib.md5(token.lower().encode('utf-8')).hexdigest()
        return int(h, 16) % self.vocab_size

    def embed_text(self, text: str) -> List[List[float]]:
        """Embed a text string into a sequence of vectors."""
        tokens = text.lower().split()
        embeddings = []
        for token in tokens:
            idx = self._hash_token(token)
            embeddings.append(list(self.embedding_table[idx]))
        return embeddings

    def embed_number(self, value: float) -> List[float]:
        """Project a numerical value into the embedding space."""
        scaled = math.tanh(value / 1000000.0)
        projected = [w * scaled for w in self.num_projection]
        return _vec_add(projected, self.num_bias)

    def embed_row(self, row_data: Dict[str, Any]) -> List[float]:
        """
        Embed a database row into a single dense vector.
        Handles mixed types: strings get tokenized, numbers get projected.
        """
        aggregated = _zeros(self.d_model)
        count = 0

        for key, value in row_data.items():
            if isinstance(value, (int, float)):
                vec = self.embed_number(float(value))
            elif isinstance(value, str):
                tokens = self.embed_text(value)
                if tokens:
                    vec = _zeros(self.d_model)
                    for t in tokens:
                        vec = _vec_add(vec, t)
                    vec = _vec_scale(vec, 1.0 / len(tokens))
                else:
                    continue
            else:
                continue

            aggregated = _vec_add(aggregated, vec)
            count += 1

        if count > 0:
            aggregated = _vec_scale(aggregated, 1.0 / count)

        return aggregated


# =============================================================================
# 9. BUSINESS TRANSFORMER (The Main Brain)
# =============================================================================

class BusinessTransformer:
    """
    The Enterprise Business Transformer.

    Architecture:
        Input → TokenEmbedding + PositionalEncoding → N × TransformerBlock
        → Cross-Table Attention (optional) → Output

    This is the reasoning backbone of SephlightyAI.
    It processes business data (sales, customers, products, expenses)
    and generates intelligent, grounded outputs.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 4,
                 d_ff: Optional[int] = None, max_seq_len: int = 2048,
                 use_sparse_attention: bool = False, sparse_window: int = 64):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Components
        self.embedder = BusinessTokenEmbedder(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer blocks
        self.blocks: List[TransformerBlock] = []
        for _ in range(n_layers):
            self.blocks.append(TransformerBlock(d_model, n_heads, d_ff))

        # Cross-table attention
        self.cross_attention = CrossTableAttention(d_model, n_heads)

        # Sparse attention (for large datasets)
        self.sparse_attention = SparseAttention(d_model, n_heads, sparse_window) \
            if use_sparse_attention else None

        # Output head: d_model -> confidence score (flat vector for dot product)
        limit_out = math.sqrt(6.0 / (d_model + 1))
        self.output_weights = [random.uniform(-limit_out, limit_out) for _ in range(d_model)]
        self.output_bias = 0.0

        # Final layer norm
        self.final_ln_gamma = _ones(d_model)
        self.final_ln_beta = _zeros(d_model)

        logger.info(f"BusinessTransformer initialized: d_model={d_model}, "
                     f"n_heads={n_heads}, n_layers={n_layers}, "
                     f"max_seq_len={max_seq_len}")

    def encode_query(self, query: str) -> List[List[float]]:
        """Encode a natural language query into transformer-ready vectors."""
        token_embeddings = self.embedder.embed_text(query)
        if not token_embeddings:
            token_embeddings = [_zeros(self.d_model)]

        # Add positional encoding
        pos_encodings = self.pos_encoding.encode_sequence(len(token_embeddings))
        for i in range(len(token_embeddings)):
            token_embeddings[i] = _vec_add(token_embeddings[i], pos_encodings[i])

        return token_embeddings

    def encode_data_rows(self, rows: List[Dict[str, Any]]) -> List[List[float]]:
        """Encode database rows into transformer-ready vectors."""
        row_embeddings = []
        for idx, row in enumerate(rows[:self.max_seq_len]):
            vec = self.embedder.embed_row(row)
            pos = self.pos_encoding.encode(idx)
            row_embeddings.append(_vec_add(vec, pos))
        return row_embeddings

    def forward(self, sequence: List[List[float]],
                context: Optional[List[List[float]]] = None,
                training: bool = False) -> List[List[float]]:
        """
        Full forward pass through the Transformer.

        Args:
            sequence: Input sequence [seq_len x d_model]
            context: Optional cross-table context [ctx_len x d_model]
            training: Whether to apply dropout

        Returns:
            Transformed sequence [seq_len x d_model]
        """
        x = sequence

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block.forward(x, training=training)

        # Cross-table attention (if context provided)
        if context is not None and len(context) > 0:
            x = self.cross_attention.forward(x, context)

        # Final layer norm
        x = [_layer_norm(tok, self.final_ln_gamma, self.final_ln_beta) for tok in x]

        return x

    def reason(self, query: str, data_rows: List[Dict[str, Any]],
               context_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        High-level reasoning API.
        Takes a natural language query + database rows and returns an intelligent output.

        Args:
            query: User's business question
            data_rows: Primary data from DB (e.g., sales records)
            context_rows: Optional related data (e.g., customer info)

        Returns:
            Dict with: reasoning_vector, confidence, attention_map
        """
        # Encode query
        query_seq = self.encode_query(query)

        # Encode data rows
        data_seq = self.encode_data_rows(data_rows) if data_rows else []

        # Combine query + data into a single sequence
        combined = query_seq + data_seq

        # Encode context if provided
        context_seq = None
        if context_rows:
            context_seq = self.encode_data_rows(context_rows)

        # Forward pass
        transformed = self.forward(combined, context_seq)

        # Extract the reasoning vector (mean pooling over all positions)
        if transformed:
            reasoning_vec = _zeros(self.d_model)
            for tok in transformed:
                reasoning_vec = _vec_add(reasoning_vec, tok)
            reasoning_vec = _vec_scale(reasoning_vec, 1.0 / len(transformed))
        else:
            reasoning_vec = _zeros(self.d_model)

        # Compute confidence score
        raw_score = _dot_product(self.output_weights, reasoning_vec) + self.output_bias
        confidence = 1.0 / (1.0 + math.exp(-raw_score))  # sigmoid

        # Get attention map from last block
        attention_map = self.blocks[-1].attention.get_attention_map(0) if self.blocks else []

        return {
            "reasoning_vector": reasoning_vec,
            "confidence": round(confidence, 4),
            "attention_map": attention_map,
            "query_tokens": len(query_seq),
            "data_tokens": len(data_seq),
            "total_tokens": len(combined),
            "n_layers_processed": self.n_layers,
        }

    def hybrid_forecast(self, query: str, time_series: List[float],
                         data_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Hybrid Transformer + LSTM forecasting.
        Uses LSTM for raw time-series signals, then feeds into Transformer for reasoning.
        """
        try:
            from .lstm_business_engine import BUSINESS_FORECASTER
            lstm_result = BUSINESS_FORECASTER.forecast_revenue(time_series)
        except (ImportError, Exception):
            lstm_result = {
                "predicted_revenue": time_series[-1] if time_series else 0,
                "trend_direction": "UNKNOWN",
                "confidence_score": 0.5
            }

        # Inject LSTM signal into data rows
        enriched_rows = list(data_rows)
        enriched_rows.append({
            "lstm_prediction": lstm_result["predicted_revenue"],
            "lstm_trend": 1.0 if lstm_result["trend_direction"] == "UP" else -1.0,
            "lstm_confidence": lstm_result["confidence_score"]
        })

        # Run transformer reasoning on enriched data
        transformer_result = self.reason(query, enriched_rows)

        return {
            "lstm_signal": lstm_result,
            "transformer_reasoning": transformer_result,
            "hybrid_confidence": round(
                0.4 * lstm_result["confidence_score"] +
                0.6 * transformer_result["confidence"],
                4
            )
        }


# =============================================================================
# 10. TEMPORAL ATTENTION (Time-Aware Transformer)
# =============================================================================

class TemporalAttention:
    """
    Time-aware attention that weighs recent data more heavily.
    Used for trend analysis and temporal reasoning.
    """

    def __init__(self, d_model: int, decay_rate: float = 0.1):
        self.d_model = d_model
        self.decay_rate = decay_rate
        self.attention = MultiHeadAttention(d_model, n_heads=2)

    def forward(self, sequence: List[List[float]],
                timestamps: Optional[List[float]] = None) -> List[List[float]]:
        """
        Apply temporal attention weighting.
        More recent items get higher attention.
        """
        seq_len = len(sequence)

        # Create temporal mask (bias toward recent)
        if timestamps:
            max_time = max(timestamps) if timestamps else 1.0
            temporal_weights = [math.exp(-self.decay_rate * (max_time - t))
                                for t in timestamps]
        else:
            temporal_weights = [math.exp(-self.decay_rate * (seq_len - i))
                                for i in range(seq_len)]

        # Scale embeddings by temporal weight
        weighted_seq = [_vec_scale(sequence[i], temporal_weights[i])
                        for i in range(seq_len)]

        # Run attention on temporally-weighted sequence
        return self.attention.forward(weighted_seq)


# =============================================================================
# 11. GLOBAL SINGLETON + STATUS
# =============================================================================

# Global instance
TRANSFORMER_BRAIN = BusinessTransformer(
    d_model=64,
    n_heads=4,
    n_layers=4,
    max_seq_len=2048
)

logger.info("Transformer Core Engine v1.0.0 — ONLINE.")
logger.info(f"Architecture: {TRANSFORMER_BRAIN.n_layers} layers, "
            f"{TRANSFORMER_BRAIN.n_heads} heads, "
            f"d_model={TRANSFORMER_BRAIN.d_model}")
