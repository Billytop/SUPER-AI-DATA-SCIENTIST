import math
import random
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# --- LONG SHORT-TERM MEMORY (LSTM) ENGINE ---
# Specialized for Business Time-Series: Revenue, Stock, Churn

class LSTMCell:
    """
    Sovereign LSTM Cell.
    Manages Forget Gate, Input Gate, and Output Gate for long-term dependency tracking.
    """
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weights for Forget Gate
        self.Wf = [[random.uniform(-0.1, 0.1) for _ in range(input_size + hidden_size)] for _ in range(hidden_size)]
        self.bf = [0.0] * hidden_size
        
        # Weights for Input Gate
        self.Wi = [[random.uniform(-0.1, 0.1) for _ in range(input_size + hidden_size)] for _ in range(hidden_size)]
        self.bi = [0.0] * hidden_size
        
        # Weights for Cell Candidate
        self.Wc = [[random.uniform(-0.1, 0.1) for _ in range(input_size + hidden_size)] for _ in range(hidden_size)]
        self.bc = [0.0] * hidden_size
        
        # Weights for Output Gate
        self.Wo = [[random.uniform(-0.1, 0.1) for _ in range(input_size + hidden_size)] for _ in range(hidden_size)]
        self.bo = [0.0] * hidden_size

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def _tanh(self, x: float) -> float:
        return math.tanh(x)

    def forward(self, x: List[float], h_prev: List[float], c_prev: List[float]) -> Tuple[List[float], List[float]]:
        """
        Forward pass for a single time step.
        x: Input vector
        h_prev: Previous hidden state
        c_prev: Previous cell state
        Returns: (h_next, c_next)
        """
        # Concatenate input and previous hidden state
        concat_input = x + h_prev
        
        # Forget Gate
        f_t = []
        for i in range(self.hidden_size):
            dot = sum(w * inp for w, inp in zip(self.Wf[i], concat_input)) + self.bf[i]
            f_t.append(self._sigmoid(dot))
            
        # Input Gate
        i_t = []
        for i in range(self.hidden_size):
            dot = sum(w * inp for w, inp in zip(self.Wi[i], concat_input)) + self.bi[i]
            i_t.append(self._sigmoid(dot))
            
        # Cell Candidate
        c_tilde = []
        for i in range(self.hidden_size):
            dot = sum(w * inp for w, inp in zip(self.Wc[i], concat_input)) + self.bc[i]
            c_tilde.append(self._tanh(dot))
            
        # Update Cell State
        c_next = [f * c + i * c_til for f, c, i, c_til in zip(f_t, c_prev, i_t, c_tilde)]
        
        # Output Gate
        o_t = []
        for i in range(self.hidden_size):
            dot = sum(w * inp for w, inp in zip(self.Wo[i], concat_input)) + self.bo[i]
            o_t.append(self._sigmoid(dot))
            
        # Update Hidden State
        h_next = [o * self._tanh(c) for o, c in zip(o_t, c_next)]
        
        return h_next, c_next

class BusinessLSTMEngine:
    """
    Forecasting Engine using stacked LSTM cells.
    Predicts next-day Revenue, Inventory Demand, and Customer Churn risk.
    """
    def __init__(self):
        self.input_dim = 5 # DayOfWeek, PrevDaySales, MarketingSpend, HolidayFlag, RainFlag
        self.hidden_dim = 10
        self.cell = LSTMCell(self.input_dim, self.hidden_dim)
        
        # Mocking Output Layer (Hidden -> 1 Value)
        self.W_out = [random.uniform(-0.1, 0.1) for _ in range(self.hidden_dim)]
        self.b_out = 0.0

    def predict_sequence(self, time_series_data: List[List[float]]) -> List[float]:
        """
        Feeds a sequence of days into the LSTM to predict the trend.
        """
        h_state = [0.0] * self.hidden_dim
        c_state = [0.0] * self.hidden_dim
        
        predictions = []
        
        for day_data in time_series_data:
            h_state, c_state = self.cell.forward(day_data, h_state, c_state)
            
            # Output projection (Regression)
            pred = sum(w * h for w, h in zip(self.W_out, h_state)) + self.b_out
            predictions.append(pred)
            
        return predictions

    def forecast_revenue(self, history: List[float]) -> Dict[str, float]:
        """
        Wrapper to forecast next week's revenue based on last 30 days.
        """
        # Preprocessing: Normalize data (mock)
        max_val = max(history) if history else 1
        normalized_input = [[val / max_val, 0, 0, 0, 0] for val in history] # Padding output match input_dim
        
        trends = self.predict_sequence(normalized_input)
        
        # Simple post-processing
        next_val = trends[-1] * max_val
        confidence = 0.85 # Mock confidence
        
        return {
            "predicted_revenue": next_val,
            "trend_direction": "UP" if next_val > history[-1] else "DOWN",
            "confidence_score": confidence
        }

BUSINESS_FORECASTER = BusinessLSTMEngine()
