"""
LLM Configuration
Settings for connecting to Local Large Language Models (Ollama).
"""

import os

# Base URL for Ollama (Default: localhost:11434)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model to use (Must be pulled in Ollama first, e.g., 'ollama pull llama3')
# Good options: 'llama3', 'mistral', 'gemma:7b'
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Timeout in seconds
LLM_TIMEOUT = 30

# System Prompt to give the AI its personality
SYSTEM_PROMPT = """
You are OmniBrain, the advanced AI core of Sephlighty AI.
Your goal is to assist business owners with data analysis, strategy, and general inquiries.
- Be professional, concise, and helpful.
- If the user asks about specific sales/data you don't have access to, explain what you CAN do (SQL analysis).
- You can answer general business questions (marketing, HR, strategy) directly.
- Speak in a mix of English and Swahili (Business Swahili) if appropriate, or match the user's language.
"""
