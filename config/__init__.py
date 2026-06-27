"""
config/llm_config.py
Central LLM configuration.
Switch between OpenRouter and Ollama by changing .env

OPENAI_BASE_URL=https://openrouter.ai/api/v1  → OpenRouter
OPENAI_BASE_URL=http://localhost:11434/v1      → Ollama (free)
"""
import os
from dotenv import load_dotenv
load_dotenv()


def get_model_name() -> str:
    """
    Returns the correct model name for the
    current LLM backend.

    OpenRouter uses:  "openai/gpt-4o-mini"
    Ollama uses:      "llama3.2" or "qwen2.5:7b"
    Direct OpenAI:    "gpt-4o-mini"
    """
    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1"
    )

    if "openrouter" in base_url:
        return "openai/gpt-4o-mini"
    elif "11434" in base_url or "ollama" in base_url:
        # Use whichever model you pulled
        return os.getenv(
            "OLLAMA_MODEL", "llama3.2")
    else:
        return "gpt-4o-mini"


def get_llm_config() -> dict:
    """Returns full LLM config dict."""
    return {
        "api_key":  os.getenv(
            "OPENAI_API_KEY", "ollama"),
        "base_url": os.getenv(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1"),
        "model":    get_model_name(),
    }