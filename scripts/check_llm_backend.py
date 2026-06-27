import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()

from config.llm_config import get_llm_config, get_model_name

cfg = get_llm_config()
url = cfg["base_url"]

if "11434" in url or "ollama" in url:
    backend = "Ollama (local — free, no limits)"
elif "openrouter" in url:
    backend = "OpenRouter (cloud — fast, rate limited)"
elif "groq" in url:
    backend = "Groq (cloud — very fast, rate limited)"
else:
    backend = "Direct OpenAI"

print(f"Current LLM backend: {backend}")
print(f"Model:    {get_model_name()}")
print(f"Base URL: {url}")
print(f"API Key:  {cfg['api_key'][:12]}...")