"""
scripts/test_groq_direct.py
Isolates the Planner's LangChain LLM call to see
the REAL underlying exception, not the generic
"Connection error." message.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from langchain_openai import ChatOpenAI
from config.llm_config import get_llm_config
import traceback

cfg = get_llm_config()
print("Config being used:")
print(f"  model:    {cfg['model']}")
print(f"  base_url: {cfg['base_url']}")
print(f"  api_key:  {cfg['api_key'][:10]}...")
print()

llm = ChatOpenAI(
    model=cfg["model"],
    api_key=cfg["api_key"],
    base_url=cfg["base_url"],
    temperature=0.1,
    max_tokens=100,
)

try:
    resp = llm.invoke("Say hello in exactly 5 words")
    print("✅ SUCCESS:", resp.content)
except Exception as e:
    print("❌ FAILED — full traceback below:\n")
    traceback.print_exc()