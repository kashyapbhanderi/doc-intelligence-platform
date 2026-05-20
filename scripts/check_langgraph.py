import sys

print("Checking LangGraph dependencies...")
print("=" * 50)

packages = [
    ("langgraph", "langgraph"),
    ("langchain", "langchain"),
    ("langchain_openai", "langchain-openai"),
    ("langsmith", "langsmith"),
]

all_ok = True
for module, package in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "ok")
        print(f"  ✅ {package}: {version}")
    except ImportError:
        print(f"  ❌ {package}: NOT INSTALLED")
        print(f"     Fix: pip install {package}")
        all_ok = False

print()
if all_ok:
    print("All LangGraph dependencies ready!")
else:
    print("Install missing packages above.")