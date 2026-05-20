import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()


def check_langsmith():
    print("Checking LangSmith setup...")
    print("=" * 50)

    key     = os.getenv("LANGCHAIN_API_KEY", "")
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "")
    project = os.getenv("LANGCHAIN_PROJECT", "")

    print(f"  LANGCHAIN_TRACING_V2: "
          f"{'✅ ' + tracing if tracing else '❌ not set'}")
    print(f"  LANGCHAIN_API_KEY:    "
          f"{'✅ set (' + key[:8] + '...)' if key else '❌ not set'}")
    print(f"  LANGCHAIN_PROJECT:    "
          f"{'✅ ' + project if project else '❌ not set'}")

    if not key:
        print("\n❌ LANGCHAIN_API_KEY missing.")
        print("   Get it at: https://smith.langchain.com")
        print("   Add to .env: LANGCHAIN_API_KEY=ls__xxx")
        return False

    # Test connection
    try:
        from langsmith import Client
        client = Client(api_key=key)
        projects = list(client.list_projects())
        print(f"\n✅ LangSmith connected!")
        print(f"   Projects found: {len(projects)}")
        return True
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    ok = check_langsmith()
    if ok:
        print("\nLangSmith ready — traces will appear at:")
        print("https://smith.langchain.com")
    else:
        print("\nFix the issues above, then re-run.")