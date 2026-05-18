import sys

print("Checking fine-tuning dependencies...")
print("=" * 50)

checks = [
    ("sentence_transformers", "sentence-transformers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("datasets", "datasets"),
    ("torch", "torch"),
]

all_good = True
for module, package in checks:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"  ✅ {package}: {version}")
    except ImportError:
        print(f"  ❌ {package}: NOT INSTALLED")
        print(f"     Fix: pip install {package}")
        all_good = False

print()
if all_good:
    print("All dependencies ready!")
    print("Fine-tuning can start.")
else:
    print("Install missing packages above first.")

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"\nGPU detected: {gpu}")
        print(f"GPU memory:   {mem // 1024**3} GB")
        print("Training will use GPU (faster)")
    else:
        print("\nNo GPU detected — using CPU")
        print("Training will work but take longer (~30 min)")
        print("Alternative: use Google Colab for free GPU")
except Exception as e:
    print(f"\nCould not check GPU: {e}")