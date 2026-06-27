"""
scripts/upload_to_ec2.py
Uploads your processed data and fine-tuned model
to EC2 using SCP/rsync.

Run from local machine:
  python scripts/upload_to_ec2.py 54.162.45.123

What it uploads:
  data/processed/   ← chunked JSON files (fast)
  models/finetuned/ ← fine-tuned embedding model
"""
import sys
import subprocess
import os

if len(sys.argv) < 2:
    print("Usage: python upload_to_ec2.py <EC2_IP>")
    sys.exit(1)

EC2_IP  = sys.argv[1]
KEY     = r"C:\Users\kashy\.ssh\doc-intel-key.pem"
EC2_DIR = "ubuntu@" + EC2_IP + \
          ":~/doc-intelligence-platform/"


def run_scp(local_path: str,
            remote_path: str,
            is_dir: bool = True):
    """Upload file/folder to EC2 via SCP."""
    flag = "-r" if is_dir else ""
    cmd  = (
        f'scp {flag} -i "{KEY}" '
        f'-o StrictHostKeyChecking=no '
        f'"{local_path}" {remote_path}'
    )
    print(f"Uploading {local_path}...")
    result = subprocess.run(
        cmd, shell=True,
        capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✅ Done")
    else:
        print(f"  ❌ Failed: {result.stderr[:100]}")


print(f"Uploading data to EC2: {EC2_IP}")
print("=" * 50)

# Upload processed docs (fast — JSON files)
if os.path.exists("data/processed"):
    run_scp("data/processed",
            EC2_DIR + "data/")
else:
    print("  ⚠️  data/processed not found")

# Upload fine-tuned model (slower — ~90MB)
if os.path.exists("models/finetuned/best"):
    run_scp("models/finetuned/best",
            EC2_DIR + "models/finetuned/")
elif os.path.exists("models/finetuned/final"):
    run_scp("models/finetuned/final",
            EC2_DIR + "models/finetuned/")
else:
    print("  ⚠️  Fine-tuned model not found")

print("\nUpload complete!")
print(f"Now SSH to EC2 and run:")
print(f"  python embeddings/reembed.py")