"""
Deploy VNet ASR to Hugging Face Model Hub (free)
"""
import os, sys
from huggingface_hub import HfApi, create_repo, upload_file

TOKEN = "hf_xxxxx"  # 替换为你的 Hugging Face Token（https://huggingface.co/settings/tokens）
REPO_ID = "yaweiyuan/vnet-asr1"
SPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_space")

api = HfApi(token=TOKEN)

print("=" * 60)
print("[1/2] Creating Model Hub repo...")
create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
print(f"  Repo: https://huggingface.co/{REPO_ID}")

print("[2/2] Uploading files...")
files = [
    "app.py", "model.py", "requirements.txt", "README.md",
    ".gitattributes", "units.txt", "cmvn_stats.pt", "best.pt",
]
for fname in files:
    local_path = os.path.join(SPACE_DIR, fname)
    if not os.path.exists(local_path):
        print(f"  SKIP (not found): {fname}")
        continue
    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"  Uploading {fname} ({size_mb:.1f}MB) ...", end=" ", flush=True)
    try:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type="model",
            token=TOKEN,
        )
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)

print()
print("=" * 60)
print("All files uploaded to Model Hub!")
print(f"  Repo: https://huggingface.co/{REPO_ID}")
print()
print("NOTE: Gradio Spaces now require HF Pro ($9/month).")
print("Alternative deployment methods are presented below.")
