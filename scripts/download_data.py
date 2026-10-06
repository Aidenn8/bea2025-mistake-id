"""Download the BEA 2025 Shared Task data (MRBench v3) from the organisers' repository.

The files are fetched from a fixed commit and checked against known SHA-256 hashes,
so every run of this project uses byte-identical data. The data is released by the
organisers under CC BY-SA 4.0 and is not redistributed in this repository.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mistake_id.data import DATA_DIR, FILES  # noqa: E402

COMMIT = "bbef521ddb875f2cc8a5ee798f4066965a7cfd8a"
BASE = f"https://raw.githubusercontent.com/kaushal0494/UnifyingAITutorEvaluation/{COMMIT}/BEA_Shared_Task_2025_Datasets"
SHA256 = {
    "mrbench_v3_devset.json": "a395feab801de50d44b73e1e9b88f84824a1f1d92e15ae86842d90c2b6dbf33c",
    "mrbench_v3_testset.json": "5a73434fd9e1dfc73026e146d3709f95f2483a51a5e5b354bc96d981448d92cd",
}


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES.values():
        path = DATA_DIR / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(f"{BASE}/{name}", path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SHA256[name]:
            sys.exit(f"{name}: SHA-256 {digest} does not match the expected {SHA256[name]}")
        print(f"{name}: ok ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
