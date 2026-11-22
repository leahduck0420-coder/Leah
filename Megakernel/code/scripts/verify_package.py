"""Verify the extracted delivery against its SHA256 inventory. No external dependencies."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def verify():
    manifest = json.loads((ROOT / "validation/package_manifest.json").read_text(encoding="utf8"))
    damaged = []
    for name, expected in manifest["sha256"].items():
        p = ROOT / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            damaged.append(name)
    return damaged
if __name__ == "__main__":
    errors = verify()
    print("Package verification:", ", ".join(errors) if errors else "PASS")
    sys.exit(bool(errors))
