"""Check every file listed in MANIFEST.sha256 against its SHA-256, and that no listed file is missing.

    python verify/check_manifest.py            # check
    python verify/check_manifest.py --write    # rewrite MANIFEST.sha256 from the files tracked by git
"""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(ROOT, "MANIFEST.sha256")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if "--write" in sys.argv:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    files = sorted(f for f in files if f and f != "MANIFEST.sha256")
    with open(MAN, "w") as out:
        for f in files:
            out.write(f"{sha(os.path.join(ROOT, f))}  {f}\n")
    print(f"MANIFEST.sha256: {len(files)} files")
    sys.exit(0)
bad = n = 0
for line in open(MAN):
    h, f = line.rstrip("\n").split("  ", 1)
    n += 1
    p = os.path.join(ROOT, f)
    if not os.path.isfile(p) or sha(p) != h:
        bad += 1
        print(f"MISMATCH {f}")
print(f"check_manifest: {n} files listed, {n - bad} match, {bad} differ or are missing")
sys.exit(1 if bad else 0)
