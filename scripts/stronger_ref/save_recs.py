"""Save Colab poll output (lines 'SHA <i> <hash>' then 'REC <json>') as colab_raw/rec_<i>.json, verifying each SHA-256."""
import hashlib
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "results/curve_forecast/stronger_ref/colab_raw")
lines = [l.rstrip("\n") for l in sys.stdin if l.strip()]
i = 0
while i < len(lines):
    if lines[i].startswith("SHA "):
        _, idx, h = lines[i].split()
        rec = lines[i + 1][4:]
        got = hashlib.sha256(rec.encode()).hexdigest()[:12]
        status = "ok" if got == h else f"MISMATCH {got}"
        if got == h:
            open(os.path.join(OUT, f"rec_{int(idx):03d}.json"), "w").write(rec + "\n")
        print(idx, status, rec[:40])
        i += 2
    else:
        i += 1
