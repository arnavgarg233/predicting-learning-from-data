"""scripts/train.py unchanged, with torch.mps.empty_cache() after every evaluation (MPS memory hygiene).
Usage: same arguments as scripts/train.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: F401,E402  (numpy before torch avoids the duplicate-libomp abort in this env)
import torch  # noqa: E402
import train  # noqa: E402

_eval = train.evaluate


def evaluate(*a, **k):
    out = _eval(*a, **k)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out


train.evaluate = evaluate
if __name__ == "__main__":
    train.main()
