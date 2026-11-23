"""Trim the train slice to a story boundary, train byte-level BPEs, write uint16 token arrays."""
import json, os
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer

D = Path(__file__).resolve().parent.parent / "data" / "text"
EOT = "<|endoftext|>"
VOCABS = [512, 2048, 8192]
os.environ.setdefault("RAYON_NUM_THREADS", "4")

raw, train = D / "train_slice_raw.txt", D / "train_slice.txt"
if raw.exists():
    b = raw.read_bytes()
    cut = b.rfind((EOT + "\n").encode()) + len(EOT) + 1
    train.write_bytes(b[:cut])
    del b
    raw.unlink()
valid = D / "TinyStoriesV2-GPT4-valid.txt"


def encode_file(tok, path, block=20000):
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        while True:
            lines = [l for _, l in zip(range(block), f)]
            if not lines:
                break
            out += [np.array(e.ids, np.uint16) for e in tok.encode_batch(lines)]
    return np.concatenate(out)


meta = {}
for name, path in [("train", train), ("valid", valid)]:
    arr = np.frombuffer(path.read_bytes(), np.uint8).astype(np.uint16)
    np.save(D / f"tok_bytes_{name}.npy", arr)
    meta[f"bytes_{name}"] = dict(V=256, n_tokens=int(len(arr)), n_bytes=int(len(arr)))

for V in VOCABS:
    tpath = D / f"bpe{V}.json"
    if not tpath.exists():
        t = ByteLevelBPETokenizer()
        t.train([str(train)], vocab_size=V, min_frequency=2, special_tokens=[EOT], show_progress=False)
        t.save(str(tpath))
    tok = Tokenizer.from_file(str(tpath))
    sample = valid.read_text(encoding="utf-8")[:2_000_000]
    assert tok.decode(tok.encode(sample).ids, skip_special_tokens=False) == sample, "round trip failed"
    for name, path in [("train", train), ("valid", valid)]:
        arr = encode_file(tok, path)
        assert arr.max() < V
        np.save(D / f"tok_bpe{V}_{name}.npy", arr)
        nb = path.stat().st_size
        meta[f"bpe{V}_{name}"] = dict(V=V, n_tokens=int(len(arr)), n_bytes=nb, bytes_per_token=nb / len(arr))
        print(V, name, len(arr), nb / len(arr), flush=True)

for k, v in meta.items():
    v.setdefault("bytes_per_token", 1.0)
json.dump(meta, open(D / "tokenization_meta.json", "w"), indent=1)
