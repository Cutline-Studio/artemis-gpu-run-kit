"""On-instance: tokenize mix parts with the fixed 24k BPE tokenizer -> uint16 .bin files + counts.json.
train.bin = web_train + wordnet + artemis_train x UPSAMPLE (docs separated by <|eos|>)."""
import json, sys, os, numpy as np
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[1]); UP = int(sys.argv[2]); D = "mix"
def enc(path):
    lines = open(path, encoding="utf-8").read().split("\n<|eos|>\n") if "web" in path else [open(path, encoding="utf-8").read()]
    if "wordnet" in path:
        t = lines[0].split("\n"); lines = ["\n".join(t[i:i + 2000]) + "\n" for i in range(0, len(t), 2000)]
    if "artemis" in path:
        t = lines[0]; lines = [t[i:i + 20000] for i in range(0, len(t), 20000)]   # chunk (split at char boundary is fine for counts; join below keeps text identical)
        out = tk.encode(t).ids; return np.array(out, dtype=np.uint16)
    eos = tk.token_to_id("<|eos|>"); ids = []
    for e in tk.encode_batch([l for l in lines if l.strip()]):
        ids.extend(e.ids); ids.append(eos) if "web" in path else None
    return np.array(ids, dtype=np.uint16)
parts = {p: enc(f"{D}/{p}.txt") for p in ["web_train", "web_val", "wordnet", "artemis_train", "artemis_val"]}
train = np.concatenate([parts["web_train"], parts["wordnet"]] + [parts["artemis_train"]] * UP)
train.tofile("train.bin"); parts["web_val"].tofile("web_val.bin"); parts["artemis_val"].tofile("artemis_val.bin")
counts = {k: int(len(v)) for k, v in parts.items()}; counts["artemis_upsample"] = UP
counts["artemis_train_upsampled"] = counts["artemis_train"] * UP; counts["train_total"] = int(len(train))
counts["artemis_share"] = round(counts["artemis_train_upsampled"] / counts["train_total"], 4)
counts["bytes"] = {p: os.path.getsize(f"{D}/{p}.txt") for p in parts}
json.dump(counts, open("token_counts.json", "w"), indent=1); print(json.dumps(counts))
