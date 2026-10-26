"""Split sources into train/val parts for the GPU run (upsampling happens on the instance).
web: every 200th cleaned doc -> val. artemis: train.split_corpus(combined.txt) (same split as CPU run).
wordnet: WordNet 3.1 glosses from the local dictionary.jsonl (gcide entries excluded), train only."""
import json, os, sys, random
SNAP = "/workspace/gpu-snap-20261025"; sys.path.insert(0, SNAP)
from artemis_dense.train import split_corpus
os.makedirs("mix", exist_ok=True); SEP = "\n<|eos|>\n"
docs = open("data/web_clean.txt", encoding="utf-8").read().split("\n\n")
docs = [d for d in docs if d.strip()]
open("mix/web_train.txt", "w").write(SEP.join(d for i, d in enumerate(docs) if i % 200 != 199) + SEP)
open("mix/web_val.txt", "w").write(SEP.join(d for i, d in enumerate(docs) if i % 200 == 199) + SEP)
tr, va = split_corpus(open(f"{SNAP}/artemis_dense/data/corpus/combined.txt", encoding="utf-8").read())
open("mix/artemis_train.txt", "w").write(tr); open("mix/artemis_val.txt", "w").write(va)
wn = []
for l in open(f"{SNAP}/artemis_dense/data/dictionary/dictionary.jsonl"):
    r = json.loads(l)
    if r["source"].startswith("wordnet"):
        s = f"{r['word'].replace('_', ' ')} ({r['pos']}): " + "; ".join(r["definitions"])
        if r.get("examples"): s += ' e.g. "' + '"; "'.join(r["examples"][:2]) + '"'
        wn.append(s)
random.Random(567885).shuffle(wn)
open("mix/wordnet.txt", "w").write("\n".join(wn) + "\n")
for f in sorted(os.listdir("mix")): print(f, os.path.getsize("mix/" + f))
