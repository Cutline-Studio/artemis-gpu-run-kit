"""Local CPU eval: held-out bits-per-byte (artemis_val from the rebuilt corpus split; first N KB of web_val)
and greedy replies to 5 fixed prompts, for each checkpoint given. Usage: eval_local.py OUT.json name=ckpt ..."""
import json, sys, time, torch
sys.path.insert(0, "/workspace/a567-dense")
from artemis_dense import train as T
from artemis_dense.engines import transformer as tx
from artemis_dense.tok import get_codec
torch.set_num_threads(4)
K = "/workspace/gpu-run-kit/mix/"
art = open(K + "artemis_val.txt", encoding="utf-8").read()
web = open(K + "web_val.txt", encoding="utf-8").read()[:60000].replace("\n<|eos|>\n", "\n\n")
P = ["define serendipity", "My name is Latoya. What can you do?", "who must approve a GPU rental?", "what is 12*13?", "fix it"]
res = {}
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1); m = tx.load(path); m.eval(); c = get_codec(m.cfg); t0 = time.time()
    r = {"ckpt": path, "params": m.num_params(), "tokenizer": m.cfg.tokenizer, "context_len": m.cfg.context_len,
         "artemis_val_bpb": round(T.bits_per_byte(m, c, art)["bits_per_byte"], 4),
         "web_val_60KB_bpb": round(T.bits_per_byte(m, c, web)["bits_per_byte"], 4),
         "samples": {p: T.sample(m, c, p, max_new=120) for p in P}}
    r["eval_s"] = round(time.time() - t0, 1); res[name] = r; print(json.dumps({name: r}, indent=1), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
