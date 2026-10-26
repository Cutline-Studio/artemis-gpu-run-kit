"""GPU path for the Artemis dense model on the web+WordNet+Artemis mix (BPE 24k).
bf16 autocast; attention = F.scaled_dot_product_attention (flash kernel on Ampere when eligible);
torch.compile (falls back to eager on error); packed Stage-1 windows (random contiguous windows over the
concatenated token stream, <|eos|> between docs, no padding); AdamW + LR warmup + cosine (to 10%);
held-out bits-per-byte on web_val and artemis_val; early stopping (patience); wall-clock watchdog
(--deadline-epoch) that ends training, saves and writes summary/DONE. Stage 2: SFT with assistant-only loss.
Stage-1 steps = min(--stage1-steps, steps that fit --stage1-budget-s at the throughput measured over steps 20-70,
 --max-epochs)."""
import argparse, json, math, os, sys, time, random, hashlib
import numpy as np, torch, torch.nn.functional as F
from artemis_dense import train as T
from artemis_dense.engines import transformer as tx
from artemis_dense.tok import get_codec, n_bytes
IGN = -100
EVAL_PROMPTS = ["define serendipity", "My name is Latoya. What can you do?", "who must approve a GPU rental?", "what is 12*13?", "fix it"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--size", default="medium"); ap.add_argument("--tok", required=True)
    ap.add_argument("--data", default="."); ap.add_argument("--context-len", type=int, default=1024); ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--n-layers", type=int, default=0); ap.add_argument("--seed", type=int, default=567885)
    ap.add_argument("--stage1-steps", type=int, default=100000); ap.add_argument("--stage1-budget-s", type=float, default=7200)
    ap.add_argument("--max-epochs", type=float, default=5.0); ap.add_argument("--stage2-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--sft-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4); ap.add_argument("--sft-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=6); ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--deadline-epoch", type=float, default=0); ap.add_argument("--out", default="out")
    ap.add_argument("--web-val-windows", type=int, default=200)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else dict(device_type="cpu", enabled=False)
    torch.manual_seed(a.seed); random.seed(a.seed); g = torch.Generator().manual_seed(a.seed)
    if dev == "cuda": torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    rd = os.path.join(a.out, a.run); os.makedirs(rd, exist_ok=True)
    logf = open(os.path.join(rd, "train_log.jsonl"), "a")
    def log(r):
        r["t"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); s = json.dumps(r); logf.write(s + "\n"); logf.flush(); print(s, flush=True)
    late = lambda: bool(a.deadline_epoch) and time.time() > a.deadline_epoch
    codec = get_codec(kind="bpe", path=a.tok); V = codec.vocab_size
    over = dict(context_len=a.context_len, dropout=a.dropout, tokenizer="bpe", tokenizer_path=os.path.abspath(a.tok), vocab_size=V)
    if a.n_layers: over["n_layers"] = a.n_layers
    model = tx.build(a.size, seed=a.seed, **over).to(dev)
    ld = lambda f: torch.from_numpy(np.fromfile(os.path.join(a.data, f), dtype=np.uint16).astype(np.int32)).to(dev)
    tr, wv, av = ld("train.bin"), ld("web_val.bin"), ld("artemis_val.bin")
    wv_bytes = n_bytes(codec.decode(wv[:a.web_val_windows * a.context_len + 1].tolist()))
    av_bytes = n_bytes(open(os.path.join(a.data, "mix", "artemis_val.txt"), encoding="utf-8").read())
    art_train_text = open(os.path.join(a.data, "mix", "artemis_train.txt"), encoding="utf-8").read()
    recs = [T.encode_masked(d, codec) for d in T.sft_dialogues(art_train_text)]
    recs = [r for r in recs if any(y != IGN for y in r[1][:a.context_len])]
    sft_va = [r for i, r in enumerate(recs) if i % 10 == 9]; sft_tr = [r for i, r in enumerate(recs) if i % 10 != 9]
    fwd = model
    if a.compile and dev == "cuda":
        try:
            t0 = time.time(); fwd = torch.compile(model)
            with torch.autocast(**amp): fwd(torch.zeros(a.batch, a.context_len, dtype=torch.long, device=dev))[0].float().sum().backward()
            model.zero_grad(set_to_none=True); log({"event": "compile_ok", "s": round(time.time() - t0, 1)})
        except Exception as e:
            fwd = model; log({"event": "compile_failed", "err": str(e)[:300]})

    @torch.no_grad()
    def bpb(ids, nbytes, maxw=None):
        model.eval(); Tn = model.cfg.context_len; nll = 0.0; nt = 0; w = 0
        for s in range(0, len(ids) - 1, Tn):
            c = ids[s:s + Tn + 1].long()
            if len(c) < 2: break
            with torch.autocast(**amp): lg, _ = model(c[None, :-1])
            nll += F.cross_entropy(lg[0].float(), c[1:], reduction="sum").item(); nt += len(c) - 1; w += 1
            if maxw and w >= maxw: break
        return round(nll / math.log(2) / nbytes, 4)
    @torch.no_grad()
    def sftv():
        model.eval(); tot = 0.0; n = 0
        for x, y in sft_va:
            x = torch.tensor([x[:a.context_len]], device=dev); y = torch.tensor([y[:a.context_len]], device=dev)
            with torch.autocast(**amp): lg, _ = model(x)
            tot += F.cross_entropy(lg[0].float(), y[0], ignore_index=IGN, reduction="sum").item(); n += int((y != IGN).sum())
        return round(tot / max(n, 1), 4)
    ev = lambda: {"web_val_bpb": bpb(wv, wv_bytes, a.web_val_windows), "artemis_val_bpb": bpb(av, av_bytes), "sft_val_nll": sftv()}
    def samples():
        model.to("cpu"); out = {p: T.sample(model, codec, p) for p in EVAL_PROMPTS}; model.to(dev); return out
    def save(name, info=None):
        model.to("cpu"); model.trained_info = info; tx.save(model, os.path.join(rd, name)); model.to(dev)

    summ = {"run": a.run, "command": "python gpu_train.py " + " ".join(sys.argv[1:]), "args": vars(a), "device": dev,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "torch": torch.__version__, "python": sys.version.split()[0],
            "model_card": tx.model_card(model), "params_total": model.num_params(),
            "params_non_embedding": model.num_params() - model.embed.weight.numel(),
            "train_tokens": int(len(tr)), "sft_train": len(sft_tr), "sft_val": len(sft_va), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    log({"event": "start", **summ}); summ["before"] = ev(); log({"event": "before", **summ["before"]})
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=(dev == "cuda"))
    B, Tn = a.batch, a.context_len; total = a.stage1_steps; best = {"bpb": 1e9, "step": 0}; bad = 0; seen = 0; stop = "steps"
    t0 = time.time(); ar = torch.arange(Tn, device=dev); step = 0; tm = None
    while step < total:
        if late(): stop = "deadline"; break
        model.train()
        ix = torch.randint(0, len(tr) - Tn - 1, (B,), generator=g).to(dev)
        x = tr[ix[:, None] + ar].long(); y = tr[ix[:, None] + ar + 1].long()
        for pg in opt.param_groups: pg["lr"] = T.lr_at(step, total, a.lr, a.warmup)
        with torch.autocast(**amp): lg, _ = fwd(x)
        loss = F.cross_entropy(lg.float().reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1; seen += B * Tn
        if step == 20: torch.cuda.synchronize() if dev == "cuda" else None; tm = time.time()
        if step == 70:
            if dev == "cuda": torch.cuda.synchronize()
            sps = 50 / (time.time() - tm); fit = int(sps * (a.stage1_budget_s - (time.time() - t0)) * 0.93) + 70
            ep = int(a.max_epochs * len(tr) / (B * Tn)); total = max(100, min(a.stage1_steps, fit, ep))
            log({"event": "plan", "steps_per_s": round(sps, 2), "tokens_per_s": round(sps * B * Tn), "stage1_steps": total,
                 "planned_tokens": total * B * Tn, "planned_epochs": round(total * B * Tn / len(tr), 2),
                 "train_flops_6ND": 6 * summ["params_total"] * total * B * Tn})
        r = {"stage": 1, "step": step, "loss": round(loss.item(), 4), "tokens": seen, "lr": round(opt.param_groups[0]["lr"], 6), "elapsed_s": round(time.time() - t0, 1)} if step % 100 == 0 or step == total or step % a.eval_every == 0 else None
        if step % a.eval_every == 0 or step == total:
            r["val"] = ev()
            if r["val"]["web_val_bpb"] < best["bpb"]: best = {"bpb": r["val"]["web_val_bpb"], "step": step}; bad = 0; save("stage1_best.pt")
            else: bad += 1
        if r: log(r)
        if bad >= a.patience: stop = "early_stop"; break
    summ["stage1"] = {"steps_run": step, "planned_steps": total, "stop": stop, "tokens": seen, "wall_s": round(time.time() - t0, 1),
                      "epochs": round(seen / len(tr), 3), "best": best}
    if best["step"] and best["step"] != step: model.load_state_dict(tx.load(os.path.join(rd, "stage1_best.pt")).state_dict()); model.to(dev)
    elif not best["step"]: save("stage1_best.pt")
    summ["after_stage1"] = ev(); summ["samples_after_stage1"] = samples()
    log({"event": "stage1_done", **summ["stage1"], "after": summ["after_stage1"], "samples": summ["samples_after_stage1"]})
    opt = torch.optim.AdamW(model.parameters(), lr=a.sft_lr, betas=(0.9, 0.95), weight_decay=0.0, fused=(dev == "cuda"))
    t0 = time.time(); seen2 = 0; s2 = 0; bestv = 1e9
    for s2 in range(a.stage2_steps):
        if late(): break
        model.train(); bt = [sft_tr[i] for i in torch.randint(0, len(sft_tr), (a.sft_batch,), generator=g).tolist()]
        L = min(Tn, max(len(x) for x, _ in bt)); X = torch.zeros(len(bt), L, dtype=torch.long); Y = torch.full((len(bt), L), IGN, dtype=torch.long)
        for j, (x, y) in enumerate(bt):
            x, y = x[:L], y[:L]; X[j, :len(x)] = torch.tensor(x); Y[j, :len(y)] = torch.tensor(y)
        X, Y = X.to(dev), Y.to(dev)
        for pg in opt.param_groups: pg["lr"] = T.lr_at(s2, a.stage2_steps, a.sft_lr, 30)
        with torch.autocast(**amp): lg, _ = model(X)
        loss = F.cross_entropy(lg.float().reshape(-1, V), Y.reshape(-1), ignore_index=IGN)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        seen2 += int((Y != IGN).sum())
        if (s2 + 1) % 100 == 0:
            v = sftv(); log({"stage": 2, "step": s2 + 1, "loss": round(loss.item(), 4), "assistant_tokens": seen2, "sft_val_nll": v})
            if v < bestv: bestv = v; save("stage2_best.pt")
    summ["stage2"] = {"steps_run": s2 + 1, "assistant_tokens": seen2, "wall_s": round(time.time() - t0, 1), "best_sft_val_nll": bestv}
    summ["after_stage2"] = ev(); summ["samples_after_stage2"] = samples(); summ["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save("stage2_final.pt", {"run": a.run, "web_val_bpb": summ["after_stage2"]["web_val_bpb"]})
    json.dump(summ, open(os.path.join(rd, "summary.json"), "w"), indent=2)
    log({"event": "done", "stage1": summ["stage1"], "stage2": summ["stage2"], "after_stage2": summ["after_stage2"], "samples": summ["samples_after_stage2"]})
    open(os.path.join(rd, "DONE"), "w").write("ok")

if __name__ == "__main__":
    main()
