"""Train/eval harness. Trains a model on a task at short length, evaluates
per-token accuracy at train length and longer lengths (length generalization)."""
import argparse, json, resource, time, torch, torch.nn.functional as F
from tasks import TASKS
from models import SSRModel, TinyTransformer, count_params

def evaluate(model, task, lengths, batches=8, batch=64, seed=999):
    g = torch.Generator().manual_seed(seed)
    model.eval(); accs = {}
    with torch.no_grad():
        for L in lengths:
            correct = total = 0
            for _ in range(batches):
                x, y, _, _ = TASKS[task](batch, L, g)
                pred = model(x).argmax(-1)
                correct += (pred == y).sum().item(); total += y.numel()
            accs[L] = correct / total
    model.train(); return accs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ssr", "transformer"], required=True)
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--train_len", type=int, default=16)
    p.add_argument("--eval_lens", type=int, nargs="+", default=[16, 64, 128])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--d_slot", type=int, default=16)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--sink_iters", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--hard", action="store_true")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--anneal", type=float, nargs=2, default=None, metavar=("T0","T1"))
    p.add_argument("--hard_eval", action="store_true")
    p.add_argument("--hard_after", type=float, default=None)
    p.add_argument("--n_proto", type=int, default=8)
    p.add_argument("--no_write", action="store_true")
    p.add_argument("--crisp", type=float, default=0.0)
    a = p.parse_args()
    torch.manual_seed(a.seed)

    _, _, vin, vout = TASKS[a.task](1, 2)
    if a.model == "ssr":
        model = SSRModel(vin, vout, a.d_model, a.k, a.d_slot, a.n_layers,
                         a.tau, a.sink_iters, hard=a.hard, n_proto=a.n_proto,
                         use_write=not a.no_write)
    else:
        model = TinyTransformer(vin, vout, a.d_model, 2, 2,
                                max_len=max(a.eval_lens) + 8)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(a.seed + 1)

    t0, losses = time.time(), []
    for step in range(1, a.steps + 1):
        L = a.train_len
        if a.curriculum:
            frac = step / a.steps
            L = 4 if frac < 0.3 else (8 if frac < 0.6 else a.train_len)
        if a.hard_after is not None and a.model == "ssr" and step == int(a.hard_after * a.steps):
            for lyr in model.layers: lyr.hard = True
            print(f"[step {step}] switching to HARD-ST fine-tune", flush=True)
        if a.anneal and a.model == "ssr":
            ta0, ta1 = a.anneal
            tau = ta0 * (ta1 / ta0) ** ((step - 1) / max(1, a.steps - 1))
            for lyr in model.layers: lyr.tau = tau
        x, y, _, _ = TASKS[a.task](a.batch, L, g)
        loss = F.cross_entropy(model(x).reshape(-1, vout), y.reshape(-1))
        if a.crisp > 0 and a.model == "ssr":
            lam = a.crisp * step / a.steps
            loss = loss + lam * sum(l.reg for l in model.layers)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % max(1, a.steps // 10) == 0:
            print(f"step {step:5d}  loss {sum(losses[-50:])/len(losses[-50:]):.4f}",
                  flush=True)

    if a.hard_eval and a.model == "ssr":
        for lyr in model.layers: lyr.hard = True
    accs = evaluate(model, a.task, a.eval_lens)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result = dict(model=a.model, task=a.task, tag=a.tag, params=count_params(model),
                  steps=a.steps, train_len=a.train_len,
                  final_loss=sum(losses[-50:]) / 50,
                  acc={str(k): round(v, 4) for k, v in accs.items()},
                  peak_ram_mb=round(peak_mb, 1), wall_s=round(time.time() - t0, 1),
                  cfg=dict(k=a.k, d_slot=a.d_slot, d_model=a.d_model, hard=a.hard,
                           n_layers=a.n_layers, tau=a.tau, sink=a.sink_iters))
    print("RESULT " + json.dumps(result), flush=True)
    with open("results.jsonl", "a") as f: f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()
