"""Harness for PRAM on fused track5 benchmark. Query-position accuracy only."""
import argparse, json, resource, time, torch, torch.nn.functional as F
from tasks2 import TASKS2
from models2 import PRAM
from models import SSRModel, TinyTransformer, count_params

def evaluate(model, task, lengths, batches=4, batch=32, seed=999):
    g = torch.Generator().manual_seed(seed)
    model.eval(); accs = {}
    with torch.no_grad():
        for L in lengths:
            correct = total = 0
            for _ in range(batches):
                x, y, _, _ = TASKS2[task](batch, L, g)
                pred = model(x).argmax(-1)
                m = y != -100
                correct += (pred[m] == y[m]).sum().item(); total += m.sum().item()
            accs[L] = correct / max(1, total)
    model.train(); return accs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["pram", "ssr", "transformer"], required=True)
    p.add_argument("--task", default="track5")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--train_len", type=int, default=64)
    p.add_argument("--eval_lens", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--eval_far", action="store_true")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--d_slot", type=int, default=16)
    p.add_argument("--n_proto", type=int, default=12)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--anneal", type=float, nargs=2, default=None)
    p.add_argument("--hard", action="store_true")
    p.add_argument("--hard_after", type=float, default=None)
    p.add_argument("--hard_eval", action="store_true")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--mix", action="store_true")
    p.add_argument("--ramp", action="store_true")
    p.add_argument("--aux", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    torch.manual_seed(a.seed)

    _, _, vin, vout = TASKS2[a.task](1, 8)
    if a.model == "pram":
        model = PRAM(vin, vout, a.d_model, a.k, a.d_slot, 1, a.tau, 5, a.n_proto,
                     hard=a.hard, use_scan=a.scan)
    elif a.model == "ssr":
        model = SSRModel(vin, vout, a.d_model, a.k, a.d_slot, 1, 0.15, 5, hard=True)
    else:
        model = TinyTransformer(vin, vout, a.d_model, 2, 2,
                                max_len=max(a.eval_lens) + 8)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(a.seed + 1)

    t_start, losses = time.time(), []
    for step in range(1, a.steps + 1):
        L = a.train_len
        if a.curriculum:
            frac = step / a.steps
            L = max(16, a.train_len // 4) if frac < 0.3 else \
                (max(32, a.train_len // 2) if frac < 0.6 else a.train_len)
        if a.hard_after is not None and a.model == "pram" and \
           step == int(a.hard_after * a.steps):
            for lyr in model.layers: lyr.hard = True
            print(f"[step {step}] HARD-ST fine-tune ON", flush=True)
        if a.anneal and a.model == "pram":
            ta0, ta1 = a.anneal
            tau = ta0 * (ta1 / ta0) ** ((step - 1) / max(1, a.steps - 1))
            for lyr in model.layers: lyr.tau = tau
        tk = a.task
        if a.aux > 0 and a.model == "pram":
            import tasks2 as T2
            pg = 0.05 + 0.45 * min(1.0, step / (0.5 * a.steps)) if a.ramp else 0.5
            x, y, yaux, _, _ = T2.gen_track5(a.batch, L, g, pg=pg, ps=0.3, aux=True)
            logits, auxl = model(x, with_aux=True)
            loss_q = F.cross_entropy(logits.reshape(-1, vout), y.reshape(-1),
                                     ignore_index=-100)
            loss_a = F.cross_entropy(auxl.reshape(-1, 8), yaux.reshape(-1))
            loss = loss_q + a.aux * loss_a
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss_q.item())
            if step % max(1, a.steps // 10) == 0:
                print(f"step {step:5d} L={L:4d} pg={pg:.2f} q_loss "
                      f"{sum(losses[-50:])/len(losses[-50:]):.4f} aux {loss_a.item():.4f}",
                      flush=True)
            continue
        if a.ramp and a.task == "track5":
            pg = 0.05 + 0.45 * min(1.0, step / (0.6 * a.steps))
            x, y, _, _ = TASKS2["track5"](a.batch, L, g) if False else \
                __import__("tasks2").gen_track5(a.batch, L, g, pg=pg, ps=0.3)
            loss = F.cross_entropy(model(x).reshape(-1, vout), y.reshape(-1),
                                   ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
            if step % max(1, a.steps // 10) == 0:
                print(f"step {step:5d} L={L:4d} pg={pg:.2f} loss "
                      f"{sum(losses[-50:])/len(losses[-50:]):.4f}", flush=True)
            continue
        if a.mix:
            u = torch.rand(1, generator=g).item()
            tk = "recall5" if u < 0.3 else ("shuffle5" if u < 0.6 else a.task)
        x, y, _, _ = TASKS2[tk](a.batch, L, g)
        loss = F.cross_entropy(model(x).reshape(-1, vout), y.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); losses.append(loss.item())
        if step % max(1, a.steps // 10) == 0:
            print(f"step {step:5d} L={L:4d} loss {sum(losses[-50:])/len(losses[-50:]):.4f}",
                  flush=True)

    if a.hard_eval and a.model == "pram":
        for lyr in model.layers: lyr.hard = True
    accs = evaluate(model, a.task, a.eval_lens)
    far = evaluate(model, "track5far", a.eval_lens) if a.eval_far else {}
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result = dict(model=a.model, task=a.task, tag=a.tag, params=count_params(model),
                  steps=a.steps, train_len=a.train_len,
                  final_loss=round(sum(losses[-50:]) / 50, 6),
                  acc={str(k): round(v, 4) for k, v in accs.items()},
                  acc_far={str(k): round(v, 4) for k, v in far.items()},
                  peak_ram_mb=round(peak_mb, 1),
                  wall_s=round(time.time() - t_start, 1),
                  cfg=dict(k=a.k, d_slot=a.d_slot, n_proto=a.n_proto, scan=a.scan,
                           hard=a.hard, anneal=a.anneal, hard_after=a.hard_after))
    print("RESULT " + json.dumps(result), flush=True)
    with open("results.jsonl", "a") as f: f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()
