"""P3 protocol: soft-flat -> ramped difficulty -> hard-ST fine-tune -> hard eval."""
import json, resource, time, torch, torch.nn.functional as F
import tasks2 as T2
from models2 import PRAM
from models import count_params

torch.manual_seed(0)
model = PRAM(50, 8, use_scan=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(1)
t0 = time.time()

def train(steps, pg0, pg1, L0, L1, hard, tag, lr=None):
    if lr:
        for gr in opt.param_groups: gr["lr"] = lr
    for lyr in model.layers: lyr.hard = hard
    for step in range(1, steps + 1):
        f = step / steps
        pg = pg0 + (pg1 - pg0) * min(1.0, f / 0.8)
        L = L0 if f < 0.5 else L1
        x, y, ya, _, _ = T2.gen_track5(32, L, g, pg=pg, ps=0.3, aux=True)
        logits, auxl = model(x, with_aux=True)
        lq = F.cross_entropy(logits.reshape(-1, 8), y.reshape(-1), ignore_index=-100)
        la = F.cross_entropy(auxl.reshape(-1, 8), ya.reshape(-1))
        (lq + la).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % max(1, steps // 8) == 0:
            print(f"[{tag}] step {step} pg={pg:.2f} L={L} q {lq.item():.4f} "
                  f"aux {la.item():.4f}", flush=True)

train(9000, 0.10, 0.50, 32, 64, hard=False, tag="A-soft")
train(3000, 0.50, 0.50, 64, 64, hard=True, tag="B-hardST", lr=1e-3)

for lyr in model.layers: lyr.hard = True
model.eval()
res = {}
with torch.no_grad():
    for far in (False, True):
        for L in (64, 256, 1024):
            c = t = 0
            for _ in range(4):
                x, y, _, _ = T2.gen_track5(16, L, g, far=far)
                pred = model(x).argmax(-1); m = y != -100
                c += (pred[m] == y[m]).sum().item(); t += m.sum().item()
            res[("far" if far else "std") + str(L)] = round(c / t, 4)
out = dict(model="pram", task="track5", tag="EXP034-P3", params=count_params(model),
           acc=res, peak_ram_mb=round(
               resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
           wall_s=round(time.time() - t0, 1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
torch.save(model.state_dict(), "pram_p3.pt")
