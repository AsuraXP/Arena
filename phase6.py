"""Phase 6: PRAM + tied value codebook (M18). P4 protocol (dense-aux soft, ramp,
snap, NO hard fine-tune). Stress eval to L=4096 std+far with RSS profiling."""
import json, resource, time, torch, torch.nn.functional as F
import tasks2 as T2
from models2 import PRAM
from models import count_params

def rss(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

torch.manual_seed(0)
model = PRAM(50, 8, use_scan=True, tie_vals=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(1)
t0 = time.time()

for step in range(1, 9001):
    f = step / 9000
    pg = 0.10 + 0.40 * min(1.0, f / 0.8)
    L = 32 if f < 0.5 else 64
    x, y, ya, _, _ = T2.gen_track5(32, L, g, pg=pg, ps=0.3, aux=True)
    logits, auxl = model(x, with_aux=True)
    lq = F.cross_entropy(logits.reshape(-1, 8), y.reshape(-1), ignore_index=-100)
    la = F.cross_entropy(auxl.reshape(-1, 8), ya.reshape(-1))
    (lq + la).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 1500 == 0:
        print(f"[A] {step} pg={pg:.2f} q {lq.item():.4f} aux {la.item():.4f} "
              f"rss {rss()}MB", flush=True)
torch.save(model.state_dict(), "pram_m18.pt")
train_rss = rss()

for lyr in model.layers: lyr.hard = True
model.eval()
res, times = {}, {}
with torch.no_grad():
    for far in (False, True):
        for L in (64, 256, 1024, 4096):
            bs = 16 if L <= 1024 else 4
            c = t = 0; te = time.time()
            for _ in range(4):
                x, y, _, _ = T2.gen_track5(bs, L, g, far=far)
                p = model(x).argmax(-1); mk = y != -100
                c += (p[mk] == y[mk]).sum().item(); t += mk.sum().item()
            key = ("far" if far else "std") + str(L)
            res[key] = round(c / t, 4); times[key] = round(time.time() - te, 1)
out = dict(model="pram+M18", task="track5", tag="EXP036-P6-TIED",
           params=count_params(model), acc=res, eval_s=times,
           train_rss_mb=train_rss, peak_rss_mb=rss(),
           wall_s=round(time.time() - t0, 1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
