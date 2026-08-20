"""FBISA on wwr. python3 phase32.py <seed>"""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks7 import gen_wwr
from models8 import FBISA
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED)
model = FBISA(3, 4)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
for step in range(1, 12001):
    f = step / 12000
    nm = 2 if f < .2 else 4 if f < .4 else 6 if f < .6 else 8
    L = 32 if f < 0.5 else 64
    x, y, _, _ = gen_wwr(32, L, g, nmax=nm)
    loss = F.cross_entropy(model(x).reshape(-1, 4), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 6000 == 0: print(f"[fbisa-s{SEED}] {step} loss {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), f"fbisa_wwr_s{SEED}.pt")
model.eval(); model.hard = True; res = {}
with torch.no_grad():
    for L in (64, 256, 1024, 4096):
        c = t = 0
        for i in range(3):
            x, y, _, _ = gen_wwr(4 if L > 256 else 16, L,
                                 torch.Generator().manual_seed(9810 + L + i))
            p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
        res[str(L)] = round(c / t, 4)
out = dict(tag=f"EXP102-FBISA-WWR-S{SEED}", acc=res,
           certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
