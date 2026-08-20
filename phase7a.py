"""Milestone A: reliability recipe v2 on track5 (orthogonal vcode init, n_proto=16,
cosine LR). One seed per invocation: python3 phase7a.py <seed>"""
import json, resource, sys, time, math, torch, torch.nn.functional as F
import tasks2 as T2
from models2 import PRAM

SEED = int(sys.argv[1])
torch.manual_seed(SEED)
model = PRAM(50, 8, n_proto=16, use_scan=True, tie_vals=True)
for l in model.layers: torch.nn.init.orthogonal_(l.vcode, gain=2.0)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / 9000))) * 0.9 + 0.1)
g = torch.Generator().manual_seed(SEED + 100)
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
    opt.step(); opt.zero_grad(); sched.step()

for l in model.layers: l.hard = True
model.eval(); res = {}
with torch.no_grad():
    c = t = 0
    for _ in range(3):
        x, y, _, _ = T2.gen_track5(16, 64, g)
        p = model(x).argmax(-1); mk = y != -100
        c += (p[mk] == y[mk]).sum().item(); t += mk.sum().item()
    cert = (c == t); res["cert64"] = round(c / t, 4)
    if cert:
        for far in (False, True):
            c = t = 0
            for _ in range(3):
                x, y, _, _ = T2.gen_track5(4, 4096, g, far=far)
                p = model(x).argmax(-1); mk = y != -100
                c += (p[mk] == y[mk]).sum().item(); t += mk.sum().item()
            res[("far" if far else "std") + "4096"] = round(c / t, 4)
out = dict(tag=f"EXP040-RELV2-SEED{SEED}", certified=bool(cert), acc=res,
           wall_s=round(time.time() - t0, 1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
