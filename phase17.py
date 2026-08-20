"""TRACK A: CycleOpPRAM on abcp — counters must be LEARNED. python3 phase17.py <seed>"""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks4 import gen_abcp, gen_abc
from models4 import CycleOpPRAM
from models import count_params
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED)
model = CycleOpPRAM(4, 10)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
for step in range(1, 12001):
    f = step / 12000
    nm = 2 if f < .2 else 3 if f < .4 else 4 if f < .6 else 5
    L = 32 if f < 0.5 else 64
    x, y, _, _ = gen_abcp(32, L, g, nmax=nm)
    loss = F.cross_entropy(model(x).reshape(-1, 10), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 4000 == 0: print(f"[cyc-abcp] {step} loss {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), f"cycleop_abcp_s{SEED}.pt")
for l in model.layers: l.hard = True
model.eval(); res = {}
with torch.no_grad():
    l = model.layers[0]
    h = model.emb(torch.arange(4))
    a = F.softmax(l.alpha(h), -1)
    offs = [F.softmax(l.off[b], -1) for b in range(2)]
    for i, tk in enumerate("abcP"):
        o = a[i].argmax().item()
        d0, d1 = offs[0][o].argmax().item(), offs[1][o].argmax().item()
        q = F.softmax(l.readq(h), -1)[i].argmax().item()
        print(f"[dump] {tk}: op#{o} g={int(l.gbits[o])} rot=({d0},{d1}) q={q}", flush=True)
    for L in (64, 256, 1024, 4096):
        bs = 16 if L <= 256 else 4
        c = t = 0
        for i in range(3):
            x, y, _, _ = gen_abcp(bs, L, torch.Generator().manual_seed(8700 + L + i))
            p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
        res[str(L)] = round(c / t, 4)
out = dict(tag=f"EXP070-CYCLEOP-ABCP-S{SEED}", params=count_params(model), acc=res,
           certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
