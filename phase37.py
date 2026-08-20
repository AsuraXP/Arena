"""Polyglot daemon attempt: LMISA(13,13) on gen_poly, per-family label-free gate.
python3 phase37.py <seed>"""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks13 import gen_poly, FAMS
SEED = int(sys.argv[1])
torch.manual_seed(SEED)
exec(open("phase33.py").read().split("model = LMISA()")[0])
model = LMISA(vin=13, vout=13)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 200)
t0 = time.time()
for step in range(1, 14001):
    L = 32 if step < 7000 else 64
    x, y, _ = gen_poly(32, L, g)
    loss = F.cross_entropy(model(x).reshape(-1, 13), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
torch.save(model.state_dict(), f"poly_lmisa_s{SEED}.pt")
model.eval()
def ce(f, hard, L, n=3):
    model.hard = hard
    tot = cnt = 0.0
    with torch.no_grad():
        for i in range(n):
            x, y, _ = FAMS[f](8 if L > 256 else 16, L,
                              torch.Generator().manual_seed(9720 + L + i))
            lp = F.log_softmax(model(x), -1)
            tot += -lp.gather(-1, y.unsqueeze(-1)).sum().item(); cnt += y.numel()
    return tot / cnt
gates, stats = {}, {}
for f in FAMS:
    h64, s64 = ce(f, True, 64), ce(f, False, 64)
    h1k, s1k = ce(f, True, 1024), ce(f, False, 1024)
    gates[f] = bool((h64 - s64 < 0.05) and (h1k <= s1k + 0.02))
    stats[f] = dict(h64=round(h64,4), s64=round(s64,4), h1k=round(h1k,4),
                    s1k=round(s1k,4))
verify = {}
if all(gates.values()):
    model.hard = True
    with torch.no_grad():
        for f in FAMS:
            for L in (64, 1024, 4096):
                tot = orc = cnt = 0.0
                for i in range(3):
                    x, y, o = FAMS[f](4 if L > 256 else 16, L,
                                      torch.Generator().manual_seed(9710 + L + i))
                    lp = F.log_softmax(model(x), -1)
                    tot += -lp.gather(-1, y.unsqueeze(-1)).sum().item()
                    orc += o.sum().item(); cnt += y.numel()
                verify[f + str(L)] = round((tot - orc) / cnt, 5)
out = dict(tag=f"EXP115-POLYDAEMON-S{SEED}", gates=gates, stats=stats,
           oracle_verify=verify, wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
