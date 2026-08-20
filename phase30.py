"""KRISA2 runner. python3 phase30.py <mode> ; mode in {control, modal}"""
import json, resource, sys, time, torch, torch.nn.functional as F
from models7 import KRISA2
MODE = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
torch.manual_seed(SEED)
if MODE == "control":
    from tasks3 import gen_dyck2p as GEN; VIN, VOUT = 7, 3
    MD = [0] * 7                                   # modes inert
    def curr(f): return dict(max_depth=1 if f<.15 else 2 if f<.3 else 3 if f<.5 else
                             4 if f<.7 else 6)
else:
    from tasks8 import gen_modal as GEN; VIN, VOUT = 6, 3
    MD = [1, 2, 0, 0, 0, 0]                        # selected mode program (EXP087)
    def curr(f): return dict(max_depth=1 if f<.15 else 2 if f<.3 else 3 if f<.5 else
                             4 if f<.7 else 5)
model = KRISA2(VIN, VOUT)
with torch.no_grad():
    model.mdisp.fill_(-10.0)
    for i, instr in enumerate(MD): model.mdisp[i, instr] = 10.0
model.mdisp.requires_grad_(False)
params = [p for n, p in model.named_parameters() if n != "mdisp"]
opt = torch.optim.AdamW(params, lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
for step in range(1, 12001):
    f = step / 12000
    L = 32 if f < 0.5 else 64
    x, y, _, _ = GEN(32, L, g, **curr(f))
    loss = F.cross_entropy(model(x).reshape(-1, VOUT), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); opt.zero_grad()
    if step % 6000 == 0: print(f"[{MODE}] {step} loss {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), f"krisa2_{MODE}_s{SEED}.pt")
model.eval(); model.hard = True; res = {}
with torch.no_grad():
    for L in (64, 256, 1024, 4096):
        c = t = 0
        for i in range(3):
            x, y, _, _ = GEN(4 if L > 256 else 16, L,
                             torch.Generator().manual_seed(9870 + L + i))
            p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
        res[str(L)] = round(c / t, 4)
out = dict(tag=f"EXP096-KRISA2-{MODE.upper()}-S{SEED}", acc=res,
           certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
