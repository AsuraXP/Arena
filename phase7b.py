"""Milestone B: emergent stack. PRAM on bounded-depth Dyck-2. python3 phase7b.py <seed>"""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks3 import gen_dyck2
from models2 import PRAM
from models import count_params

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED)
model = PRAM(4, 3, k=8, n_proto=12, use_scan=True, tie_vals=True)
for l in model.layers:                       # M21: invert no-op priors for Dyck
    torch.nn.init.constant_(l.rho.bias, 2.0)
    torch.nn.init.constant_(l.gate.bias, 0.0)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
for step in range(1, 12001):
    f = step / 12000
    md = 1 if f < 0.15 else (2 if f < 0.3 else (3 if f < 0.5 else (4 if f < 0.7 else 6)))
    L = 32 if f < 0.5 else 64
    x, y, _, _ = gen_dyck2(32, L, g, max_depth=md)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 1500 == 0:
        print(f"[dyck] {step} md={md} loss {loss.item():.4f}", flush=True)

# ---- phase B: crispness pressure (M22) ----
def crisp_reg():
    import torch as T
    from models2 import sinkhorn
    reg = 0.0
    for l in model.layers:
        P = sinkhorn(l.protos / l.tau, l.sink_iters)
        reg = reg + (1.0 - P.max(-1).values.mean())
        reg = reg + l._ent
    return reg

# monkeypatch: capture soft decision entropies during forward
import types, torch as T
import torch.nn.functional as FF
from models2 import PRAMLayer
_orig = PRAMLayer._ops
def _ops_ent(self, h):
    A, b, q = _orig(self, h)
    a = FF.softmax(self.alpha(h), -1)
    rho = T.sigmoid(self.rho(h)); gg = T.sigmoid(self.gate(h))
    ent = -(a.clamp_min(1e-9) * a.clamp_min(1e-9).log()).sum(-1).mean()
    for p in (rho, gg):
        ent = ent + -(p.clamp_min(1e-9) * p.clamp_min(1e-9).log()
                      + (1-p).clamp_min(1e-9) * (1-p).clamp_min(1e-9).log()).mean()
    self._ent = ent
    return A, b, q
PRAMLayer._ops = _ops_ent

for step in range(1, 3001):
    lam = 0.3 * step / 3000
    x, y, _, _ = gen_dyck2(32, 64, g, max_depth=6)
    loss = F.cross_entropy(model(x).reshape(-1, 3), y.reshape(-1)) + lam * crisp_reg()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 1000 == 0:
        print(f"[crisp] {step} lam={lam:.2f} loss {loss.item():.4f}", flush=True)

for l in model.layers: l.hard = True
model.eval(); res = {}
with torch.no_grad():
    c = t = 0
    for _ in range(3):
        x, y, _, _ = gen_dyck2(16, 64, g)
        p = model(x).argmax(-1)
        c += (p == y).sum().item(); t += y.numel()
    cert = (c == t); res["cert64"] = round(c / t, 4)
    for L in (256, 1024, 4096):
        c = t = 0
        for _ in range(3):
            x, y, _, _ = gen_dyck2(4, L, g)
            p = model(x).argmax(-1)
            c += (p == y).sum().item(); t += y.numel()
        res[str(L)] = round(c / t, 4)
out = dict(tag=f"EXP045-DYCK-M22-SEED{SEED}", certified=bool(cert),
           params=count_params(model), acc=res,
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
           wall_s=round(time.time() - t0, 1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
if cert: torch.save(model.state_dict(), f"pram_dyck_s{SEED}.pt")
