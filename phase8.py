"""Cycle 14: forensic dump + staged hardening on Dyck-2.
Protocol: 12k soft (M21 priors) -> DUMP program table -> stage-1: snap prototypes only
(ST), retrain rest 2k -> DUMP again -> snap all -> cert + eval to 4096."""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks3 import gen_dyck2
from models2 import PRAM, sinkhorn, st_onehot, st_binary
from models import hard_perm, count_params

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
torch.manual_seed(SEED)
model = PRAM(4, 3, k=8, n_proto=12, use_scan=True, tie_vals=True)
for l in model.layers:
    torch.nn.init.constant_(l.rho.bias, 2.0)
    torch.nn.init.constant_(l.gate.bias, 2.0)   # M23b: writes default ON
    l.use_flat = False                          # M23a: decode bottleneck
    l.hard_gates = True                         # M24: gates ST-binary from step 0
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
TOK = ["(", ")", "[", "]"]

def dump(tag):
    l = model.layers[0]
    with torch.no_grad():
        h = model.emb(torch.arange(4))
        P = sinkhorn(l.protos / l.tau, l.sink_iters); Ph = hard_perm(P)
        a = F.softmax(l.alpha(h), -1); rho = torch.sigmoid(l.rho(h)).squeeze(-1)
        gg = torch.sigmoid(l.gate(h)).squeeze(-1)
        w = F.softmax(l.sel(h), -1); q = F.softmax(l.readq(h), -1)
        beta = F.softmax(l.beta(h), -1)
        print(f"--- PROGRAM DUMP [{tag}] ---", flush=True)
        for i in range(4):
            m = a[i].argmax().item()
            perm = tuple(Ph[m, r].argmax().item() for r in range(8))  # new[r]=old[perm[r]]
            print(f"tok {TOK[i]}: rho={rho[i]:.2f} g={gg[i]:.2f} "
                  f"proto#{m}(p={a[i].max():.2f}) perm(new<-old)={perm} "
                  f"w={w[i].argmax().item()}(p={w[i].max():.2f}) "
                  f"beta={beta[i].argmax().item()}(p={beta[i].max():.2f}) "
                  f"q={q[i].argmax().item()}(p={q[i].max():.2f})", flush=True)

def train(steps, tag, hard_P=False, lr=3e-3):
    for gr in opt.param_groups: gr["lr"] = lr
    for step in range(1, steps + 1):
        f = step / steps
        md = (1 if f < .15 else 2 if f < .3 else 3 if f < .5 else 4 if f < .7 else 6) \
             if tag == "soft" else 6
        L = 32 if (tag == "soft" and f < 0.5) else 64
        x, y, _, _ = gen_dyck2(32, L, g, max_depth=md)
        loss = F.cross_entropy(model(x).reshape(-1, 3), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % max(1, steps // 4) == 0:
            print(f"[{tag}] {step} loss {loss.item():.5f}", flush=True)

# stage-0 soft
train(12000, "soft")
torch.save(model.state_dict(), f"pram_dyck_soft_s{SEED}.pt")
dump("after soft")

# stage-1: snap prototypes only (ST), others stay soft
import models2, types
def _ops_hardP(self, h):
    B, L, _ = h.shape
    P = sinkhorn(self.protos / self.tau, self.sink_iters)
    P = hard_perm(P) + P - P.detach()                    # ONLY P hard
    a = F.softmax(self.alpha(h), -1)
    w = F.softmax(self.sel(h), -1)
    q = F.softmax(self.readq(h), -1)
    gg = torch.sigmoid(self.gate(h))
    rho = torch.sigmoid(self.rho(h))
    R = torch.einsum("blm,mij->blij", a, P)
    I0 = torch.eye(self.k, device=h.device).expand_as(R)
    R = (1 - rho.unsqueeze(-1)) * I0 + rho.unsqueeze(-1) * R
    beta = F.softmax(self.beta(h), -1)
    v = beta @ self.vcode
    gw = gg.unsqueeze(-1) * w.unsqueeze(-1)
    I = torch.eye(self.k, device=h.device)
    A = torch.matmul(R, I - gw * w.unsqueeze(-2))
    b = torch.matmul(R, gw * v.unsqueeze(-2))
    return A, b, q
orig_ops = models2.PRAMLayer._ops
models2.PRAMLayer._ops = _ops_hardP
train(2000, "hardP", lr=1e-3)
dump("after stage-1 (P snapped)")
models2.PRAMLayer._ops = orig_ops

# stage-2: snap everything, certify, stress
for l in model.layers: l.hard = True
model.eval(); res = {}
with torch.no_grad():
    c = t = 0
    for _ in range(3):
        x, y, _, _ = gen_dyck2(16, 64, g)
        p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    cert = (c == t); res["cert64"] = round(c / t, 4)
    for L in (256, 1024, 4096):
        c = t = 0
        for _ in range(3):
            x, y, _, _ = gen_dyck2(4, L, g)
            p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
        res[str(L)] = round(c / t, 4)
out = dict(tag=f"EXP048-DYCK-M24-SEED{SEED}", certified=bool(cert), acc=res,
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
           wall_s=round(time.time() - t0, 1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
if cert: torch.save(model.state_dict(), f"pram_dyck_cert_s{SEED}.pt")
