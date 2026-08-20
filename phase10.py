"""Cycle 15 / H4: stack-probe supervision on Dyck-2. python3 phase10.py <seed>"""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks3 import gen_dyck2p
from models2 import PRAM, sinkhorn
from models import hard_perm

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED)
model = PRAM(7, 3, k=8, n_proto=12, use_scan=True, tie_vals=True)
for l in model.layers:
    torch.nn.init.constant_(l.rho.bias, 2.0)
    torch.nn.init.constant_(l.gate.bias, 2.0)
    l.use_flat = False
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
TOK = ["(", ")", "[", "]", "P0", "P1", "P2"]

def dump(tag):
    l = model.layers[0]
    with torch.no_grad():
        h = model.emb(torch.arange(7))
        P = sinkhorn(l.protos / l.tau, l.sink_iters); Ph = hard_perm(P)
        a = F.softmax(l.alpha(h), -1); rho = torch.sigmoid(l.rho(h)).squeeze(-1)
        gg = torch.sigmoid(l.gate(h)).squeeze(-1)
        w = F.softmax(l.sel(h), -1); q = F.softmax(l.readq(h), -1)
        beta = F.softmax(l.beta(h), -1)
        print(f"--- DUMP [{tag}] ---", flush=True)
        for i in range(7):
            m = a[i].argmax().item()
            perm = tuple(Ph[m, r].argmax().item() for r in range(8))
            print(f"{TOK[i]:>2}: rho={rho[i]:.2f} g={gg[i]:.2f} proto#{m}"
                  f"(p={a[i].max():.2f}) perm={perm} w={w[i].argmax().item()}"
                  f"(p={w[i].max():.2f}) beta={beta[i].argmax().item()} "
                  f"q={q[i].argmax().item()}(p={q[i].max():.2f})", flush=True)

for step in range(1, 12001):
    f = step / 12000
    md = 1 if f < .15 else 2 if f < .3 else 3 if f < .5 else 4 if f < .7 else 6
    L = 32 if f < 0.5 else 64
    x, y, _, _ = gen_dyck2p(32, L, g, max_depth=md)
    loss = F.cross_entropy(model(x).reshape(-1, 3), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 3000 == 0:
        print(f"[soft] {step} md={md} loss {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), f"pram_dyckp_s{SEED}.pt")
dump("after soft")

for l in model.layers: l.hard = True
model.eval(); res = {}
with torch.no_grad():
    c = t = 0
    for i in range(3):
        x, y, _, _ = gen_dyck2p(16, 64, torch.Generator().manual_seed(7100 + i))
        p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    cert = (c == t); res["cert64"] = round(c / t, 4)
    for L in (256, 1024, 4096):
        c = t = 0
        for i in range(3):
            x, y, _, _ = gen_dyck2p(4, L, torch.Generator().manual_seed(9100 + L + i))
            p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
        res[str(L)] = round(c / t, 4)
out = dict(tag=f"EXP051-DYCKP-H4-SEED{SEED}", certified=bool(cert), acc=res,
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
           wall_s=round(time.time() - t0, 1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
