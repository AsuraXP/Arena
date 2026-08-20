"""Generic offset-surgery for CycleOpPRAM on abcp. python3 phase19.py <ckpt>"""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks4 import gen_abcp
from models4 import CycleOpPRAM, _rot_basis
CKPT = sys.argv[1]
torch.manual_seed(0)
model = CycleOpPRAM(4, 10)
model.load_state_dict(torch.load(CKPT))
L0 = model.layers[0]; t0 = time.time()
K, S1, S2 = 12, 6, 6
B1, B2 = _rot_basis(6), _rot_basis(6)

def extract():
    with torch.no_grad():
        h = model.emb(torch.arange(4))
        tab = dict(
            o=[F.softmax(L0.alpha(h), -1)[i].argmax().item() for i in range(4)],
            q=[F.softmax(L0.readq(h), -1)[i].argmax().item() for i in range(4)],
            b=[F.softmax(L0.beta(h), -1)[i].argmax().item() for i in range(4)],
            w=[F.softmax(L0.wlog, -1)[o].argmax().item() for o in range(16)],
            d0=[F.softmax(L0.off[0], -1)[o].argmax().item() for o in range(16)],
            d1=[F.softmax(L0.off[1], -1)[o].argmax().item() for o in range(16)])
    return tab

def perms(tab):
    P = torch.zeros(16, K, K)
    for o in range(16):
        P[o, :6, :6] = B1[tab["d0"][o]]
        P[o, 6:, 6:] = B2[tab["d1"][o]]
    return P

def run(tab, x, grad=False):
    B, L = x.shape
    Ph = perms(tab)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = model.emb(x)
        o = torch.tensor(tab["o"])[x]; Rt = Ph[o]
        gt = L0.gbits[o].unsqueeze(-1)
        wt = F.one_hot(torch.tensor(tab["w"]), K).float()[o]
        qt = F.one_hot(torch.tensor(tab["q"])[x], K).float()
        vt = L0.vcode[torch.tensor(tab["b"])[x]]
        S = L0.S0.expand(B, -1, -1); reads = []
        for t in range(L):
            gw = (gt[:, t] * wt[:, t]).unsqueeze(-1)
            S = torch.matmul(Rt[:, t], S - gw * (wt[:, t].unsqueeze(1) @ S)
                             + gw * vt[:, t].unsqueeze(1))
            reads.append(torch.bmm(qt[:, t].unsqueeze(1), S).squeeze(1))
        r = torch.stack(reads, 1)
        ho = h + L0.out(torch.cat([r, h], -1))
        return model.head(model.norm(ho)) + r @ L0.vcode.t()

VAL = [gen_abcp(24, 64, torch.Generator().manual_seed(8900 + i)) for i in range(3)]
def score(tab):
    c = t = 0
    for x, y, _, _ in VAL:
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    return c / t

tab = extract()
base = score(tab)
print(f"[osurg] base {base:.4f} tab={ {k: v[:4] for k, v in tab.items()} }", flush=True)
improved, rounds, log = True, 0, []
while improved and rounds < 20 and base < 1.0:
    improved = False; rounds += 1
    used = sorted(set(tab["o"]))
    best, bs = None, base
    for key, dom, idxs in (("o", range(16), range(4)), ("q", range(K), range(4)),
                           ("b", range(10), range(4)),
                           ("d0", range(6), used), ("d1", range(6), used),
                           ("w", range(K), used)):
        for i in idxs:
            for val in dom:
                if tab[key][i] == val: continue
                t2 = {kk: list(v) for kk, v in tab.items()}; t2[key][i] = val
                s = score(t2)
                if s > bs: best, bs = (f"{key}[{i}]={val}", t2), s
    if best:
        log.append(best[0]); tab = best[1]; base = bs; improved = True
        print(f"[osurg] r{rounds}: {best[0]} -> {base:.4f}", flush=True)

print(f"[osurg] repairs={log}", flush=True)
cont = [model.emb.weight, L0.out.weight, L0.out.bias, model.head.weight,
        model.head.bias, model.norm.weight, model.norm.bias, L0.vcode, L0.S0]
for pp in model.parameters(): pp.requires_grad_(False)
for pp in cont: pp.requires_grad_(True)
opt = torch.optim.AdamW(cont, lr=3e-3)
g = torch.Generator().manual_seed(91)
for step in range(1, 1501):
    x, y, _, _ = gen_abcp(32, 64, g)
    loss = F.cross_entropy(run(tab, x, grad=True).reshape(-1, 10), y.reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
print(f"[osurg] recalib loss {loss.item():.6f} val {score(tab):.4f}", flush=True)
# second surgery round post-recalib if needed
if score(tab) < 1.0:
    improved = True
    while improved and score(tab) < 1.0:
        improved = False
        used = sorted(set(tab["o"])); bs = score(tab); best = None
        for key, dom, idxs in (("q", range(K), range(4)), ("b", range(10), range(4)),
                               ("d0", range(6), used), ("d1", range(6), used),
                               ("w", range(K), used)):
            for i in idxs:
                for val in dom:
                    if tab[key][i] == val: continue
                    t2 = {kk: list(v) for kk, v in tab.items()}; t2[key][i] = val
                    s = score(t2)
                    if s > bs: best, bs = t2, s
        if best: tab = best; improved = True; print(f"[osurg] post-recalib -> {bs:.4f}", flush=True)

res = {"cert64": round(score(tab), 4)}
for L in (256, 1024, 4096):
    c = t = 0
    for i in range(3):
        x, y, _, _ = gen_abcp(4, L, torch.Generator().manual_seed(9950 + L + i))
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    res[str(L)] = round(c / t, 4)
out = dict(tag="EXP072-CYCLEOP-OFFSET-SURGERY", ckpt=CKPT, repairs=log,
           final_tab={k: v[:6] for k, v in tab.items()}, acc=res,
           certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
