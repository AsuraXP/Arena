"""Cycle 20 TRACK A: beam search over 2-edit offset neighborhoods (CycleOp abcp).
python3 phase20.py <ckpt>"""
import itertools, json, resource, sys, time, torch, torch.nn.functional as F
from tasks4 import gen_abcp
from models4 import CycleOpPRAM, _rot_basis

CKPT = sys.argv[1]
torch.manual_seed(0)
model = CycleOpPRAM(4, 10)
model.load_state_dict(torch.load(CKPT))
L0 = model.layers[0]; t0 = time.time()
K = 12; B1 = _rot_basis(6)

def extract():
    with torch.no_grad():
        h = model.emb(torch.arange(4))
        return dict(
            o=[F.softmax(L0.alpha(h), -1)[i].argmax().item() for i in range(4)],
            q=[F.softmax(L0.readq(h), -1)[i].argmax().item() for i in range(4)],
            b=[F.softmax(L0.beta(h), -1)[i].argmax().item() for i in range(4)],
            w=[F.softmax(L0.wlog, -1)[o].argmax().item() for o in range(16)],
            d0=[F.softmax(L0.off[0], -1)[o].argmax().item() for o in range(16)],
            d1=[F.softmax(L0.off[1], -1)[o].argmax().item() for o in range(16)])

def perms(tab):
    P = torch.zeros(16, K, K)
    for o in range(16):
        P[o, :6, :6] = B1[tab["d0"][o]]
        P[o, 6:, 6:] = B1[tab["d1"][o]]
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

FVAL = [gen_abcp(12, 48, torch.Generator().manual_seed(9000 + i)) for i in range(2)]
SVAL = [gen_abcp(24, 64, torch.Generator().manual_seed(9010 + i)) for i in range(3)]
def score(tab, full=False):
    c = t = 0
    for x, y, _, _ in (SVAL if full else FVAL):
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    return c / t

def polish(tab):
    s = score(tab)
    for key, dom, idxs in (("q", range(K), range(4)), ("b", range(10), range(4)),
                           ("w", range(K), sorted(set(tab["o"])))):
        for i in idxs:
            for val in dom:
                if tab[key][i] == val: continue
                t2 = {k: list(v) for k, v in tab.items()}; t2[key][i] = val
                s2 = score(t2)
                if s2 > s: tab, s = t2, s2
    return tab, s

def keyt(tab): return json.dumps(tab, sort_keys=True)

tab0 = extract()
tab0, s0 = polish(tab0)
print(f"[beam] init (post-polish) {s0:.4f}", flush=True)
beam = [(s0, tab0)]; best_s, best_tab = s0, tab0
seen = {keyt(tab0)}
for rnd in range(1, 6):
    cands = []
    for s, tab in beam:
        used = sorted(set(tab["o"]))
        edits = [(k, o, v) for k in ("d0", "d1") for o in used for v in range(6)
                 if tab[k][o] != v]
        edits += [("o", i, v) for i in range(4) for v in range(16) if tab["o"][i] != v]
        singles = edits
        pairs = list(itertools.combinations([e for e in edits if e[0] in ("d0", "d1")], 2))
        for eset in [ [e] for e in singles ] + [list(p) for p in pairs]:
            t2 = {k: list(v) for k, v in tab.items()}
            for k, i, v in eset: t2[k][i] = v
            kk = keyt(t2)
            if kk in seen: continue
            seen.add(kk)
            cands.append((score(t2), t2))
    cands.sort(key=lambda z: -z[0])
    beam = []
    for s, tab in cands[:4]:
        tab, s = polish(tab)
        beam.append((s, tab))
        if s > best_s: best_s, best_tab = s, tab
    beam.sort(key=lambda z: -z[0])
    print(f"[beam] round {rnd}: top={beam[0][0]:.4f} best={best_s:.4f} "
          f"({len(cands)} cands, {time.time()-t0:.0f}s)", flush=True)
    if best_s >= 1.0: break

tab = best_tab
print(f"[beam] final search best {best_s:.4f} tab_o={tab['o']} "
      f"d0={[tab['d0'][o] for o in sorted(set(tab['o']))]} "
      f"d1={[tab['d1'][o] for o in sorted(set(tab['o']))]}", flush=True)

cont = [model.emb.weight, L0.out.weight, L0.out.bias, model.head.weight,
        model.head.bias, model.norm.weight, model.norm.bias, L0.vcode, L0.S0]
for pp in model.parameters(): pp.requires_grad_(False)
for pp in cont: pp.requires_grad_(True)
opt = torch.optim.AdamW(cont, lr=3e-3)
g = torch.Generator().manual_seed(92)
for step in range(1, 1501):
    x, y, _, _ = gen_abcp(32, 64, g)
    loss = F.cross_entropy(run(tab, x, grad=True).reshape(-1, 10), y.reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
tab, sfin = polish(tab)
print(f"[beam] recalib loss {loss.item():.6f} full-val {score(tab, full=True):.4f}", flush=True)

res = {"cert64": round(score(tab, full=True), 4)}
for L in (256, 1024, 4096):
    c = t = 0
    for i in range(3):
        x, y, _, _ = gen_abcp(4, L, torch.Generator().manual_seed(9990 + L + i))
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    res[str(L)] = round(c / t, 4)
out = dict(tag="EXP073-BEAM-OFFSET", acc=res,
           certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
