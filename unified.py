"""Unified ISA pipeline: train -> generic beam surgery -> recalib -> certify -> stress.
python3 unified.py <task> <seed>   task in {dyck2p, agree, abcp}"""
import itertools, json, resource, sys, time, torch, torch.nn.functional as F
from models5 import RoleOpPRAM, role_basis
from models import count_params

TASK, SEED = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
if TASK == "dyck2p":
    from tasks3 import gen_dyck2p as GEN
    VIN, VOUT = 7, 3
    def curr(f): return dict(max_depth=1 if f < .15 else 2 if f < .3 else
                             3 if f < .5 else 4 if f < .7 else 6)
elif TASK == "agree":
    from tasks5 import gen_agree as GEN
    VIN, VOUT = 5, 3
    def curr(f): return dict(max_depth=1 if f < .15 else 2 if f < .3 else
                             3 if f < .5 else 4 if f < .7 else 5)
elif TASK == "wwr":
    from tasks7 import gen_wwr as GEN
    VIN, VOUT = 3, 4
    def curr(f): return dict(nmax=2 if f < .2 else 4 if f < .4 else 6 if f < .6 else 8)
else:
    from tasks4 import gen_abcp as GEN
    VIN, VOUT = 4, 10
    def curr(f): return dict(nmax=2 if f < .2 else 3 if f < .4 else 4 if f < .6 else 5)

torch.manual_seed(SEED)
model = RoleOpPRAM(VIN, VOUT, fixed_isa=True)
L0 = model.layers[0]; K = 12; NOPS = L0.n_ops
PH = torch.stack([role_basis(K)[o % 8] for o in range(NOPS)])
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
for step in range(1, 12001):
    f = step / 12000
    L = 32 if f < 0.5 else 64
    x, y, _, _ = GEN(32, L, g, **curr(f))
    loss = F.cross_entropy(model(x).reshape(-1, VOUT), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
print(f"[{TASK}] train done loss {loss.item():.5f} {time.time()-t0:.0f}s", flush=True)
torch.save(model.state_dict(), f"isa_{TASK}_s{SEED}.pt")

def extract():
    with torch.no_grad():
        h = model.emb(torch.arange(VIN))
        return dict(
            o=[F.softmax(L0.alpha(h), -1)[i].argmax().item() for i in range(VIN)],
            q=[F.softmax(L0.readq(h), -1)[i].argmax().item() for i in range(VIN)],
            b=[F.softmax(L0.beta(h), -1)[i].argmax().item() for i in range(VIN)],
            w=[F.softmax(L0.wlog, -1)[o].argmax().item() for o in range(NOPS)])

def run(tab, x, grad=False):
    B, L = x.shape
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = model.emb(x)
        o = torch.tensor(tab["o"])[x]; Rt = PH[o]
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

FVAL = [GEN(12, 48, torch.Generator().manual_seed(9200 + i)) for i in range(2)]
SVAL = [GEN(24, 64, torch.Generator().manual_seed(9210 + i)) for i in range(3)]
def score(tab, full=False):
    c = t = 0
    for x, y, _, _ in (SVAL if full else FVAL):
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    return c / t

def keyt(tab): return json.dumps(tab, sort_keys=True)
def polish(tab):
    s = score(tab)
    for key, dom, idxs in (("q", range(K), range(VIN)), ("b", range(VOUT), range(VIN)),
                           ("w", range(K), sorted(set(tab["o"])))):
        for i in idxs:
            for val in dom:
                if tab[key][i] == val: continue
                t2 = {k: list(v) for k, v in tab.items()}; t2[key][i] = val
                s2 = score(t2)
                if s2 > s: tab, s = t2, s2
    return tab, s

tab = extract()
raw = score(tab, full=True)
tab, s = polish(tab)
beam = [(s, tab)]; best_s, best_tab = s, tab
seen = {keyt(tab)}; n_edits = 0
print(f"[{TASK}] raw {raw:.4f} post-polish {s:.4f}", flush=True)
for rnd in range(1, 7):
    if best_s >= 1.0: break
    cands = []
    for s_, tb in beam:
        es = [("o", i, v) for i in range(VIN) for v in range(NOPS) if tb["o"][i] != v]
        es += [("q", i, v) for i in range(VIN) for v in range(K) if tb["q"][i] != v]
        sets = [[e] for e in es] + [list(p) for p in itertools.combinations(
                [e for e in es if e[0] == "o"], 2)]
        for eset in sets:
            t2 = {k: list(v) for k, v in tb.items()}
            for k, i, v in eset: t2[k][i] = v
            kk = keyt(t2)
            if kk in seen: continue
            seen.add(kk); cands.append((score(t2), t2))
    cands.sort(key=lambda z: -z[0])
    beam = []
    for s_, tb in cands[:3]:
        tb, s_ = polish(tb)
        beam.append((s_, tb))
        if s_ > best_s: best_s, best_tab = s_, tb; n_edits = rnd
    beam.sort(key=lambda z: -z[0])
    print(f"[{TASK}] beam r{rnd}: best {best_s:.4f}", flush=True)

tab = best_tab
cont = [model.emb.weight, L0.out.weight, L0.out.bias, model.head.weight,
        model.head.bias, model.norm.weight, model.norm.bias, L0.vcode, L0.S0]
for pp in model.parameters(): pp.requires_grad_(False)
for pp in cont: pp.requires_grad_(True)
opt = torch.optim.AdamW(cont, lr=3e-3)
g2 = torch.Generator().manual_seed(94)
for step in range(1, 1501):
    x, y, _, _ = GEN(32, 64, g2)
    loss = F.cross_entropy(run(tab, x, grad=True).reshape(-1, VOUT), y.reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
tab, _ = polish(tab)
res = {"cert64": round(score(tab, full=True), 4)}
for L in (256, 1024, 4096):
    c = t = 0
    for i in range(3):
        x, y, _, _ = GEN(4, L, torch.Generator().manual_seed(9970 + L + i))
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    res[str(L)] = round(c / t, 4)
out = dict(tag=f"EXP078-UNIFIED-{TASK.upper()}-S{SEED}", raw=round(raw, 4),
           beam_rounds=n_edits, acc=res, params=count_params(model),
           certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time() - t0, 1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
