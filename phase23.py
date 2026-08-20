"""ISA-space beam surgery on abcp. python3 phase23.py <ckpt>"""
import itertools, json, resource, sys, time, torch, torch.nn.functional as F
from tasks4 import gen_abcp
from models5 import RoleOpPRAM, role_basis
CKPT = sys.argv[1]
torch.manual_seed(0)
model = RoleOpPRAM(4, 10, fixed_isa=True)
model.load_state_dict(torch.load(CKPT))
L0 = model.layers[0]; t0 = time.time()
K = 12; ROLES = role_basis(K)
PH = torch.stack([ROLES[o % 6] for o in range(12)])   # fixed instruction perms

def extract():
    with torch.no_grad():
        h = model.emb(torch.arange(4))
        return dict(
            o=[F.softmax(L0.alpha(h), -1)[i].argmax().item() for i in range(4)],
            q=[F.softmax(L0.readq(h), -1)[i].argmax().item() for i in range(4)],
            b=[F.softmax(L0.beta(h), -1)[i].argmax().item() for i in range(4)],
            w=[F.softmax(L0.wlog, -1)[o].argmax().item() for o in range(12)],
            s0=[-1, -1, -1])            # s0 = (base_code, lane0_code, lane6_code); -1 = keep learned

def s0_tensor(spec):
    if spec[0] < 0: return None
    S0 = L0.vcode[spec[0]].unsqueeze(0).repeat(K, 1).clone()
    S0[0] = L0.vcode[spec[1]]; S0[6] = L0.vcode[spec[2]]
    return S0

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
        S0 = s0_tensor(tab["s0"])
        S = (S0 if S0 is not None else L0.S0).expand(B, -1, -1); reads = []
        for t in range(L):
            gw = (gt[:, t] * wt[:, t]).unsqueeze(-1)
            S = torch.matmul(Rt[:, t], S - gw * (wt[:, t].unsqueeze(1) @ S)
                             + gw * vt[:, t].unsqueeze(1))
            reads.append(torch.bmm(qt[:, t].unsqueeze(1), S).squeeze(1))
        r = torch.stack(reads, 1)
        ho = h + L0.out(torch.cat([r, h], -1))
        return model.head(model.norm(ho)) + r @ L0.vcode.t()

FVAL = [gen_abcp(12, 48, torch.Generator().manual_seed(9100 + i)) for i in range(2)]
SVAL = [gen_abcp(24, 64, torch.Generator().manual_seed(9110 + i)) for i in range(3)]
def score(tab, full=False):
    c = t = 0
    for x, y, _, _ in (SVAL if full else FVAL):
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    return c / t

def keyt(tab): return json.dumps(tab, sort_keys=True)
def edits_of(tab):
    es = [("o", i, v) for i in range(4) for v in range(12) if tab["o"][i] != v]
    es += [("q", i, v) for i in range(4) for v in range(K) if tab["q"][i] != v]
    return es

tab = extract()
base = score(tab)
print(f"[isa-beam] init {base:.4f} o={tab['o']} q={tab['q']}", flush=True)
# S0 candidates up front
best_s0, bs0 = tab["s0"], base
for c0, c1, c2 in itertools.product(range(4), range(4, 8), range(4, 8)):
    t2 = {k: list(v) for k, v in tab.items()}; t2["s0"] = [c0, c1, c2]
    s = score(t2)
    if s > bs0: best_s0, bs0 = [c0, c1, c2], s
tab["s0"] = best_s0; base = max(base, bs0)
print(f"[isa-beam] S0 pass -> {base:.4f} s0={tab['s0']}", flush=True)

beam = [(base, tab)]; best_s, best_tab = base, tab
seen = {keyt(tab)}
for rnd in range(1, 7):
    cands = []
    for s, tb in beam:
        es = edits_of(tb)
        sets = [[e] for e in es] + [list(p) for p in itertools.combinations(
                [e for e in es if e[0] == "o"], 2)]
        for eset in sets:
            t2 = {k: list(v) for k, v in tb.items()}
            for k, i, v in eset: t2[k][i] = v
            kk = keyt(t2)
            if kk in seen: continue
            seen.add(kk)
            cands.append((score(t2), t2))
    cands.sort(key=lambda z: -z[0])
    beam = cands[:4]
    if beam and beam[0][0] > best_s: best_s, best_tab = beam[0]
    print(f"[isa-beam] r{rnd}: top={beam[0][0]:.4f} best={best_s:.4f} "
          f"({len(cands)} cands {time.time()-t0:.0f}s)", flush=True)
    if best_s >= 1.0: break

tab = best_tab
print(f"[isa-beam] search best {best_s:.4f} o={tab['o']} q={tab['q']} s0={tab['s0']}",
      flush=True)
cont = [model.emb.weight, L0.out.weight, L0.out.bias, model.head.weight,
        model.head.bias, model.norm.weight, model.norm.bias, L0.vcode]
for pp in model.parameters(): pp.requires_grad_(False)
for pp in cont: pp.requires_grad_(True)
opt = torch.optim.AdamW(cont, lr=3e-3)
g = torch.Generator().manual_seed(93)
for step in range(1, 1501):
    x, y, _, _ = gen_abcp(32, 64, g)
    loss = F.cross_entropy(run(tab, x, grad=True).reshape(-1, 10), y.reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
print(f"[isa-beam] recalib loss {loss.item():.6f} val {score(tab, full=True):.4f}",
      flush=True)
res = {"cert64": round(score(tab, full=True), 4)}
for L in (256, 1024, 4096):
    c = t = 0
    for i in range(3):
        x, y, _, _ = gen_abcp(4, L, torch.Generator().manual_seed(9980 + L + i))
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    res[str(L)] = round(c / t, 4)
out = dict(tag="EXP077-ISA-BEAM-ABCP", acc=res, final=dict(o=tab["o"], q=tab["q"],
           s0=tab["s0"]), certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
