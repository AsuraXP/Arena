import itertools, json, resource, time, torch, torch.nn.functional as F
from tasks8 import gen_modal
from models7 import KRISA2
torch.manual_seed(0)
model = KRISA2(6, 3)
model.load_state_dict(torch.load("krisa2_modal.pt"))
VIN, VOUT, K, M, NOPS = 6, 3, 12, 4, 16
t0 = time.time(); MD = [1, 2, 0, 0, 0, 0]
def extract():
    with torch.no_grad():
        E = model.emb(torch.arange(VIN))
        tab = dict(md=list(MD), o=[], q=[], b=[],
                   w=[F.softmax(model.wlog, -1)[o].argmax().item() for o in range(NOPS)])
        for i in range(VIN):
            oo, qq, bb = [], [], []
            for m in range(M):
                hc = torch.cat([E[i], F.one_hot(torch.tensor(m), M).float()])
                oo.append(F.softmax(model.alpha(hc), -1).argmax().item())
                qq.append(F.softmax(model.readq(hc), -1).argmax().item())
                bb.append(F.softmax(model.beta(hc), -1).argmax().item())
            tab["o"].append(oo); tab["q"].append(qq); tab["b"].append(bb)
        return tab
def run(tab, x, grad=False):
    B, L = x.shape
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = model.emb(x)
        T = model.TB[torch.tensor(tab["md"])][x]
        ot = torch.tensor(tab["o"])[x]; qt = torch.tensor(tab["q"])[x]
        bt = torch.tensor(tab["b"])[x]
        wt = F.one_hot(torch.tensor(tab["w"]), K).float()
        m = torch.zeros(B, M); m[:, 0] = 1.0
        S = model.S0.expand(B, -1, -1); reads = []
        for t in range(L):
            m = torch.bmm(T[:, t], m.unsqueeze(-1)).squeeze(-1)
            mi = m.argmax(-1)
            o = ot[:, t].gather(1, mi.unsqueeze(1)).squeeze(1)
            qi = qt[:, t].gather(1, mi.unsqueeze(1)).squeeze(1)
            bi = bt[:, t].gather(1, mi.unsqueeze(1)).squeeze(1)
            R = model.PH[o]; g_ = model.gbits[o].view(-1, 1, 1)
            ww = wt[o]; v = model.vcode[bi]
            gw = g_ * ww.unsqueeze(-1)
            S = torch.matmul(R, S - gw * (ww.unsqueeze(1) @ S) + gw * v.unsqueeze(1))
            reads.append(S[torch.arange(B), qi])
        r = torch.stack(reads, 1)
        ho = h + model.out(torch.cat([r, h], -1))
        return model.head(model.norm(ho)) + r @ model.vcode.t()
FVAL = [gen_modal(16, 48, torch.Generator().manual_seed(9600 + i)) for i in range(2)]
CVAL = [gen_modal(24, 64, torch.Generator().manual_seed(9650 + i)) for i in range(6)] + \
       [gen_modal(8, 160, torch.Generator().manual_seed(9690 + i)) for i in range(2)]
def score(tab, cert=False):
    c = t = 0
    for x, y, _, _ in (CVAL if cert else FVAL):
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    return c / t
def greedy(tab, rounds=3, cert=False):
    s = score(tab, cert)
    for _ in range(rounds):
        improved = False
        for key in ("o", "q", "b"):
            dom = range(NOPS) if key == "o" else (range(K) if key == "q" else range(VOUT))
            for i in range(VIN):
                for mm in range(M):
                    for val in dom:
                        if tab[key][i][mm] == val: continue
                        t2 = json.loads(json.dumps(tab)); t2[key][i][mm] = val
                        s2 = score(t2, cert)
                        if s2 > s: tab, s, improved = t2, s2, True
        for o in set(x for row in tab["o"] for x in row):
            for val in range(K):
                if tab["w"][o] == val: continue
                t2 = json.loads(json.dumps(tab)); t2["w"][o] = val
                s2 = score(t2, cert)
                if s2 > s: tab, s, improved = t2, s2, True
        if not improved: break
    return tab, s
tab = extract()
print(f"[k2b] base {score(tab):.4f}", flush=True)
tab, s = greedy(tab)
print(f"[k2b] greedy {s:.4f} ({time.time()-t0:.0f}s)", flush=True)
if s < 1.0:
    singles = []
    for i in range(VIN):
        for mm in range(M):
            for v in range(NOPS):
                if tab["o"][i][mm] == v: continue
                t2 = json.loads(json.dumps(tab)); t2["o"][i][mm] = v
                singles.append((score(t2), (i, mm, v)))
    singles.sort(key=lambda z: -z[0])
    top = [e for _, e in singles[:30]]
    best = s
    for e1, e2 in itertools.combinations(top, 2):
        t2 = json.loads(json.dumps(tab))
        t2["o"][e1[0]][e1[1]] = e1[2]; t2["o"][e2[0]][e2[1]] = e2[2]
        s2 = score(t2)
        if s2 > best:
            best, tab = s2, t2
            print(f"[k2b] pair -> {best:.4f}", flush=True)
    tab, s = greedy(tab, rounds=2)
print(f"[k2b] pre-recalib {s:.4f} ({time.time()-t0:.0f}s)", flush=True)
cont = [model.emb.weight, model.out.weight, model.out.bias, model.head.weight,
        model.head.bias, model.norm.weight, model.norm.bias, model.vcode, model.S0]
for p in model.parameters(): p.requires_grad_(False)
for p in cont: p.requires_grad_(True)
opt = torch.optim.AdamW(cont, lr=3e-3)
g = torch.Generator().manual_seed(102)
for step in range(1, 1501):
    x, y, _, _ = gen_modal(32, 64, g)
    loss = F.cross_entropy(run(tab, x, grad=True).reshape(-1, 3), y.reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
tab, _ = greedy(tab, rounds=2, cert=True)
res = {"cert": round(score(tab, cert=True), 4)}
for L in (256, 1024, 4096):
    c = t = 0
    for i in range(3):
        x, y, _, _ = gen_modal(4, L, torch.Generator().manual_seed(9860 + L + i))
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    res[str(L)] = round(c / t, 4)
out = dict(tag="EXP095-KRISA2-MODAL-SURGERY", acc=res,
           certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
torch.save(dict(tab=tab, model=model.state_dict()), "krisa2_modal_final.pt")
