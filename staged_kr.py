"""Staged compilation for KR-ISA modal+probes: enumerate stage-1 mode programs,
context-local stage-B greedy, select by val, recalib, certify. argv: <ckpt>"""
import itertools, json, resource, sys, time, torch, torch.nn.functional as F
from tasks8 import gen_modalp
from models6 import KRISA
CKPT = sys.argv[1]
torch.manual_seed(0)
model = KRISA(7, 5)
model.load_state_dict(torch.load(CKPT))
VIN, VOUT, K, M, NOPS = 7, 5, 12, 4, 16
t0 = time.time()

def extract():
    with torch.no_grad():
        return dict(
            md=[model.mdisp[i].argmax().item() for i in range(VIN)],
            o=[[model.ta[i, m].argmax().item() for m in range(M)] for i in range(VIN)],
            q=[[model.tq[i, m].argmax().item() for m in range(M)] for i in range(VIN)],
            b=[[model.tb[i, m].argmax().item() for m in range(M)] for i in range(VIN)],
            w=[F.softmax(model.wlog, -1)[o].argmax().item() for o in range(NOPS)])

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

FVAL = [gen_modalp(16, 48, torch.Generator().manual_seed(9600 + i)) for i in range(2)]
CVAL = [gen_modalp(24, 64, torch.Generator().manual_seed(9650 + i)) for i in range(6)] + \
       [gen_modalp(8, 160, torch.Generator().manual_seed(9690 + i)) for i in range(2)]
def score(tab, cert=False):
    c = t = 0
    for x, y, _, _ in (CVAL if cert else FVAL):
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    return c / t

def greedy_ctx(tab, keys=("o",), rounds=2):
    s = score(tab)
    for _ in range(rounds):
        improved = False
        for key in keys:
            dom = range(NOPS) if key == "o" else (range(K) if key == "q" else range(VOUT))
            for i in range(VIN):
                for mm in range(M):
                    for val in dom:
                        if tab[key][i][mm] == val: continue
                        t2 = json.loads(json.dumps(tab)); t2[key][i][mm] = val
                        s2 = score(t2)
                        if s2 > s: tab, s, improved = t2, s2, True
        if not improved: break
    return tab, s

base_tab = extract()
print(f"[staged] raw {score(base_tab, cert=True):.4f}", flush=True)
best_s, best_tab = -1, None
for m0, m1 in itertools.product(range(6), range(6)):
    tab = json.loads(json.dumps(base_tab))
    tab["md"] = [m0, m1, 0, 0, 0, 0, 0]           # content tokens: id
    tab, s = greedy_ctx(tab, keys=("o",), rounds=1)
    if s > best_s: best_s, best_tab = s, tab
print(f"[staged] stage-1 winner md={best_tab['md'][:2]} score {best_s:.4f} "
      f"({time.time()-t0:.0f}s)", flush=True)
tab, s = greedy_ctx(best_tab, keys=("o", "q", "b"), rounds=3)
# w pass
for o in set(x for row in tab["o"] for x in row):
    for val in range(K):
        if tab["w"][o] == val: continue
        t2 = json.loads(json.dumps(tab)); t2["w"][o] = val
        s2 = score(t2)
        if s2 > s: tab, s = t2, s2
print(f"[staged] stage-B done {s:.4f} cert-val {score(tab, cert=True):.4f}", flush=True)

cont = [model.emb.weight, model.out.weight, model.out.bias, model.head.weight,
        model.head.bias, model.norm.weight, model.norm.bias, model.vcode, model.S0]
for p in model.parameters(): p.requires_grad_(False)
for p in cont: p.requires_grad_(True)
opt = torch.optim.AdamW(cont, lr=3e-3)
g = torch.Generator().manual_seed(98)
for step in range(1, 1501):
    x, y, _, _ = gen_modalp(32, 64, g)
    loss = F.cross_entropy(run(tab, x, grad=True).reshape(-1, VOUT), y.reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
tab, _ = greedy_ctx(tab, keys=("o", "q", "b"), rounds=1)
res = {"cert": round(score(tab, cert=True), 4)}
for L in (256, 1024, 4096):
    c = t = 0
    for i in range(3):
        x, y, _, _ = gen_modalp(4, L, torch.Generator().manual_seed(9920 + L + i))
        p = run(tab, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    res[str(L)] = round(c / t, 4)
out = dict(tag="EXP086-STAGED-KR", md=tab["md"], acc=res,
           certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
