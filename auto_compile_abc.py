"""abc auto-compiler: generic pass + counter-canonicalization. argv: <ckpt>"""
import itertools, json, resource, sys, time, torch, torch.nn.functional as F
from tasks4 import gen_abc
from models3 import OpPRAM
from models import sinkhorn, hard_perm
CKPT = sys.argv[1]; VIN, VOUT, K = 3, 4, 12
torch.manual_seed(0)
model = OpPRAM(VIN, VOUT, k=K, n_ops=16)
model.load_state_dict(torch.load(CKPT))
L0 = model.layers[0]; t0 = time.time()

def extract():
    with torch.no_grad():
        h = model.emb(torch.arange(VIN))
        P = sinkhorn(L0.protos / L0.tau, L0.sink_iters); Ph = hard_perm(P)
        tab = dict(
            o=[F.softmax(L0.alpha(h), -1)[i].argmax().item() for i in range(VIN)],
            q=[F.softmax(L0.readq(h), -1)[i].argmax().item() for i in range(VIN)],
            b=[F.softmax(L0.beta(h), -1)[i].argmax().item() for i in range(VIN)],
            w=[F.softmax(L0.wlog, -1)[o].argmax().item() for o in range(16)])
    return tab, Ph

def run(tab, Ph, x, grad=False):
    B, L = x.shape
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

VAL = [gen_abc(24, 64, torch.Generator().manual_seed(8200 + i)) for i in range(3)]
def score(tab, Ph):
    c = t = 0
    for x, y, _, _ in VAL:
        p = run(tab, Ph, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    return c / t

def set_perm(Ph, o, p):
    Ph = Ph.clone(); Ph[o].zero_()
    for r in range(K): Ph[o, r, p[r]] = 1.0
    return Ph

tab, Ph = extract()
print(f"[abc] raw {score(tab, Ph):.4f} tab={tab['o']} q={tab['q']}", flush=True)

# counter-canonicalization: two 6-cycles, synchronized rotations, marker S0
def rot(cyc, d):
    p = list(range(K))
    n = len(cyc)
    for idx, lane in enumerate(cyc):
        p[cyc[(idx + d) % n]] = lane      # new[cyc[idx+d]] = old[cyc[idx]]
    return p
C1, C2 = list(range(6)), list(range(6, 12))
pa = rot(C1, 1); pb_ = rot(C1, -1); pc_ = rot(C2, -1)
pa2 = rot(C2, 1)
pa = [pa2[r] if r >= 6 else pa[r] for r in range(K)]   # a rotates both cycles
oa, ob, oc = 8, 9, 10                                   # pure-route slots (g=0)
Ph = set_perm(Ph, oa, pa); Ph = set_perm(Ph, ob, pb_); Ph = set_perm(Ph, oc, pc_)
tab["o"] = [oa, ob, oc]
tab["q"] = [0, 0, 6]
with torch.no_grad():
    L0.S0.copy_(L0.vcode[0].unsqueeze(0).repeat(K, 1))
    L0.S0[0] = L0.vcode[1]; L0.S0[6] = L0.vcode[3]
s = score(tab, Ph)
print(f"[abc] canonical counters {s:.4f}", flush=True)
# greedy q/b polish
for key, dom in (("q", range(K)), ("b", range(VOUT))):
    for i in range(VIN):
        for val in dom:
            if tab[key][i] == val: continue
            t2 = {kk: list(v) for kk, v in tab.items()}; t2[key][i] = val
            s2 = score(t2, Ph)
            if s2 > s: tab, s = t2, s2
print(f"[abc] post-polish {s:.4f}", flush=True)

cont = [model.emb.weight, L0.out.weight, L0.out.bias, model.head.weight,
        model.head.bias, model.norm.weight, model.norm.bias]
for pp in model.parameters(): pp.requires_grad_(False)
for pp in cont: pp.requires_grad_(True)
opt = torch.optim.AdamW(cont, lr=3e-3)
g = torch.Generator().manual_seed(79)
for step in range(1, 1501):
    x, y, _, _ = gen_abc(32, 64, g)
    loss = F.cross_entropy(run(tab, Ph, x, grad=True).reshape(-1, VOUT), y.reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
print(f"[abc] recalib loss {loss.item():.6f} val {score(tab, Ph):.4f}", flush=True)
res = {"cert64": round(score(tab, Ph), 4)}
for L in (256, 1024, 4096):
    c = t = 0
    for i in range(3):
        x, y, _, _ = gen_abc(4, L, torch.Generator().manual_seed(9600 + L + i))
        p = run(tab, Ph, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    res[str(L)] = round(c / t, 4)
out = dict(tag="EXP063-ABC-COMPILED", acc=res,
           certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time() - t0, 1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
