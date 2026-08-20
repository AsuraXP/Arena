"""AUTO-COMPILER v2: composite canonicalization pass for OpPRAM (M30).
python3 auto_compile2.py <ckpt> <vin> <vout> <k> <depth_req>"""
import itertools, json, resource, sys, time, torch, torch.nn.functional as F
from tasks3 import gen_dyck3p, gen_dyck3
from models3 import OpPRAM
from models import sinkhorn, hard_perm

CKPT, VIN, VOUT, K, DREQ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
                           int(sys.argv[4]), int(sys.argv[5])
OPENS, CLOSES, PEEKS = [0, 2, 4], [1, 3, 5], [6, 7, 8]
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

VAL = [gen_dyck3p(24, 64, torch.Generator().manual_seed(7600 + i)) for i in range(3)]
def score(tab, Ph):
    c = t = 0
    for x, y, _, _ in VAL:
        p = run(tab, Ph, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    return c / t

def parr(Ph, o): return [Ph[o, r].argmax().item() for r in range(K)]
def inv_of(p):
    inv = [0] * K
    for r in range(K): inv[p[r]] = r
    return inv
def orbit(p, s):
    inv = inv_of(p); seen, l = [], s
    while l not in seen: seen.append(l); l = inv[l]
    return seen
def set_perm(Ph, o, p):
    Ph = Ph.clone(); Ph[o].zero_()
    for r in range(K): Ph[o, r, p[r]] = 1.0
    return Ph

tab0, Ph0 = extract()
print(f"[v2] base {score(tab0, Ph0):.4f}", flush=True)
wops = sorted(set(tab0["o"][i] for i in OPENS if L0.gbits[tab0['o'][i]] > 0))
best_prog, best_s = None, -1.0
for o_push in wops:
    for o_pop in [o for o in range(8) if o != o_push]:
        tab = {k: list(v) for k, v in tab0.items()}
        p = parr(Ph0, o_push); s_lane = tab["w"][o_push]
        # extend staging orbit if needed (best single/double swap by orbit length)
        if len(orbit(p, s_lane)) < DREQ + 1:
            bestp, bl = p, len(orbit(p, s_lane))
            for i, j in itertools.combinations(range(K), 2):
                p2 = list(p); p2[i], p2[j] = p2[j], p2[i]
                l2 = len(orbit(p2, s_lane))
                if l2 > bl: bestp, bl = p2, l2
            p = bestp
        Ph = set_perm(Ph0, o_push, p)
        Ph = set_perm(Ph, o_pop, inv_of(p))
        # identity read-op for peeks
        o_id = 15
        Ph = set_perm(Ph, o_id, list(range(K)))
        top = p.index(s_lane)                       # new[top] = old[staging]
        orb = orbit(p, s_lane)                      # [staging, top, below, ...]
        for i in OPENS:  tab["o"][i] = o_push
        for i in CLOSES: tab["o"][i] = o_pop; tab["b"][i] = 0
        for d, i in enumerate(PEEKS):
            tab["o"][i] = o_id
            tab["q"][i] = orb[1 + d] if 1 + d < len(orb) else orb[-1]
        for i in OPENS + CLOSES: tab["q"][i] = top
        tab["w"][o_pop] = top                        # erase removed top pre-perm
        with torch.no_grad():
            oldS0 = L0.S0.clone()
            L0.S0.copy_(L0.vcode[0].unsqueeze(0).repeat(K, 1))
        s = score(tab, Ph)
        # greedy q/b polish
        for key, dom, idxs in (("q", range(K), range(VIN)), ("b", range(VOUT), range(VIN))):
            for i in idxs:
                for val in dom:
                    if tab[key][i] == val: continue
                    t2 = {kk: list(v) for kk, v in tab.items()}; t2[key][i] = val
                    s2 = score(t2, Ph)
                    if s2 > s: tab, s = t2, s2
        print(f"[v2] push#{o_push} pop#{o_pop}: {s:.4f}", flush=True)
        if s > best_s:
            best_s = s
            best_prog = (tab, Ph, L0.S0.clone())
        with torch.no_grad(): L0.S0.copy_(oldS0)

tab, Ph, S0v = best_prog
with torch.no_grad(): L0.S0.copy_(S0v)
print(f"[v2] best composite {best_s:.4f}", flush=True)

cont = [model.emb.weight, L0.out.weight, L0.out.bias, model.head.weight,
        model.head.bias, model.norm.weight, model.norm.bias, L0.vcode, L0.S0]
for pp in model.parameters(): pp.requires_grad_(False)
for pp in cont: pp.requires_grad_(True)
opt = torch.optim.AdamW(cont, lr=3e-3)
g = torch.Generator().manual_seed(78)
for step in range(1, 1501):
    x, y, _, _ = gen_dyck3p(32, 64, g, max_depth=DREQ)
    loss = F.cross_entropy(run(tab, Ph, x, grad=True).reshape(-1, VOUT), y.reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
print(f"[v2] recalib loss {loss.item():.6f} val {score(tab, Ph):.4f}", flush=True)

res = {"cert64": round(score(tab, Ph), 4)}
def acc(gen, L, bs, off):
    c = t = 0
    for i in range(3):
        x, y, _, _ = gen(bs, L, torch.Generator().manual_seed(off + L + i))
        p = run(tab, Ph, x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
    return round(c / t, 4)
for L in (256, 1024, 4096):
    res["peek" + str(L)] = acc(gen_dyck3p, L, 4, 9500)
    res["plain" + str(L)] = acc(gen_dyck3, L, 4, 9900)
out = dict(tag="EXP061-AUTOCOMPILE-V2-DYCK3", ckpt=CKPT, acc=res,
           certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time() - t0, 1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
torch.save(dict(tab=tab, Ph=Ph, model=model.state_dict()), "oppram_dyck3_compiled.pt")
