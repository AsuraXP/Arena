"""AUTO-COMPILER: automated neuro-algebraic compilation for OpPRAM checkpoints.
python3 auto_compile.py <ckpt> <vocab_in> <vocab_out> <k> <depth_req>
Pipeline: extract table -> repair loop (algebraic invariants + greedy fallback)
-> continuous recalibration -> certification -> stress eval to L=4096."""
import itertools, json, resource, sys, time, torch, torch.nn.functional as F
from tasks3 import gen_dyck3p, gen_dyck3
from models3 import OpPRAM
from models import sinkhorn, hard_perm

CKPT, VIN, VOUT, K, DREQ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
                           int(sys.argv[4]), int(sys.argv[5])
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
def orbit_len(Ph, o, s):
    p = parr(Ph, o); inv = [0] * K
    for r in range(K): inv[p[r]] = r
    seen, l = set(), s
    while l not in seen: seen.add(l); l = inv[l]
    return len(seen)

def set_perm(Ph, o, p):
    Ph = Ph.clone(); Ph[o].zero_()
    for r in range(K): Ph[o, r, p[r]] = 1.0
    return Ph

tab, Ph = extract()
base = score(tab, Ph)
used = sorted(set(tab["o"]))
wops = [o for o in used if L0.gbits[o] > 0]; rops = [o for o in used if L0.gbits[o] == 0]
print(f"[auto] base={base:.4f} used_ops={used} write={wops} read={rops}", flush=True)
for o in wops:
    print(f"[auto] write-op#{o}: staging=w{tab['w'][o]} "
          f"orbit_len={orbit_len(Ph, o, tab['w'][o])} (need>={DREQ + 1})", flush=True)
log = []
improved, sweep = True, 0
while improved and sweep < 10 and base < 1.0:
    improved = False; sweep += 1
    cands = []
    for o in rops:                                        # R1 identity-snap
        cands.append(("id-snap", tab, set_perm(Ph, o, list(range(K)))))
    for o1 in wops:                                       # R2 inverse pairing
        p1 = parr(Ph, o1); inv = [0] * K
        for r in range(K): inv[p1[r]] = r
        for o2 in used:
            if o2 != o1: cands.append((f"inv{o1}->{o2}", tab, set_perm(Ph, o2, inv)))
    for o in wops:                                        # R3 orbit extension swaps
        if orbit_len(Ph, o, tab["w"][o]) >= DREQ + 1: continue
        p = parr(Ph, o)
        for i, j in itertools.combinations(range(K), 2):
            p2 = list(p); p2[i], p2[j] = p2[j], p2[i]
            cands.append((f"orbit{o}-swap{i},{j}", tab, set_perm(Ph, o, p2)))
    for c in range(VOUT):                                 # R4 S0 normalization
        with torch.no_grad():
            old = L0.S0.clone()
            L0.S0.copy_(L0.vcode[c].unsqueeze(0).repeat(K, 1))
            s = score(tab, Ph)
            L0.S0.copy_(old)
        if s > base:
            cands.append((f"S0<-code{c}", "S0", c))
    for key, dom in (("o", range(16)), ("q", range(K)), ("b", range(VOUT))):
        for i in range(VIN):                              # greedy fallback
            for val in dom:
                if tab[key][i] == val: continue
                t2 = {kk: list(v) for kk, v in tab.items()}; t2[key][i] = val
                cands.append((f"{key}[{i}]={val}", t2, Ph))
    for o in used:
        for val in range(K):
            if tab["w"][o] == val: continue
            t2 = {kk: list(v) for kk, v in tab.items()}; t2["w"][o] = val
            cands.append((f"w[{o}]={val}", t2, Ph))
    best, bs = None, base
    for name, a1, a2 in cands:
        if a1 == "S0":
            with torch.no_grad():
                old = L0.S0.clone()
                L0.S0.copy_(L0.vcode[a2].unsqueeze(0).repeat(K, 1))
                s = score(tab, Ph); L0.S0.copy_(old)
            if s > bs: best, bs = (name, a1, a2), s
        else:
            s = score(a1, a2)
            if s > bs: best, bs = (name, a1, a2), s
    if best:
        name, a1, a2 = best
        if a1 == "S0":
            with torch.no_grad():
                L0.S0.copy_(L0.vcode[a2].unsqueeze(0).repeat(K, 1))
        else:
            tab, Ph = a1, a2
        base = bs; improved = True; log.append(name)
        print(f"[auto] sweep {sweep}: ACCEPT {name} -> {base:.4f}", flush=True)

print(f"[auto] repairs={log}", flush=True)
# continuous recalibration with frozen program
cont = [model.emb.weight, L0.out.weight, L0.out.bias, model.head.weight,
        model.head.bias, model.norm.weight, model.norm.bias, L0.vcode, L0.S0]
for p in model.parameters(): p.requires_grad_(False)
for p in cont: p.requires_grad_(True)
opt = torch.optim.AdamW(cont, lr=3e-3)
g = torch.Generator().manual_seed(77)
for step in range(1, 1501):
    x, y, _, _ = gen_dyck3p(32, 64, g, max_depth=DREQ)
    loss = F.cross_entropy(run(tab, Ph, x, grad=True).reshape(-1, VOUT), y.reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
print(f"[auto] recalib loss {loss.item():.6f} val {score(tab, Ph):.4f}", flush=True)

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
out = dict(tag="EXP060-AUTOCOMPILE-DYCK3", ckpt=CKPT, repairs=log, acc=res,
           certified=all(v == 1.0 for v in res.values()),
           wall_s=round(time.time() - t0, 1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
torch.save(dict(tab=tab, Ph=Ph, model=model.state_dict()), "oppram_dyck3_compiled.pt")
