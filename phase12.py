"""M27: program surgery on OpPRAM dyck2p checkpoint."""
import itertools, json, resource, sys, time, torch, torch.nn.functional as F
from tasks3 import gen_dyck2p, gen_dyck2
from models3 import OpPRAM
from models import sinkhorn, hard_perm

CKPT = sys.argv[1] if len(sys.argv) > 1 else "oppram_dyckp_s0.pt"
torch.manual_seed(0)
model = OpPRAM(7, 3)
model.load_state_dict(torch.load(CKPT))
L0 = model.layers[0]; t0 = time.time()

def extract():
    with torch.no_grad():
        h = model.emb(torch.arange(7))
        P = sinkhorn(L0.protos / L0.tau, L0.sink_iters); Ph = hard_perm(P)
        tab = dict(
            o=[F.softmax(L0.alpha(h), -1)[i].argmax().item() for i in range(7)],
            q=[F.softmax(L0.readq(h), -1)[i].argmax().item() for i in range(7)],
            b=[F.softmax(L0.beta(h), -1)[i].argmax().item() for i in range(7)],
            w=[F.softmax(L0.wlog, -1)[o].argmax().item() for o in range(16)])
    return tab, Ph

def run_table(tab, Ph, x):
    B, L = x.shape
    with torch.no_grad():
        h = model.emb(x)
        o = torch.tensor(tab["o"])[x]                       # (B,L)
        Rt = Ph[o]
        gt = L0.gbits[o].unsqueeze(-1)
        wt = F.one_hot(torch.tensor(tab["w"]), 8).float()[o]
        qt = F.one_hot(torch.tensor(tab["q"])[x], 8).float()
        vt = L0.vcode[torch.tensor(tab["b"])[x]]
        S = L0.S0.expand(B, -1, -1).contiguous(); reads = []
        for t in range(L):
            gw = (gt[:, t] * wt[:, t]).unsqueeze(-1)
            S = torch.matmul(Rt[:, t], S - gw * (wt[:, t].unsqueeze(1) @ S)
                             + gw * vt[:, t].unsqueeze(1))
            reads.append(torch.bmm(qt[:, t].unsqueeze(1), S).squeeze(1))
        r = torch.stack(reads, 1)
        ho = h + L0.out(torch.cat([r, h], -1))
        return model.head(model.norm(ho)) + r @ L0.vcode.t()

VAL = [gen_dyck2p(24, 64, torch.Generator().manual_seed(7300 + i)) for i in range(3)]
def score(tab, Ph):
    c = t = 0
    for x, y, _, _ in VAL:
        p = run_table(tab, Ph, x).argmax(-1)
        c += (p == y).sum().item(); t += y.numel()
    return c / t

tab, Ph = extract()
base = score(tab, Ph)
print(f"[surgery] base {base:.4f} tab={tab}", flush=True)
used_ops = sorted(set(tab["o"]))
improved, rounds = True, 0
while improved and rounds < 15 and base < 1.0:
    improved = False; rounds += 1
    bg, be = 0.0, None
    for key, dom, n in (("o", range(16), 7), ("q", range(8), 7), ("b", range(3), 7)):
        for i in range(n):
            for val in dom:
                if tab[key][i] == val: continue
                t2 = {k: list(v) for k, v in tab.items()}; t2[key][i] = val
                s = score(t2, Ph)
                if s - base > bg: bg, be = s - base, ("tab", t2, s)
    for o in used_ops:
        for val in range(8):
            if tab["w"][o] == val: continue
            t2 = {k: list(v) for k, v in tab.items()}; t2["w"][o] = val
            s = score(t2, Ph)
            if s - base > bg: bg, be = s - base, ("tab", t2, s)
        for i, j in itertools.combinations(range(8), 2):
            P2 = Ph.clone(); P2[o, [i, j]] = P2[o, [j, i]]
            s = score(tab, P2)
            if s - base > bg: bg, be = s - base, ("proto", P2, s)
    if be:
        kind, obj, s = be
        if kind == "tab": tab = obj
        else: Ph = obj
        base = s; improved = True
        print(f"[surgery] r{rounds}: {kind} -> {base:.4f}", flush=True)

print(f"[surgery] final {base:.4f} tab={tab}", flush=True)
res = {"cert64_peek": round(score(tab, Ph), 4)}
def acc(gen, L, bs, n, off):
    c = t = 0
    for i in range(n):
        x, y, _, _ = gen(bs, L, torch.Generator().manual_seed(off + L + i))
        p = run_table(tab, Ph, x).argmax(-1)
        c += (p == y).sum().item(); t += y.numel()
    return round(c / t, 4)
for L in (256, 1024, 4096):
    res["peek" + str(L)] = acc(gen_dyck2p, L, 4, 3, 9300)
    res["plain" + str(L)] = acc(gen_dyck2, L, 4, 3, 9700)
out = dict(tag="EXP053-OPPRAM-SURGERY", ckpt=CKPT, acc=res, table=tab,
           certified=res["cert64_peek"] == 1.0,
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
           wall_s=round(time.time() - t0, 1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
torch.save(dict(tab=tab, Ph=Ph), "oppram_dyckp_surgery.pt")
