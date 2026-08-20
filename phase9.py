"""M25: discrete program surgery on Dyck-2. Extract per-token op table from a trained
PRAM, hill-climb single-op edits vs hard accuracy, then tune continuous params only."""
import copy, itertools, json, resource, sys, time, torch, torch.nn.functional as F
from tasks3 import gen_dyck2
from models2 import PRAM, sinkhorn
from models import hard_perm

CKPT = sys.argv[1] if len(sys.argv) > 1 else "pram_dyck_soft_s1.pt"
torch.manual_seed(0)
model = PRAM(4, 3, k=8, n_proto=12, use_scan=False, tie_vals=True)
for l in model.layers: l.use_flat = False
model.load_state_dict(torch.load(CKPT))
L0 = model.layers[0]
t0 = time.time()

def extract_table():
    with torch.no_grad():
        h = model.emb(torch.arange(4))
        P = sinkhorn(L0.protos / L0.tau, L0.sink_iters)
        Ph = hard_perm(P)                                   # (12,8,8)
        tab = dict(
            m=[F.softmax(L0.alpha(h), -1)[i].argmax().item() for i in range(4)],
            w=[F.softmax(L0.sel(h), -1)[i].argmax().item() for i in range(4)],
            q=[F.softmax(L0.readq(h), -1)[i].argmax().item() for i in range(4)],
            b=[F.softmax(L0.beta(h), -1)[i].argmax().item() for i in range(4)],
            g=[int(torch.sigmoid(L0.gate(h))[i] > 0.5) for i in range(4)],
            r=[int(torch.sigmoid(L0.rho(h))[i] > 0.5) for i in range(4)])
    return tab, Ph

def run_table(tab, Ph, x):
    """Exact hard forward with table-overridden discrete ops."""
    B, L = x.shape
    with torch.no_grad():
        h = model.emb(x)
        k, d = L0.k, L0.d_slot
        m = torch.tensor(tab["m"])[x]                       # (B,L)
        Rt = Ph[m]                                          # (B,L,8,8)
        rho = torch.tensor(tab["r"], dtype=torch.float)[x].view(B, L, 1, 1)
        I = torch.eye(k)
        Rt = (1 - rho) * I + rho * Rt
        wt = F.one_hot(torch.tensor(tab["w"])[x], k).float()
        qt = F.one_hot(torch.tensor(tab["q"])[x], k).float()
        gt = torch.tensor(tab["g"], dtype=torch.float)[x].unsqueeze(-1)
        vt = L0.vcode[torch.tensor(tab["b"])[x]]            # (B,L,d)
        S = L0.S0.expand(B, -1, -1).contiguous()
        reads = []
        for t in range(L):
            gw = (gt[:, t] * wt[:, t]).unsqueeze(-1)        # (B,k,1)
            S = torch.matmul(Rt[:, t], S - gw * (wt[:, t].unsqueeze(1) @ S)
                             + gw * vt[:, t].unsqueeze(1))
            reads.append(torch.bmm(qt[:, t].unsqueeze(1), S).squeeze(1))
        r_addr = torch.stack(reads, 1)                      # (B,L,d)
        z = torch.zeros(B, L, 32)
        ho = h + L0.out(torch.cat([r_addr, z, h], -1))
        logits = model.head(model.norm(ho)) + r_addr @ L0.vcode.t()
    return logits

VAL = [gen_dyck2(24, 64, torch.Generator().manual_seed(7000 + i)) for i in range(3)]
def score(tab, Ph):
    c = t = 0
    for x, y, _, _ in VAL:
        p = run_table(tab, Ph, x).argmax(-1)
        c += (p == y).sum().item(); t += y.numel()
    return c / t

tab, Ph = extract_table()
base = score(tab, Ph)
print(f"[surgery] base hard acc {base:.4f} table={tab}", flush=True)

improved = True; rounds = 0
while improved and rounds < 12:
    improved = False; rounds += 1
    best_gain, best_edit = 0.0, None
    # single-op table edits
    for key, dom in (("m", range(12)), ("w", range(8)), ("q", range(8)),
                     ("b", range(3)), ("g", range(2)), ("r", range(2))):
        for i in range(4):
            for val in dom:
                if tab[key][i] == val: continue
                t2 = {k2: list(v) for k2, v in tab.items()}; t2[key][i] = val
                s = score(t2, Ph)
                if s - base > best_gain: best_gain, best_edit = s - base, ("tab", t2, s)
    # prototype swaps (only protos in use)
    for m in set(tab["m"]):
        for i, j in itertools.combinations(range(8), 2):
            Ph2 = Ph.clone(); Ph2[m, [i, j]] = Ph2[m, [j, i]]
            s = score(tab, Ph2)
            if s - base > best_gain: best_gain, best_edit = s - base, ("proto", Ph2, s)
    if best_edit:
        kind, obj, s = best_edit
        if kind == "tab": tab = obj
        else: Ph = obj
        base = s; improved = True
        print(f"[surgery] round {rounds}: accepted {kind} edit -> {base:.4f}", flush=True)

print(f"[surgery] final table={tab}", flush=True)
res = {"cert64_presurgtune": round(base, 4)}

# continuous-only fine-tune with frozen discrete program
cont = [model.emb.weight, L0.out.weight, L0.out.bias, model.head.weight,
        model.head.bias, model.norm.weight, model.norm.bias, L0.S0, L0.vcode]
opt = torch.optim.AdamW(cont, lr=1e-3)
g = torch.Generator().manual_seed(31337)
for p in model.parameters(): p.requires_grad_(False)
for p in cont: p.requires_grad_(True)
for step in range(1, 1501):
    x, y, _, _ = gen_dyck2(32, 64, g, max_depth=6)
    # differentiable variant of run_table (rebuild graph on cont params)
    B, Lx = x.shape
    h = model.emb(x)
    m = torch.tensor(tab["m"])[x]; Rt = Ph[m]
    rho = torch.tensor(tab["r"], dtype=torch.float)[x].view(B, Lx, 1, 1)
    Rt = (1 - rho) * torch.eye(8) + rho * Rt
    wt = F.one_hot(torch.tensor(tab["w"])[x], 8).float()
    qt = F.one_hot(torch.tensor(tab["q"])[x], 8).float()
    gt = torch.tensor(tab["g"], dtype=torch.float)[x].unsqueeze(-1)
    vt = L0.vcode[torch.tensor(tab["b"])[x]]
    S = L0.S0.expand(B, -1, -1); reads = []
    for t in range(Lx):
        gw = (gt[:, t] * wt[:, t]).unsqueeze(-1)
        S = torch.matmul(Rt[:, t], S - gw * (wt[:, t].unsqueeze(1) @ S)
                         + gw * vt[:, t].unsqueeze(1))
        reads.append(torch.bmm(qt[:, t].unsqueeze(1), S).squeeze(1))
    r_addr = torch.stack(reads, 1)
    ho = h + L0.out(torch.cat([r_addr, torch.zeros(B, Lx, 32), h], -1))
    logits = model.head(model.norm(ho)) + r_addr @ L0.vcode.t()
    loss = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 500 == 0: print(f"[cont-tune] {step} loss {loss.item():.5f}", flush=True)

# final certification + stress
def acc_at(Lx, bs, n):
    c = t = 0
    for i in range(n):
        x, y, _, _ = gen_dyck2(bs, Lx, torch.Generator().manual_seed(9000 + Lx + i))
        p = run_table(tab, Ph, x).argmax(-1)
        c += (p == y).sum().item(); t += y.numel()
    return c / t
res["cert64"] = round(acc_at(64, 24, 3), 4)
for Lx in (256, 1024, 4096): res[str(Lx)] = round(acc_at(Lx, 4, 3), 4)
out = dict(tag="EXP049-DYCK-SURGERY", ckpt=CKPT, acc=res, table=tab,
           certified=res["cert64"] == 1.0,
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
           wall_s=round(time.time() - t0, 1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
