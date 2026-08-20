"""ARC-2 cycle 7 M2: certified KR router + multi-organ mount (division + addition).
Marker-free in-band syntax inside Markov text streams (distractor digit runs allowed):
  DIV span: [N digits][SLASH][d digits][EQ][q digits zfill][rem][DEND]   (organ: div_t7.pt)
  ADD span: [PLUS][a1 b1 a2 b2 ... LSB pairs][EQ][sum LSB..][DEND]        (organ: trained here)
Gate = 5-state table machine (TEXT/div-q/div-ans/add-q/add-ans), direct supervision,
L-DIRECT-GRADIENT crystallization; organs self-terminate (L-GATE-EXTENT respected:
no counting anywhere — DEND flips the gate home). Host: RoPE TF, two arms:
TF-FULL (plain LM) and HOST-M (text-only loss + neural gate head as control).
"""
import json, math, random, resource, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); t0 = time.time()

VW, D0, SLASH, PLUS, EQ, DEND = 32, 32, 42, 43, 44, 45
V = 46
TEXT, DIV, ADD = 0, 2, 3
dig = lambda c: D0 + int(c)

# ---------------------------------------------- P0a: division organ (frozen)
ck = torch.load("div_t7.pt", map_location="cpu")
Tdi, Tri = ck["Td"].argmax(-1).numpy(), ck["Tr"].argmax(-1).numpy()
Hqi, Hmi = ck["Hq"].argmax(-1).numpy(), ck["Hm"].argmax(-1).numpy()
def div_tape(N_str, d):
    toks = [10 + d - 2] + [int(c) for c in N_str] + [21]
    dd = r = 0; q = []; rem = None
    for tok in toks:
        if int(Hmi[tok]) == 1:
            o = int(Hqi[tok, dd, r])
            if o <= 9: q.append(o)
            else: rem = o - 10
        ndd = int(Tdi[tok, dd]); nr = int(Tri[tok, dd, r]); dd, r = ndd, nr  # OLD-state
    return [dig(c) for c in q] + [dig(rem // 10), dig(rem % 10), DEND]

# ---------------------------------------------- P0b: addition organ (trained)
def train_addorgan(seed=0):
    torch.manual_seed(seed)
    Tc = nn.Parameter(0.1 * torch.randn(100, 2, 2))    # pair x carry -> carry'
    Hs = nn.Parameter(0.1 * torch.randn(100, 2, 10))   # pair x carry -> sum digit
    opt = torch.optim.AdamW([Tc, Hs], lr=2e-2); rng = random.Random(seed + 50)
    for step in range(1, 1201):
        rows = []
        for _ in range(64):
            k = rng.randrange(1, 5)
            a = rng.randrange(10 ** k); b = rng.randrange(10 ** k)
            c = 0
            for i in range(k):                          # LSB-first pairs
                pa, pb = (a // 10**i) % 10, (b // 10**i) % 10
                s = (pa + pb + c) % 10; nc = (pa + pb + c) // 10
                rows.append((pa * 10 + pb, c, nc, s)); c = nc
        r_ = torch.tensor(rows)
        p, c0, c1, s = r_[:, 0], r_[:, 1], r_[:, 2], r_[:, 3]
        loss = (F.cross_entropy(Tc[p, c0], c1) + F.cross_entropy(Hs[p, c0], s))
        loss.backward(); opt.step(); opt.zero_grad()
    return Tc.argmax(-1).numpy(), Hs.argmax(-1).numpy()
Tci, Hsi = train_addorgan()
def add_tape(a, b):
    k = max(len(str(a)), len(str(b))); c = 0; out = []
    for i in range(k):
        pa, pb = (a // 10**i) % 10, (b // 10**i) % 10
        out.append(int(Hsi[pa * 10 + pb, c])); c = int(Tci[pa * 10 + pb, c])
    if c: out.append(c)
    return [dig(x) for x in out] + [DEND]

rng = random.Random(21); bad = 0
for _ in range(300):
    d = rng.randrange(2, 13); nd = rng.choice([rng.randrange(1, 9), rng.randrange(80, 151)])
    N = rng.randrange(10 ** (nd - 1), 10 ** nd)
    want = [dig(c) for c in str(N // d).zfill(nd)] + [dig(c) for c in f"{N % d:02d}"] + [DEND]
    bad += div_tape(str(N), d) != want
dbad = bad
for _ in range(200):
    k = rng.randrange(20, 101); A = rng.randrange(10 ** (k - 1), 10 ** k); B = rng.randrange(10 ** k)
    digits = [int(t) - D0 for t in add_tape(A, B)[:-1]]
    s = sum(x * 10**i for i, x in enumerate(digits))
    dbad += s != A + B
print(f"[P0] organs: div remount {300-bad}/300 · add certified {200-(dbad-bad)}/200 (40-100-digit carries)", flush=True)
assert dbad == 0

# ------------------------------------------------------------- stream builder
mr = random.Random(7); chain = {}
for a0 in range(VW):
    for b0 in range(VW):
        chain[(a0, b0)] = [mr.expovariate(1.0) for _ in range(VW)]
def text_tokens(rng, n):
    a, b = rng.randrange(VW), rng.randrange(VW); out = [a, b]
    for _ in range(n - 2):
        nxt = rng.choices(range(VW), weights=chain[(a, b)])[0]; out.append(nxt); a, b = b, nxt
    return out

def build_stream(rng, nmax, n_spans=2):
    toks, cls, sph = [], [], []          # sph = (kind, meta) per span
    for _ in range(n_spans):
        toks += text_tokens(rng, rng.randrange(5, 10)); cls += [TEXT] * (len(toks) - len(cls))
        if rng.random() < 0.4:            # distractor digit run (stays TEXT)
            k = rng.randrange(1, 7); n0 = rng.randrange(10 ** k)
            toks += [dig(c) for c in str(n0)]; cls += [TEXT] * len(str(n0))
        if rng.random() < 0.5:            # DIV span
            d = rng.randrange(2, 13); nd = rng.randrange(1, nmax + 1)
            N = rng.randrange(10 ** (nd - 1), 10 ** nd) if nd > 1 else rng.randrange(1, 10)
            q = [dig(c) for c in str(N // d).zfill(nd)]
            toks += [dig(c) for c in str(N)] + [SLASH] + [dig(c) for c in str(d)] + [EQ]
            toks += q + [dig(c) for c in f"{N % d:02d}"] + [DEND]
            cls += [TEXT] * (nd + 2 + len(str(d))) + [DIV] * (nd + 3)
            sph.append(("div", N, d))
        else:                             # ADD span
            k = rng.randrange(1, nmax + 1)
            A = rng.randrange(10 ** (k - 1), 10 ** k) if k > 1 else rng.randrange(1, 10)
            B = rng.randrange(10 ** k)
            toks += [PLUS]; cls += [TEXT]
            for i in range(k):
                toks += [dig(str((A // 10**i) % 10)), dig(str((B // 10**i) % 10))]; cls += [TEXT, TEXT]
            st = str(A + B)[::-1]
            toks += [EQ] + [dig(c) for c in st] + [DEND]
            cls += [TEXT] + [ADD] * (len(st) + 1)
            sph.append(("add", A, B))
    toks += text_tokens(rng, rng.randrange(5, 10)); cls += [TEXT] * (len(toks) - len(cls))
    return toks, cls, sph

r2 = random.Random(4); okall = True
for _ in range(200):
    toks, cls, sph = build_stream(r2, 8)
    okall &= all(c in (0, 2, 3) for c in cls) and len(toks) == len(cls)
    okall &= all((c == TEXT) or (32 <= t < 42) or (t == DEND) for t, c in zip(toks, cls))
    okall &= all((c != TEXT) <= (t != EQ) for t, c in zip(toks, cls))
    for kind, X, Y in sph:
        if kind == "div": okall &= div_tape(str(X), Y) == [dig(c) for c in str(X // Y).zfill(len(str(X)))] + [dig(c) for c in f"{X % Y:02d}"] + [DEND]
        else:
            st = str(X + Y)[::-1]
            okall &= add_tape(X, Y) == [dig(c) for c in st] + [DEND]
print(f"[P0] stream+organ oracle: {'OK' if okall else 'BROKEN'}", flush=True)
assert okall

# ------------------------------------------------- P1: KR gate (table machine)
def phase_ref(toks, cls):
    s = [0]
    for t, c in zip(toks[:-1], cls[1:]):
        cur = s[-1]
        if cur == 0:
            ns = 1 if t == SLASH else (3 if t == PLUS else 0)
        elif cur == 1:
            ns = 2 if t == EQ else 1
        elif cur == 2:
            ns = 0 if t == DEND else 2
        elif cur == 3:
            ns = 4 if t == EQ else 3
        else:
            ns = 0 if t == DEND else 4
        s.append(ns)
    return s

def train_gate(seed=0, steps=1500):
    torch.manual_seed(seed)
    Ts = nn.Parameter(0.1 * torch.randn(V, 5, 5)); Th = nn.Parameter(0.1 * torch.randn(V, 5, 4))
    opt = torch.optim.AdamW([Ts, Th], lr=2e-2); rng = random.Random(seed + 60)
    for step in range(1, steps + 1):
        rows = []
        for _ in range(24):
            toks, cls, _ = build_stream(rng, 8)
            ph = phase_ref(toks, cls)
            rows += list(zip(toks[:-1], ph[:-1], ph[1:], cls[1:]))
        r_ = torch.tensor(rows)
        tk, s0, s1, c1 = r_[:, 0], r_[:, 1], r_[:, 2], r_[:, 3]
        loss = F.cross_entropy(Ts[tk, s0], s1) + F.cross_entropy(Th[tk, s0], c1)
        loss.backward(); opt.step(); opt.zero_grad()
        if step % 500 == 0: print(f"[gate] {step}/{steps} CE {loss.item():.5f}", flush=True)
    return Ts.argmax(-1).numpy(), Th.argmax(-1).numpy()
Tsi, Thi = train_gate()

def gate_route(toks):
    s = 0; out = []
    for t in toks:
        c = int(Thi[t, s]); out.append(c); s = int(Tsi[t, s])
    return out, s         # out[k] = predicted class of token k+1 (decision after consuming token k)

# ---- cycle-3-style index-collision audit: is any (tok, state) cell ambiguous?
def gate_audit():
    from collections import defaultdict, Counter
    cells = defaultdict(Counter); cellT = defaultdict(Counter); rng = random.Random(99)
    for _ in range(500):
        toks, cls, _ = build_stream(rng, rng.choice([8, 150]))
        ph = phase_ref(toks, cls)
        for tk, s0, s1, c1 in zip(toks[:-1], ph[:-1], ph[1:], cls[1:]):
            cells[(tk, s0)][c1] += 1; cellT[(tk, s0)][s1] += 1
    amb = [(k, dict(v)) for k, v in cells.items() if len(v) > 1] + [(k, dict(v)) for k, v in cellT.items() if len(v) > 1]
    under = [k for k, v in cells.items() if len(v) == 1 and int(Thi[k[0], k[1]]) != next(iter(v))]
    under += [k for k, v in cellT.items() if len(v) == 1 and int(Tsi[k[0], k[1]]) != next(iter(v))]
    print(f"[audit] visited cells {len(cells)} · ambiguous(Th+Ts) {len(amb)} · argmax-mismatch {len(under)}", flush=True)
    for k, v in amb[:6]: print(f"[audit]   AMBIG {k}: {v}", flush=True)
    for k in under[:6]: print(f"[audit]   UNDER {k}", flush=True)
    return len(amb), len(under)
n_amb, n_under = gate_audit()

errs = tot = 0
rng = random.Random(31)
for _ in range(300):
    nmax = rng.choice([8, 150])
    toks, cls, _ = build_stream(rng, nmax)
    g, sf = gate_route(toks)
    errs += sum(a != b for a, b in zip(g[:-1], cls[1:])); tot += len(cls) - 1
print(f"[P1] KR gate certified: {tot-errs}/{tot} routing decisions exact "
      f"(incl. 150-digit streams){' · CERTIFIED' if errs == 0 else ' · FAILED'}", flush=True)
gate_cert = errs == 0
sweep = []
for sd in (1, 2):
    Tsi2, Thi2 = train_gate(seed=sd, steps=1000)
    rng2 = random.Random(300 + sd); e2 = 0
    for _ in range(150):
        toks, cls, _ = build_stream(rng2, rng.choice([8, 150]))
        g2 = []; s = 0
        for t in toks:
            g2.append(int(Thi2[t, s])); s = int(Tsi2[t, s])
        e2 += sum(a != b for a, b in zip(g2[:-1], cls[1:]))
    sweep.append(e2 == 0)
print(f"[P1] seed sweep (1,2): {[('0err' if x else 'ERR') for x in sweep]}", flush=True)

# ---------------------------------------------------------- host (RoPE TF)
class RopeTF(nn.Module):
    def __init__(s, d=64, L=2, H=4):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.pos_gain = nn.Parameter(torch.ones(1))
        s.ln_f = nn.LayerNorm(d); s.head = nn.Linear(d, V, bias=False); s.gate = nn.Linear(d, 4)
        s.blocks = nn.ModuleList([nn.ModuleDict(dict(
            ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), po=nn.Linear(d, d),
            ln2=nn.LayerNorm(d), fc=nn.Linear(d, 4 * d), fc2=nn.Linear(4 * d, d))) for _ in range(L)])
        s.hd = d // H; s.H = H
        s.register_buffer("inv", 1.0 / (10000 ** (torch.arange(0, s.hd, 2).float() / s.hd)))
    def rope(s, x):
        L = x.shape[-2]; f = torch.outer(torch.arange(L, device=x.device).float(), s.inv)
        c, s_ = f.cos(), f.sin()
        x1, x2 = x[..., :s.hd//2], x[..., s.hd//2:]
        return torch.cat([x1 * c - x2 * s_, x2 * c + x1 * s_], -1)
    def forward(s, x):
        B, L = x.shape; h = s.emb(x); m = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        for b in s.blocks:
            r = b["ln1"](h); qkv = b["qkv"](r).view(B, L, s.H, 3 * s.hd).permute(0, 2, 1, 3)
            q, k, v = qkv.chunk(3, -1); q, k = s.rope(q), s.rope(k)
            a = ((q @ k.transpose(-1, -2)) / math.sqrt(s.hd)).masked_fill(~m, float("-inf")).softmax(-1) @ v
            h = h + b["po"](a.transpose(1, 2).reshape(B, L, -1))
            h = h + b["fc2"](F.gelu(b["fc"](b["ln2"](h))))
        h = s.ln_f(h); return s.head(h) * s.pos_gain, s.gate(h)

def train_host(mode, steps=2500, seed=0):
    torch.manual_seed(seed); mdl = RopeTF()
    opt = torch.optim.AdamW(mdl.parameters(), lr=3e-3); rng = random.Random(seed + 70)
    for step in range(1, steps + 1):
        rows = [build_stream(rng, 8) for _ in range(24)]
        Lm = max(len(t) for t, _, _ in rows)
        x = torch.zeros(24, Lm - 1, dtype=torch.long); y = torch.zeros_like(x)
        g = torch.zeros_like(x); vm = torch.zeros_like(x, dtype=torch.bool)
        for i, (t, c, _) in enumerate(rows):
            n = len(t) - 1
            x[i, :n] = torch.tensor(t[:-1]); y[i, :n] = torch.tensor(t[1:]); g[i, :n] = torch.tensor(c[1:])
            vm[i, :n] = True
        lg, gt = mdl(x)
        ce = (F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="none").view(y.shape) * vm).sum() / vm.sum()
        if mode == "tf": loss = ce
        else:
            tm = vm & (g == TEXT)
            loss = (ce * tm).sum() / tm.sum() + F.cross_entropy(
                gt.reshape(-1, 4)[vm.reshape(-1)], g.reshape(-1)[vm.reshape(-1)])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0: print(f"[host {mode}] {step}/{steps} loss {loss.item():.4f}", flush=True)
    return mdl

tf_full = train_host("tf"); host_m = train_host("msk")

# ------------------------------------------------------------------- eval
def make_eval(rng, nmax, B=40):
    rows = []
    for i in range(B):
        kind = "div" if i % 2 == 0 else "add"
        if kind == "div":
            d = rng.randrange(2, 13); nd = nmax
            N = rng.randrange(10 ** (nd - 1), 10 ** nd)
            pre = text_tokens(rng, 8)
            q = [dig(c) for c in str(N // d).zfill(nd)]
            toks = pre + [dig(c) for c in str(N)] + [SLASH] + [dig(c) for c in str(d)] + [EQ] + q + [dig(c) for c in f"{N % d:02d}"] + [DEND]
            cls = [TEXT] * (8 + nd + 2 + len(str(d))) + [DIV] * (nd + 3)
            cut = 8 + nd + 2 + len(str(d)) + 1; tape = div_tape(str(N), d); want = q + [dig(c) for c in f"{N % d:02d}"] + [DEND]
        else:
            k = nmax; A = rng.randrange(10 ** (k - 1), 10 ** k); Bn = rng.randrange(10 ** k)
            pre = text_tokens(rng, 8); st = str(A + Bn)[::-1]
            toks = pre + [PLUS]
            for j in range(k): toks += [dig(str((A // 10**j) % 10)), dig(str((Bn // 10**j) % 10))]
            toks += [EQ] + [dig(c) for c in st] + [DEND]
            cls = [TEXT] * (8 + 1 + 2 * k + 1) + [ADD] * (len(st) + 1)
            cut = 8 + 1 + 2 * k + 1; tape = add_tape(A, Bn); want = [dig(c) for c in st] + [DEND]
        toks += text_tokens(rng, 6); cls += [TEXT] * 6
        rows.append((kind, toks, cls, cut, tape, want))
    return rows

@torch.no_grad()
def complete(mdl, rows, router):
    ex = {"div": [0, 0], "add": [0, 0]}; first_miss = 0
    for kind, toks, cls, cut, tape, want in rows:
        seq = list(toks[:cut]); k = 0
        while k < len(tape) + 6:
            lg, gt = mdl(torch.tensor([seq]))
            if router == "tf":
                nxt = int(lg[0, -1].argmax())
            elif router == "kr":
                g, _ = gate_route(seq)
                nxt = tape[k] if (g[-1] in (2, 3) and k < len(tape)) else int(lg[0, -1].argmax())
            else:  # neural, commit-once
                c = int(gt[0, -1].argmax())
                if k == 0 and c == 0: first_miss += 1
                nxt = tape[k] if (c in (2, 3) and k < len(tape)) else int(lg[0, -1].argmax())
            seq.append(nxt); k += 1
            if nxt == DEND: break
        good = bool(seq[cut:cut + len(want)] == want) and len(seq) > cut and seq[cut + len(want) - 1:cut + len(want)] == [DEND]
        ex[kind][0] += good; ex[kind][1] += 1
    return ex, first_miss

@torch.no_grad()
def txt_ce(mdl, rows):
    Lm = max(len(t) for _, t, _, _, _, _ in rows); tot = n = 0.0
    for kind, toks, cls, *_ in rows:
        assert len(toks) == len(cls), (len(toks), len(cls))
        x = torch.tensor([toks]); lg, _ = mdl(x)
        ce = F.cross_entropy(lg[0, :-1], x[0, 1:], reduction="none")
        m = torch.tensor([c == TEXT for c in cls[1:]])
        assert ce.shape[0] == m.shape[0], (ce.shape, m.shape)
        tot += ce[m].sum().item(); n += m.sum().item()
    return tot / n

print("\n[P3] EVAL — span exact % (greedy), per organ; text CE; router errors")
print(f"{'nd':>5} {'span':>5} {'TF-FULL':>8} {'HY-KR':>7} {'HY-neu':>7} {'txtCE f/m':>10}")
cert = gate_cert; rows_log = []
for nd in [8, 40, 100, 150]:
    rows = make_eval(random.Random(2000 + nd), nd)
    e_tf, _ = complete(tf_full, rows, "tf")
    e_kr, _ = complete(host_m, rows, "kr")
    e_nu, fms = complete(host_m, rows, "neu")
    tcf, tcm = txt_ce(tf_full, rows), txt_ce(host_m, rows)
    for kind in ("div", "add"):
        a, b = e_tf[kind]; c, d = e_kr[kind]; e, f = e_nu[kind]
        print(f"{nd:>5} {kind:>5} {a/b:>8.1%} {c/d:>7.1%} {e/f:>7.1%} "
              f"{tcf:>5.3f}/{tcm:>5.3f}" if kind == "div" else
              f"{'':>5} {kind:>5} {a/b:>8.1%} {c/d:>7.1%} {e/f:>7.1%}", flush=True)
        rows_log.append(dict(nd=nd, kind=kind, tf=a/b, kr=c/d, neu=e/f))
        cert &= (c / d == 1.0)          # certification = KR-routed hybrid exact
    rows_log[-2]["neu_first_miss"] = fms   # neural control's failure = the finding, not a cert term
    rows_log[-2]["txt_ce_full"] = tcf; rows_log[-2]["txt_ce_host"] = tcm

res = dict(tag="ARC2-C7-P11M2", certified=bool(cert), kr_gate=gate_cert, seed_sweep=sweep,
           rows=rows_log, wall_s=round(time.time() - t0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print("RESULT " + json.dumps(res), flush=True)
open("log.jsonl", "a").write(json.dumps(res) + "\n")
torch.save({"Ts": torch.tensor(Tsi), "Th": torch.tensor(Thi)}, "krgate_s0.pt")
torch.save({"Tc": torch.tensor(Tci), "Hs": torch.tensor(Hsi)}, "addorgan_s0.pt")
