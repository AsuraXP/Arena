"""ARC-2 cycle 7 M3: mount the ITERATED multiplication organ (IFT, ift_t3.pt) behind
the certified KR router — three-organ dispatch (add + mul + div) in one token host.
Mul span (in-band): [a MSB][STAR][b MSB][EQ][product MSB][DEND]; organ runs O(N)
internal passes (fixed-point) inside a single-pass span surface.
Gate: 7 states (text / {div,mul,add}-q / {div,mul,add}-ans), 4 organ classes,
direct supervision, dual-table audit (Th+Ts), 3-seed sweep. Host arms: TF-FULL vs
HOST-M(+neural gate control). Discipline identical to M2.
"""
import json, math, random, resource, time
import torch, torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0); t0 = time.time()

VW, D0, SLASH, PLUS, EQ, DEND, STAR = 32, 32, 42, 43, 44, 45, 46
V = 47
TEXT, DIV, ADD, MUL = 0, 2, 3, 4
dig = lambda c: D0 + int(c)

# ------------------------------------------------ P0a: IFT organ remount
IFT_SEP, IFT_ENDT, IFT_NULL = 10, 111, 112
IFT_PAIR = lambda a, r: 11 + a * 10 + r
unpair = lambda t: ((t - 11) // 10, (t - 11) % 10)
IFT_EMIT = lambda d: 113 + d
DRAIN, UNSET = 10, 11
TY_NULL, TY_COPY, TY_SEP, TY_PAIR, TY_EMIT = range(5)
def assemble(ty1, od, oa, ty2, tok):
    s1 = (IFT_NULL if ty1 == TY_NULL else tok if ty1 == TY_COPY else
          IFT_SEP if ty1 == TY_SEP else IFT_PAIR(oa, od) if ty1 == TY_PAIR else IFT_EMIT(od))
    return s1, (IFT_ENDT if ty2 == 1 else IFT_NULL)

ck = torch.load("ift_t3.pt", map_location="cpu")
Tm = ck["Tmult"].argmax(-1).numpy(); Tc = ck["Tc"].argmax(-1).numpy()
Ta = ck["Ta"].argmax(-1).numpy();    Tf = ck["Tfp"].argmax(-1).numpy()
Ht = ck["Hty"].argmax(-1).numpy();   Hd = ck["Hd"].argmax(-1).numpy()
Ha = ck["Ha"].argmax(-1).numpy();    H2 = ck["H2"].argmax(-1).numpy()

def ift_pass(tape):
    out = []; mult, c, ap, fp = UNSET, 0, 0, 0
    for tok in tape:
        nm = int(Tm[tok, mult]); nc = int(Tc[tok, mult, c])
        na = int(Ta[tok, ap]);   nfp = int(Tf[tok, fp])
        ty1 = int(Ht[tok, mult, fp]); od = int(Hd[tok, nm, c])
        oa = int(Ha[tok, ap]);   ty2 = int(H2[tok, nm])
        s1, s2 = assemble(ty1, od, oa, ty2, tok)
        out += [s1, s2]
        mult, c, ap, fp = nm, nc, na, nfp
    return out

def mul_str(A, B, max_passes=400):
    ad = [int(d) for d in str(A)[::-1]]; bd = [int(d) for d in str(B)[::-1]]
    np_ = len(ad) + 2; ad = ad + [0] * (np_ - len(ad))
    tape = bd + [IFT_SEP] + [IFT_PAIR(a, 0) for a in ad] + [IFT_ENDT]
    emitted = []
    for _ in range(max_passes):
        drain = tape[0] == IFT_SEP
        raw = ift_pass(tape); nxt = []
        for s in raw:
            if s == IFT_NULL: continue
            if s >= 113: emitted.append(s - 113)
            else: nxt.append(s)
        if drain: break
        tape = nxt
    return "".join(map(str, emitted[::-1])).lstrip("0") or "0"

rng = random.Random(41); bad = 0
for _ in range(300):
    na = rng.choice([rng.randrange(1, 9), rng.randrange(25, 51)])
    nb = rng.choice([rng.randrange(1, 9), rng.randrange(25, 51)])
    A = rng.randrange(10 ** (na - 1), 10 ** na); B = rng.randrange(10 ** (nb - 1), 10 ** nb)
    bad += mul_str(A, B) != str(A * B)
for _ in range(20):                                  # judge-scale 40x40
    A = rng.randrange(10 ** 39, 10 ** 40); B = rng.randrange(10 ** 39, 10 ** 40)
    bad += mul_str(A, B) != str(A * B)
print(f"[P0] IFT organ remount: {320-bad}/320 exact (1-50 digit operands + 40x40)", flush=True)
assert bad == 0

# --------------------------------------------- P0b: div/add organs (from M2)
ckd = torch.load("div_t7.pt", map_location="cpu")
Tdi, Tri = ckd["Td"].argmax(-1).numpy(), ckd["Tr"].argmax(-1).numpy()
Hqi, Hmi = ckd["Hq"].argmax(-1).numpy(), ckd["Hm"].argmax(-1).numpy()
def div_tape(N_str, d):
    toks = [10 + d - 2] + [int(c) for c in N_str] + [21]
    dd = r = 0; q = []; rem = None
    for tok in toks:
        if int(Hmi[tok]) == 1:
            o = int(Hqi[tok, dd, r])
            if o <= 9: q.append(o)
            else: rem = o - 10
        ndd = int(Tdi[tok, dd]); nr = int(Tri[tok, dd, r]); dd, r = ndd, nr
    return [dig(c) for c in q] + [dig(rem // 10), dig(rem % 10), DEND]

cka = torch.load("addorgan_s0.pt", map_location="cpu")
Tci, Hsi = cka["Tc"].numpy(), cka["Hs"].numpy()   # already-hard tables (saved post-argmax in M2)
def add_tape(a, b):
    k = max(len(str(a)), len(str(b))); c = 0; out = []
    for i in range(k):
        pa, pb = (a // 10**i) % 10, (b // 10**i) % 10
        out.append(int(Hsi[pa * 10 + pb, c])); c = int(Tci[pa * 10 + pb, c])
    if c: out.append(c)
    return [dig(x) for x in out] + [DEND]

rng = random.Random(43); dbad = 0
for _ in range(150):
    d = rng.randrange(2, 13); nd = rng.choice([rng.randrange(1, 9), rng.randrange(80, 151)])
    N = rng.randrange(10 ** (nd - 1), 10 ** nd)
    want = [dig(c) for c in str(N // d).zfill(nd)] + [dig(c) for c in f"{N % d:02d}"] + [DEND]
    dbad += div_tape(str(N), d) != want
for _ in range(150):
    k = rng.randrange(20, 101); A = rng.randrange(10 ** (k - 1), 10 ** k); B = rng.randrange(10 ** k)
    dgs = [int(t) - D0 for t in add_tape(A, B)[:-1]]
    dbad += sum(x * 10**i for i, x in enumerate(dgs)) != A + B
print(f"[P0] div/add organs remount: {300-dbad}/300 exact", flush=True)
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
    toks, cls = [], []
    def sync(): 
        cls.extend([TEXT] * (len(toks) - len(cls)))
    for _ in range(n_spans):
        toks += text_tokens(rng, rng.randrange(5, 10)); sync()
        if rng.random() < 0.4:
            k = rng.randrange(1, 7); n0 = rng.randrange(10 ** k)
            toks += [dig(c) for c in str(n0)]; sync()
        u = rng.random()
        if u < 0.33:                                   # DIV
            d = rng.randrange(2, 13); nd = rng.randrange(1, nmax + 1)
            N = rng.randrange(10 ** (nd - 1), 10 ** nd) if nd > 1 else rng.randrange(1, 10)
            ans = ([dig(c) for c in str(N // d).zfill(nd)]
                   + [dig(c) for c in f"{N % d:02d}"] + [DEND])
            toks += [dig(c) for c in str(N)] + [SLASH] + [dig(c) for c in str(d)] + [EQ]
            sync(); toks += ans; cls += [DIV] * len(ans)
        elif u < 0.66:                                 # ADD
            k = rng.randrange(1, nmax + 1)
            A = rng.randrange(10 ** (k - 1), 10 ** k) if k > 1 else rng.randrange(1, 10)
            B = rng.randrange(10 ** k)
            toks += [PLUS]
            for i in range(k):
                toks += [dig(str((A // 10**i) % 10)), dig(str((B // 10**i) % 10))]
            st = str(A + B)[::-1]
            ans = [dig(c) for c in st] + [DEND]
            toks += [EQ]; sync(); toks += ans; cls += [ADD] * len(ans)
        else:                                          # MUL
            ka = rng.randrange(1, nmax + 1); kb = rng.randrange(1, nmax + 1)
            A = rng.randrange(10 ** (ka - 1), 10 ** ka) if ka > 1 else rng.randrange(1, 10)
            B = rng.randrange(10 ** (kb - 1), 10 ** kb) if kb > 1 else rng.randrange(1, 10)
            p = mul_str(A, B)
            ans = [dig(c) for c in p] + [DEND]
            toks += [dig(c) for c in str(A)] + [STAR] + [dig(c) for c in str(B)] + [EQ]
            sync(); toks += ans; cls += [MUL] * len(ans)
    toks += text_tokens(rng, rng.randrange(5, 10)); sync()
    return toks, cls

r2 = random.Random(44); okall = True
for _ in range(250):
    toks, cls = build_stream(r2, 8)
    okall &= all(c in (0, 2, 3, 4) for c in cls) and len(toks) == len(cls)
    okall &= all((c == TEXT) or (32 <= t < 42) or (t == DEND) for t, c in zip(toks, cls))
print(f"[P0] stream oracle (len-aligned, organ tokens=digits/DEND only): "
      f"{'OK' if okall else 'BROKEN'}", flush=True)
assert okall

# ------------------------------------------------- P1: KR gate (7 states)
def phase_ref(toks, cls):
    s = [0]
    for t in toks[:-1]:
        cur = s[-1]
        if cur == 0:      ns = {SLASH: 1, PLUS: 3, STAR: 5}.get(t, 0)
        elif cur == 1:    ns = 2 if t == EQ else 1
        elif cur == 2:    ns = 0 if t == DEND else 2
        elif cur == 3:    ns = 4 if t == EQ else 3
        elif cur == 4:    ns = 0 if t == DEND else 4
        elif cur == 5:    ns = 6 if t == EQ else 5
        else:             ns = 0 if t == DEND else 6
        s.append(ns)
    return s

def train_gate(seed=0, steps=1500):
    torch.manual_seed(seed)
    Ts = nn.Parameter(0.1 * torch.randn(V, 7, 7)); Th = nn.Parameter(0.1 * torch.randn(V, 7, 5))
    opt = torch.optim.AdamW([Ts, Th], lr=2e-2); rng = random.Random(seed + 60)
    for step in range(1, steps + 1):
        rows = []
        for _ in range(24):
            toks, cls = build_stream(rng, 8)
            ph = phase_ref(toks, cls)
            rows += list(zip(toks[:-1], ph[:-1], ph[1:], cls[1:]))
        r_ = torch.tensor(rows)
        tk, s0, s1, c1 = r_[:, 0], r_[:, 1], r_[:, 2], r_[:, 3]
        loss = F.cross_entropy(Ts[tk, s0], s1) + F.cross_entropy(Th[tk, s0], c1)
        loss.backward(); opt.step(); opt.zero_grad()
        if step % 500 == 0: print(f"[gate] {step}/{steps} CE {loss.item():.5f}", flush=True)
    return Ts.argmax(-1).numpy(), Th.argmax(-1).numpy()
Tsi, Thi = train_gate()

def gate_route(toks, Tsi=None, Thi=None):
    Tsi = Tsi if Tsi is not None else globals()["Tsi"]; Thi = Thi if Thi is not None else globals()["Thi"]
    s = 0; out = []
    for t in toks:
        out.append(int(Thi[t, s])); s = int(Tsi[t, s])
    return out, s     # out[k] = predicted class of token k+1

from collections import defaultdict, Counter
def gate_audit():
    cth, cts = defaultdict(Counter), defaultdict(Counter); rng = random.Random(99)
    for _ in range(500):
        toks, cls = build_stream(rng, rng.choice([8, 150]))
        ph = phase_ref(toks, cls)
        for tk, s0, s1, c1 in zip(toks[:-1], ph[:-1], ph[1:], cls[1:]):
            cth[(tk, s0)][c1] += 1; cts[(tk, s0)][s1] += 1
    amb = [k for k in (list(cth) + list(cts)) if len(cth.get(k, {})) > 1 or len(cts.get(k, {})) > 1]
    under = [k for k, v in cth.items() if len(v) == 1 and int(Thi[k[0], k[1]]) != next(iter(v))]
    under += [k for k, v in cts.items() if len(v) == 1 and int(Tsi[k[0], k[1]]) != next(iter(v))]
    print(f"[audit] visited cells {len(cth)}·{len(cts)} · ambiguous {len(amb)} · mismatch {len(under)}", flush=True)
    return len(amb) + len(under)
gate_audit()

errs = tot = 0; rng = random.Random(31)
for _ in range(300):
    toks, cls = build_stream(rng, rng.choice([8, 150]))
    g, _ = gate_route(toks)
    errs += sum(a != b for a, b in zip(g[:-1], cls[1:])); tot += len(cls) - 1
print(f"[P1] KR gate certified: {tot-errs}/{tot} exact "
      f"{'CERTIFIED' if errs == 0 else 'FAILED'}", flush=True)
gate_cert = errs == 0
sweep = []
for sd in (1, 2):
    Tg, Hg = train_gate(seed=sd, steps=1000)          # NOTE: do not shadow organ tables
    rng2 = random.Random(300 + sd); e2 = 0
    for _ in range(150):
        toks, cls = build_stream(rng2, 8)
        g2, _ = gate_route(toks, Tg, Hg)
        e2 += sum(a != b for a, b in zip(g2[:-1], cls[1:]))
    sweep.append(e2 == 0)
print(f"[P1] seed sweep (1,2): {sweep}", flush=True)

# ---------------------------------------------------------- host (RoPE TF)
class RopeTF(nn.Module):
    def __init__(s, d=64, L=2, H=4):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.pos_gain = nn.Parameter(torch.ones(1))
        s.ln_f = nn.LayerNorm(d); s.head = nn.Linear(d, V, bias=False); s.gate = nn.Linear(d, 5)
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
        Lm = max(len(t) for t, _ in rows)
        x = torch.zeros(24, Lm - 1, dtype=torch.long); y = torch.zeros_like(x)
        g = torch.zeros_like(x); vm = torch.zeros_like(x, dtype=torch.bool)
        for i, (t, c) in enumerate(rows):
            n = len(t) - 1
            x[i, :n] = torch.tensor(t[:-1]); y[i, :n] = torch.tensor(t[1:]); g[i, :n] = torch.tensor(c[1:])
            vm[i, :n] = True
        lg, gt = mdl(x)
        ce = (F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="none").view(y.shape) * vm).sum() / vm.sum()
        if mode == "tf": loss = ce
        else:
            tm = vm & (g == TEXT)
            loss = (ce * tm).sum() / tm.sum() + F.cross_entropy(
                gt.reshape(-1, 5)[vm.reshape(-1)], g.reshape(-1)[vm.reshape(-1)])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0: print(f"[host {mode}] {step}/{steps} loss {loss.item():.4f}", flush=True)
    return mdl

tf_full = train_host("tf"); host_m = train_host("msk")

# ------------------------------------------------------------------- eval
ORGS = {"div": [8, 40, 100, 150], "add": [8, 40, 100, 150], "mul": [8, 15, 25, 40]}
def make_eval(rng, kind, w, B=40):
    rows = []
    for i in range(B):
        pre = text_tokens(rng, 8)
        if kind == "div":
            d = rng.randrange(2, 13); N = rng.randrange(10 ** (w - 1), 10 ** w)
            q = [dig(c) for c in str(N // d).zfill(w)]
            quest = [dig(c) for c in str(N)] + [SLASH] + [dig(c) for c in str(d)] + [EQ]
            tape = div_tape(str(N), d); want = q + [dig(c) for c in f"{N % d:02d}"] + [DEND]
        elif kind == "add":
            A = rng.randrange(10 ** (w - 1), 10 ** w); Bn = rng.randrange(10 ** w)
            st = str(A + Bn)[::-1]
            quest = [PLUS]
            for j in range(w): quest += [dig(str((A // 10**j) % 10)), dig(str((Bn // 10**j) % 10))]
            quest += [EQ]
            tape = add_tape(A, Bn); want = [dig(c) for c in st] + [DEND]
        else:
            A = rng.randrange(10 ** (w - 1), 10 ** w); Bn = rng.randrange(10 ** (w - 1), 10 ** w)
            p = mul_str(A, Bn)
            quest = [dig(c) for c in str(A)] + [STAR] + [dig(c) for c in str(Bn)] + [EQ]
            tape = [dig(c) for c in p] + [DEND]; want = tape
        cut = len(pre) + len(quest)
        toks = pre + quest + want + text_tokens(rng, 6)
        cls = [TEXT] * cut + [{ "div": DIV, "add": ADD, "mul": MUL }[kind]] * len(want) + [TEXT] * 6
        assert len(toks) == len(cls), (kind, len(toks), len(cls))
        assert toks[cut:cut + len(want)] == want
        rows.append((kind, toks, cls, cut, tape, want))
    return rows

@torch.no_grad()
def complete(mdl, rows, router):
    n_ok = n = 0; first_miss = 0
    for kind, toks, cls, cut, tape, want in rows:
        seq = list(toks[:cut]); k = 0
        while k < len(tape) + 8:
            lg, gt = mdl(torch.tensor([seq]))
            if router == "tf":
                nxt = int(lg[0, -1].argmax())
            elif router == "kr":
                g, _ = gate_route(seq)
                nxt = tape[k] if (g[-1] in (2, 3, 4) and k < len(tape)) else int(lg[0, -1].argmax())
            else:
                c = int(gt[0, -1].argmax())
                if k == 0 and c == 0: first_miss += 1
                nxt = tape[k] if (c in (2, 3, 4) and k < len(tape)) else int(lg[0, -1].argmax())
            seq.append(nxt); k += 1
            if nxt == DEND: break
        good = len(seq) > cut and seq[cut:cut + len(want)] == want
        n_ok += good; n += 1
    return n_ok / n, first_miss

@torch.no_grad()
def txt_ce(mdl, rows):
    tot = n = 0.0
    for kind, toks, cls, *_ in rows:
        assert len(toks) == len(cls)
        x = torch.tensor([toks]); lg, _ = mdl(x)
        ce = F.cross_entropy(lg[0, :-1], x[0, 1:], reduction="none")
        m = torch.tensor([c == TEXT for c in cls[1:]])
        tot += ce[m].sum().item(); n += m.sum().item()
    return tot / n

print("\n[P3] EVAL — span exact % (greedy); TF-FULL / HY-KR / HY-neural")
cert = gate_cert; rows_log = []
for kind in ("div", "add", "mul"):
    for w in ORGS[kind]:
        rows = make_eval(random.Random(3000 + w * 7 + ord(kind[0])), kind, w)
        e_tf, _ = complete(tf_full, rows, "tf")
        e_kr, _ = complete(host_m, rows, "kr")
        e_nu, fms = complete(host_m, rows, "neu")
        tcf, tcm = txt_ce(tf_full, rows), txt_ce(host_m, rows)
        print(f"  {kind:>4} w={w:>3}: TF {e_tf:>7.1%} · KR {e_kr:>7.1%} · neu {e_nu:>7.1%} "
              f"(first-miss {fms}) · txtCE {tcf:.3f}/{tcm:.3f}", flush=True)
        rows_log.append(dict(kind=kind, w=w, tf=e_tf, kr=e_kr, neu=e_nu, fms=fms))
        cert &= (e_kr == 1.0)

res = dict(tag="ARC2-C7-P11M3", certified=bool(cert), kr_gate=gate_cert, seed_sweep=sweep,
           rows=rows_log, wall_s=round(time.time() - t0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print("RESULT " + json.dumps(res), flush=True)
open("log.jsonl", "a").write(json.dumps(res) + "\n")
torch.save({"Ts": torch.tensor(Tsi), "Th": torch.tensor(Thi)}, "krgate3_s0.pt")
