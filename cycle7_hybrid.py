"""ARC-2 cycle 7 (P11 milestone 1): mount the certified division organ inside a
token-prediction host. Mixed streams = order-2 Markov text + marked division spans.
Arms (matched host arch):
  TF-FULL : RoPE host, plain LM on the whole stream (must try to learn division).
  TF-MASK : RoPE host, loss on text only (fluency-parity control; refuses arithmetic).
  HYBRID  : RoPE host (text loss) + learned 3-class gate on host states (direct
            supervision, L-DIRECT-GRADIENT) + frozen organ div_t7.pt owning answer
            tokens. Protocol: [DSPAN][dtok][N digits][ENDIN][q digits][RREM][r][DEND].
Gate classes for predicting token t+1: TEXT=0(host) OP=1(external question, masked)
ORGAN=2(organ tape). Train operand <=8 digits; eval exact-span completion at 8/40/100/150.
"""
import json, math, random, resource, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); t0 = time.time()

# ------------------------------------------------------------------ vocab
VW, D0, DSPAN, DT0, ENDIN, RREM, DEND = 32, 32, 42, 43, 54, 55, 56
V = 57
TEXT, OP, ORGAN = 0, 1, 2
dig = lambda c: D0 + int(c); undig = lambda t: int(t) - D0

# ------------------------------------------- P0: organ remount (oracle-first)
ck = torch.load("div_t7.pt", map_location="cpu")
Tdi, Tri = ck["Td"].argmax(-1).numpy(), ck["Tr"].argmax(-1).numpy()
Hqi, Hmi = ck["Hq"].argmax(-1).numpy(), ck["Hm"].argmax(-1).numpy()

def organ_tape(N_str, d):
    """mounted organ inference -> host answer tokens [q digits (len N, zfill), RREM, r, DEND]"""
    toks = [10 + d - 2] + [int(c) for c in N_str] + [21]
    dd = r = 0; q = []; rem = None
    for tok in toks:
        if int(Hmi[tok]) == 1:
            o = int(Hqi[tok, dd, r])
            if o <= 9: q.append(o)
            else: rem = o - 10
        ndd = int(Tdi[tok, dd]); nr = int(Tri[tok, dd, r])   # L-DETERMINISM(a): compute from OLD state
        dd, r = ndd, nr
    return [D0 + int(c) for c in q] + [RREM, D0 + rem, DEND]

rng = random.Random(11); bad = 0
for _ in range(300):
    d = rng.randrange(2, 13)
    nd = rng.choice([rng.randrange(1, 9), rng.randrange(80, 151)])
    N = rng.randrange(10 ** (nd - 1), 10 ** nd)
    tape = organ_tape(str(N), d)
    want = [D0 + int(c) for c in str(N // d).zfill(nd)] + [RREM, D0 + N % d, DEND]
    bad += tape != want
print(f"[P0] organ remount certified: {300-bad}/300 exact (80-150-digit incl.)", flush=True)
assert bad == 0

# ------------------------------------------------------------- Markov text
mr = random.Random(7)
LOG = {("a","b"): None}
starts = [("a","b")]
chain = {}
for a in range(VW):
    for b in range(VW):
        w = [mr.expovariate(1.0) for _ in range(VW)]
        chain[(a, b)] = w
def text_tokens(rng, n):
    a, b = rng.randrange(VW), rng.randrange(VW)
    out = [a, b]
    for _ in range(n - 2):
        w = chain[(a, b)]; nxt = rng.choices(range(VW), weights=w)[0]
        out.append(nxt); a, b = b, nxt
    return out

def build_stream(rng, nmax, pre=12):
    d = rng.randrange(2, 13); nd = rng.randrange(1, nmax + 1)
    N = rng.randrange(10 ** (nd - 1), 10 ** nd) if nd > 1 else rng.randrange(1, 10)
    toks = text_tokens(rng, pre) + [DSPAN, DT0 + d - 2] + [dig(c) for c in str(N)] + [ENDIN]
    cls = [TEXT] * pre + [OP] * (2 + nd + 1)
    ans = [D0 + int(c) for c in str(N // d).zfill(nd)] + [RREM, D0 + N % d, DEND]
    toks += ans; cls += [ORGAN] * len(ans)
    post = max(4, 24 - 2 * nd)          # fixed total length per operand width
    toks += text_tokens(rng, post); cls += [TEXT] * post
    return toks, cls, (N, d)

r2 = random.Random(3); okall = True
for _ in range(200):
    toks, cls, (N, d) = build_stream(r2, 8)
    nd = len(str(N))
    want = [D0 + int(c) for c in str(N // d).zfill(nd)] + [RREM, D0 + N % d, DEND]
    a = 12 + 2 + nd + 1
    okall &= toks[a: a + nd + 3] == want
print(f"[P0] stream builder oracle: {'OK' if okall else 'BROKEN'}", flush=True)
assert okall

# --------------------------------------------------------------- host model
class RopeTF(nn.Module):
    def __init__(s, d=64, L=2, H=4):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.pos_gain = nn.Parameter(torch.ones(1))
        s.ln_f = nn.LayerNorm(d); s.head = nn.Linear(d, V, bias=False)
        s.gate = nn.Linear(d, 3)
        s.blocks = nn.ModuleList()
        for _ in range(L):
            s.blocks.append(nn.ModuleDict(dict(
                ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), po=nn.Linear(d, d),
                ln2=nn.LayerNorm(d), fc=nn.Linear(d, 4 * d), fc2=nn.Linear(4 * d, d))))
        s.hd = d // H; s.H = H
        inv = 1.0 / (10000 ** (torch.arange(0, s.hd, 2).float() / s.hd))
        s.register_buffer("inv", inv)
    def rope(s, x):                       # x (B,H,L,hd)
        L = x.shape[-2]; t = torch.arange(L, device=x.device)
        f = torch.outer(t.float(), s.inv)                 # (L, hd/2)
        c, s_ = f.cos(), f.sin()
        x1, x2 = x[..., :s.hd//2], x[..., s.hd//2:]
        return torch.cat([x1 * c - x2 * s_, x2 * c + x1 * s_], -1)
    def forward(s, x):
        B, L = x.shape; h = s.emb(x)
        from torch import tril, ones
        m = tril(ones(L, L, device=x.device, dtype=torch.bool))
        for b in s.blocks:
            r = b["ln1"](h); qkv = b["qkv"](r).view(B, L, s.H, 3 * s.hd).permute(0, 2, 1, 3)
            q, k, v = qkv.chunk(3, -1)
            q, k = s.rope(q), s.rope(k)
            a = (q @ k.transpose(-1, -2)) / math.sqrt(s.hd)
            a = a.masked_fill(~m, float("-inf")).softmax(-1) @ v
            a = a.transpose(1, 2).reshape(B, L, -1)
            h = h + b["po"](a)
            h = h + b["fc2"](F.gelu(b["fc"](b["ln2"](h))))
        h = s.ln_f(h)
        return s.head(h) * s.pos_gain, s.gate(h)

# ------------------------------------------------------------------ training
def batch(rng, nmax, B=24):
    xs = [build_stream(rng, nmax)[0] for _ in range(B)]
    cs = [build_stream.__wrapped__ if False else None for _ in range(B)]
    return xs

def train(mode, steps=2500, seed=0):
    torch.manual_seed(seed)
    mdl = RopeTF(); opt = torch.optim.AdamW(mdl.parameters(), lr=3e-3)
    rng = random.Random(seed + 100)
    for step in range(1, steps + 1):
        rows = [build_stream(rng, 8) for _ in range(24)]
        Lm = max(len(t) for t, _, _ in rows)
        x = torch.zeros(24, Lm - 1, dtype=torch.long)
        y = torch.zeros(24, Lm - 1, dtype=torch.long)
        g = torch.zeros(24, Lm - 1, dtype=torch.long)
        for i, (t, c, _) in enumerate(rows):
            x[i] = torch.tensor(t[:-1]); y[i] = torch.tensor(t[1:]); g[i] = torch.tensor(c[1:])
        lg, gt = mdl(x)
        ce = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="none").view(y.shape)
        if mode == "tf":  loss = ce.mean()
        else:             loss = ce[g == TEXT].mean()
        if mode == "hyb": loss = loss + F.cross_entropy(gt.reshape(-1, 3), g.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0: print(f"[train {mode}] {step}/{steps} loss {loss.item():.4f}", flush=True)
    return mdl

tf_full = train("tf")
tf_mask = train("msk")
hyb     = train("hyb")

# --------------------------------------------------------------------- eval
def make_eval(rng, nd, B=40):
    rows = [build_stream(rng, nd) for _ in range(B)]
    Lm = max(len(t) for t, _, _ in rows)
    x = torch.zeros(B, Lm, dtype=torch.long); g = torch.zeros(B, Lm - 1, dtype=torch.long)
    y = torch.zeros(B, Lm - 1, dtype=torch.long)
    meta = []
    for i, (t, c, m) in enumerate(rows):
        x[i, :len(t)] = torch.tensor(t); g[i, :len(t)-1] = torch.tensor(c[1:])
        y[i, :len(t)-1] = torch.tensor(t[1:]); meta.append((m, len(t)))
    return x, g, y, meta

@torch.no_grad()
def rollout(mdl, x, use_gate, meta, cap_add=8, commit=True):
    """greedy span completion from each stream's ENDIN.
    commit=True (protocol): gate read ONCE at first answer slot; if ORGAN, the organ
    owns the tape until self-termination (DEND). commit=False: per-token gate
    arbitration (ablation — exposes gate extent-tracking weakness at length)."""
    B = x.shape[0]; exact = 0; gate_ok = True; first_miss = 0
    for i in range(B):
        N, d = meta[i][0]; nd = len(str(N))
        want = [D0 + int(c) for c in str(N // d).zfill(nd)] + [RREM, D0 + N % d, DEND]
        tape = organ_tape(str(N), d)                     # mounted organ
        pre = 12 + 2 + nd + 1                            # text + DSPAN + dtok + digits + ENDIN
        seq = [x[i, :pre].tolist()]
        cur, emitted, k = pre, [], 0
        organ_mode = None
        while len(emitted) < len(want) + cap_add:
            xx = torch.tensor([seq[-1]])
            lg, gt = mdl(xx)
            if not use_gate:
                nxt = int(lg[0, -1].argmax())
            else:
                c = int(gt[0, -1].argmax())
                if commit:
                    if organ_mode is None:               # first answer slot: single decision
                        organ_mode = (c == ORGAN)
                        if not organ_mode: first_miss += 1
                    if organ_mode and k < len(tape): nxt = tape[k]
                    else: nxt = int(lg[0, -1].argmax())
                else:
                    nxt = tape[k] if (c == ORGAN and k < len(tape)) else int(lg[0, -1].argmax())
                    if c != ORGAN and k < len(want): gate_ok = False
            seq[-1] = seq[-1] + [nxt]; emitted.append(nxt); k += 1
            if nxt == DEND: break
        good = bool(emitted) and emitted[-1] == DEND and len(emitted) <= len(want) + cap_add
        exact += int(good and emitted[:len(want)] == want)
    return exact / B, gate_ok, first_miss

@torch.no_grad()
def teacher_stats(mdl, x, g, y):
    lg, gt = mdl(x[:, :-1])
    m = torch.zeros_like(y, dtype=torch.bool); m[:, :] = (g == TEXT)
    ce = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="none").view(y.shape)
    txt = ce[m].mean().item()
    ga = (gt.argmax(-1) == g).float().mean().item()
    fn = (((gt.argmax(-1) != ORGAN) & (g == ORGAN)).sum().item())
    fp = (((gt.argmax(-1) == ORGAN) & (g != ORGAN)).sum().item())
    return txt, ga, fn, fp

print("\n[P4] EVAL — exact-span completion % (greedy rollout), text CE, gate stats")
print(f"{'nd':>5} {'TF-FULL':>9} {'TF-MASK':>9} {'HYB-commit':>11} {'HYB-pertok':>11} "
      f"{'txtCE f/m/h':>16} {'gate acc':>9} {'FN/FP':>10}")
cert = True; rows_log = []
for nd in [8, 40, 100, 150]:
    rng = random.Random(1000 + nd)
    x, g, y, meta = make_eval(rng, nd)
    e_tf, _, _ = rollout(tf_full, x, False, meta)
    e_mk, _, _ = rollout(tf_mask, x, False, meta)
    e_hy, _, fms = rollout(hyb, x, True, meta, commit=True)
    e_pt, gok, _ = rollout(hyb, x, True, meta, commit=False)
    tf_txt = teacher_stats(tf_full, x, g, y)[0]
    mk_txt = teacher_stats(tf_mask, x, g, y)[0]
    hy_txt, ga, fn, fp = teacher_stats(hyb, x, g, y)
    print(f"{nd:>5} {e_tf:>9.2%} {e_mk:>9.2%} {e_hy:>11.2%} {e_pt:>11.2%} "
          f"{tf_txt:>5.3f}/{mk_txt:>5.3f}/{hy_txt:>5.3f} {ga:>9.2%} {fn:>4}/{fp:<4}", flush=True)
    rows_log.append(dict(nd=nd, tf=e_tf, msk=e_mk, hyb=e_hy, hyb_pertok=e_pt,
                         gate_acc=ga, fn=fn, fp=fp, first_miss=fms))
    cert &= (e_hy == 1.0) and (fms == 0)

res = dict(tag="ARC2-C7-P11M1", certified=bool(cert), rows=rows_log,
           wall_s=round(time.time() - t0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print("RESULT " + json.dumps(res), flush=True)
open("log.jsonl", "a").write(json.dumps(res) + "\n")
torch.save(hyb.state_dict(), "hybrid_host_s0.pt")
