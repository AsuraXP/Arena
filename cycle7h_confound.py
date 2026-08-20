"""ARC-2 cycle 7 control A: gate-head confound. M3: masked host + JOINT 6-class gate
head drifted to CE ~4.3 at long spans; M4a (no head) stayed within 0.09 of TF.
Arms (3-organ streams: div/add/mul, V=47, identical data/steps/seed):
  TF      : plain LM (reference)
  HM-JOIN : masked host + gate head trained JOINTLY (M3 config)
  HM-NOHD : masked host, no gate head (M4a config)
  HM-DET  : masked host + gate head on DETACHED trunk (h.detach())
Metric: teacher-forced text CE on TEXT tokens at operand width 8 and 150.
Expected law L-HEAD-DECOUPLE if HM-DET ~ HM-NOHD < HM-JOIN at w=150.
"""
import json, math, random, resource, time
import torch, torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0); t0 = time.time()

VW, D0, SLASH, PLUS, EQ, DEND, STAR = 32, 32, 42, 43, 44, 45, 46
V = 47
TEXT, DIV, ADD, MUL = 0, 2, 3, 4
dig = lambda c: D0 + int(c)

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
    def sync(): cls.extend([TEXT] * (len(toks) - len(cls)))
    for _ in range(n_spans):
        toks += text_tokens(rng, rng.randrange(5, 10)); sync()
        if rng.random() < 0.4:
            k = rng.randrange(1, 7); n0 = rng.randrange(10 ** k)
            toks += [dig(c) for c in str(n0)]; sync()
        u = rng.random()
        if u < 0.33:
            d = rng.randrange(2, 13); nd = rng.randrange(1, nmax + 1)
            N = rng.randrange(10 ** (nd - 1), 10 ** nd) if nd > 1 else rng.randrange(1, 10)
            ans = ([dig(c) for c in str(N // d).zfill(nd)]
                   + [dig(c) for c in f"{N % d:02d}"] + [DEND])
            toks += [dig(c) for c in str(N)] + [SLASH] + [dig(c) for c in str(d)] + [EQ]
            sync(); toks += ans; cls += [DIV] * len(ans)
        elif u < 0.66:
            k = rng.randrange(1, nmax + 1)
            A = rng.randrange(10 ** (k - 1), 10 ** k) if k > 1 else rng.randrange(1, 10)
            B = rng.randrange(10 ** k)
            toks += [PLUS]
            for i in range(k):
                toks += [dig(str((A // 10**i) % 10)), dig(str((B // 10**i) % 10))]
            st = str(A + B)[::-1]; ans = [dig(c) for c in st] + [DEND]
            toks += [EQ]; sync(); toks += ans; cls += [ADD] * len(ans)
        else:
            ka = rng.randrange(1, nmax + 1); kb = rng.randrange(1, nmax + 1)
            A = rng.randrange(10 ** (ka - 1), 10 ** ka) if ka > 1 else rng.randrange(1, 10)
            B = rng.randrange(10 ** (kb - 1), 10 ** kb) if kb > 1 else rng.randrange(1, 10)
            import hashlib
            p = str(A * B)                       # answers from python (organ-equiv, certified M3)
            ans = [dig(c) for c in p] + [DEND]
            toks += [dig(c) for c in str(A)] + [STAR] + [dig(c) for c in str(B)] + [EQ]
            sync(); toks += ans; cls += [MUL] * len(ans)
    toks += text_tokens(rng, rng.randrange(5, 10)); sync()
    return toks, cls

r2 = random.Random(3); okall = True
for _ in range(200):
    toks, cls = build_stream(r2, 8)
    okall &= len(toks) == len(cls)
    okall &= all((c == TEXT) or (32 <= t < 42) or (t == DEND) for t, c in zip(toks, cls))
print(f"[P0] stream oracle: {'OK' if okall else 'BROKEN'}", flush=True)
assert okall

class RopeTF(nn.Module):
    def __init__(s, d=64, L=2, H=4):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.pos_gain = nn.Parameter(torch.ones(1))
        s.ln_f = nn.LayerNorm(d); s.head = nn.Linear(d, V, bias=False); s.gate = nn.Linear(d, 5)
        s.blocks = nn.ModuleList([nn.ModuleDict(dict(
            ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), po=nn.Linear(d, d),
            ln2=nn.LayerNorm(d), fc=nn.Linear(d, 4 * d), fc2= nn.Linear(4 * d, d))) for _ in range(L)])
        s.hd = d // H; s.H = H
        s.register_buffer("inv", 1.0 / (10000 ** (torch.arange(0, s.hd, 2).float() / s.hd)))
    def rope(s, x):
        L = x.shape[-2]; f = torch.outer(torch.arange(L, device=x.device).float(), s.inv)
        c, s_ = f.cos(), f.sin()
        x1, x2 = x[..., :s.hd//2], x[..., s.hd//2:]
        return torch.cat([x1 * c - x2 * s_, x2 * c + x1 * s_], -1)
    def forward(s, x, want_gate=True, live_gate=False):
        B, L = x.shape; h = s.emb(x); m = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        for b in s.blocks:
            r = b["ln1"](h); qkv = b["qkv"](r).view(B, L, s.H, 3 * s.hd).permute(0, 2, 1, 3)
            q, k, v = qkv.chunk(3, -1); q, k = s.rope(q), s.rope(k)
            a = ((q @ k.transpose(-1, -2)) / math.sqrt(s.hd)).masked_fill(~m, float("-inf")).softmax(-1) @ v
            h = h + b["po"](a.transpose(1, 2).reshape(B, L, -1))
            h = h + b["fc2"](F.gelu(b["fc"](b["ln2"](h))))
        h = s.ln_f(h)
        gh = s.gate(h if live_gate else h.detach())
        return s.head(h) * s.pos_gain, (gh if want_gate else None)

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
        lg, gt = mdl(x, want_gate=(mode in ("join", "det")), live_gate=(mode == "join"))
        ce = (F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="none").view(y.shape) * vm).sum() / vm.sum()
        if mode == "tf": loss = ce
        else:
            tm = vm & (g == TEXT)
            loss = (ce * tm).sum() / tm.sum()
            if mode == "join":
                loss = loss + F.cross_entropy(gt.reshape(-1, 5)[vm.reshape(-1)], g.reshape(-1)[vm.reshape(-1)])
            elif mode == "det":
                loss = loss + F.cross_entropy(gt.reshape(-1, 5)[vm.reshape(-1)], g.reshape(-1)[vm.reshape(-1)])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0: print(f"[host {mode}] {step}/{steps} loss {loss.item():.4f}", flush=True)
    return mdl

arms = {}
for mode in ("tf", "join", "nohd", "det"):
    arms[mode] = train_host(mode)

@torch.no_grad()
def txt_ce(mdl, rows):
    tot = n = 0.0
    for toks, cls in rows:
        x = torch.tensor([toks]); lg, _ = mdl(x, want_gate=False)
        ce = F.cross_entropy(lg[0, :-1], x[0, 1:], reduction="none")
        m = torch.tensor([c == TEXT for c in cls[1:]])
        tot += ce[m].sum().item(); n += m.sum().item()
    return tot / n

print("\n[P3] text CE on TEXT tokens (3-organ streams), teacher-forced")
res_rows = []
for w in (8, 150):
    rng = random.Random(6000 + w)
    rows = [build_stream(rng, w, n_spans=1) for _ in range(40)]
    ces = {m: txt_ce(mdl, rows) for m, mdl in arms.items()}
    print(f"  w={w:>3}: TF {ces['tf']:.3f} · JOIN {ces['join']:.3f} · NOHD {ces['nohd']:.3f} "
          f"· DET {ces['det']:.3f}", flush=True)
    res_rows.append(dict(w=w, **{k: round(v, 3) for k, v in ces.items()}))
d_join = res_rows[1]["join"] - res_rows[0]["join"]
d_nohd = res_rows[1]["nohd"] - res_rows[0]["nohd"]
d_det = res_rows[1]["det"] - res_rows[0]["det"]
decr = (res_rows[1]["det"] <= res_rows[1]["nohd"] + 0.05) and (res_rows[1]["det"] < res_rows[1]["join"] - 0.1)
print(f"[drift 8->150] JOIN {d_join:+.3f} · NOHD {d_nohd:+.3f} · DET {d_det:+.3f} "
      f"-> L-HEAD-DECOUPLE {'CONFIRMED' if decr else 'NOT CONFIRMED'}", flush=True)

res = dict(tag="ARC2-C7-CTRL-A", l_head_decouple=bool(decr), rows=res_rows,
           wall_s=round(time.time() - t0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print("RESULT " + json.dumps(res), flush=True)
open("log.jsonl", "a").write(json.dumps(res) + "\n")
