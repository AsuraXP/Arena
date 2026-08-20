"""ARC-2 cycle 7 control B: host multi-seed sweep (P9). Router/organs frozen from M5b
(krgate4_s0.pt + certified organs). Host trained at seeds 1,2 with the ADOPTED config
(detached gate head per L-HEAD-DECOUPLE). Exactness is router-owned by construction;
this sweep is the record. Eval: div w=150 and dbig 40x35 span completion + text CE.
"""
import json, math, random, resource, time
import torch, torch.nn as nn
import torch.nn.functional as F
t0 = time.time()

VW, D0, SLASH, PLUS, EQ, DEND, STAR, DSL2 = 32, 32, 42, 43, 44, 45, 46, 47
V = 48
TEXT, DIV, ADD, MUL, DBIG = 0, 2, 3, 4, 5
dig = lambda c: D0 + int(c)

bk = torch.load("bigdiv_s0.pt", map_location="cpu")
Tbs_i, Tas_i = bk["Tbs"].numpy(), bk["Tas"].numpy()
Tos_i, Toa_i, TsB = bk["Tos"].numpy(), bk["Toa"].numpy(), bk["Ts"].numpy()
def bigdiv_tape(N, D):
    sn, sd = len(str(N)), len(str(D)); L = sn + 1
    rd = [int(c) for c in str(N)[::-1]] + [0] * (L - sn)
    dd = [int(c) for c in str(D)][::-1]; s = sn - sd
    d_at = lambda i, sh: dd[i - sh] if 0 <= i - sh < len(dd) else 0
    mode, cnt = "SUB", 0; qd, rem = [], None
    for _ in range(6000):
        if mode == "DONE": break
        dseq = [d_at(i, s) for i in range(L)]
        if mode == "SUB":
            b = 0; out = []
            for r, d in zip(rd, dseq):
                code = r * 10 + d; out.append(int(Tos_i[code, b])); b = int(Tbs_i[code, b])
            rd = out
            if b == 0: cnt += 1
            else: mode = "ADD"
        elif mode == "ADD":
            c = 0; out = []
            for r, d in zip(rd, dseq):
                code = r * 10 + d; out.append(int(Toa_i[code, c])); c = int(Tas_i[code, c])
            rd = out; mode = "SHF"
        else:
            pairs = [(rd[i], dseq[i]) for i in range(L)]
            np_ = []; hld = 0
            for (r, d) in pairs:
                np_.append(int(TsB[r * 10 + d, hld])); hld = r
            np_.append(int(TsB[pairs[-1][0] * 10 + pairs[-1][1], hld]))
            rd = [p // 10 for p in np_[1:]]
            qd.append(cnt); s -= 1
            if s < 0: rem = rd[:]; mode = "DONE"
            else: mode, cnt = "SUB", 0
    q = "".join(map(str, qd)).lstrip("0") or "0"
    r = "".join(map(str, rem[::-1])).lstrip("0").zfill(sd)
    return [dig(c) for c in q] + [dig(c) for c in r] + [DEND]

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

ckg = torch.load("krgate4_s0.pt", map_location="cpu")
Tsi, Thi = ckg["Ts"].numpy(), ckg["Th"].numpy()
def gate_route(toks):
    s = 0; out = []
    for t in toks:
        out.append(int(Thi[t, s])); s = int(Tsi[t, s])
    return out, s

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
        if u < 0.25:
            d = rng.randrange(2, 13); nd = rng.randrange(1, nmax + 1)
            N = rng.randrange(10 ** (nd - 1), 10 ** nd) if nd > 1 else rng.randrange(1, 10)
            ans = ([dig(c) for c in str(N // d).zfill(nd)]
                   + [dig(c) for c in f"{N % d:02d}"] + [DEND])
            toks += [dig(c) for c in str(N)] + [SLASH] + [dig(c) for c in str(d)] + [EQ]
            sync(); toks += ans; cls += [DIV] * len(ans)
        elif u < 0.5:
            k = rng.randrange(1, nmax + 1)
            A = rng.randrange(10 ** (k - 1), 10 ** k) if k > 1 else rng.randrange(1, 10)
            B = rng.randrange(10 ** k)
            toks += [PLUS]
            for i in range(k):
                toks += [dig(str((A // 10**i) % 10)), dig(str((B // 10**i) % 10))]
            st = str(A + B)[::-1]; ans = [dig(c) for c in st] + [DEND]
            toks += [EQ]; sync(); toks += ans; cls += [ADD] * len(ans)
        elif u < 0.75:
            ka = rng.randrange(1, nmax + 1); kb = rng.randrange(1, nmax + 1)
            A = rng.randrange(10 ** (ka - 1), 10 ** ka) if ka > 1 else rng.randrange(1, 10)
            B = rng.randrange(10 ** (kb - 1), 10 ** kb) if kb > 1 else rng.randrange(1, 10)
            p = str(A * B); ans = [dig(c) for c in p] + [DEND]
            toks += [dig(c) for c in str(A)] + [STAR] + [dig(c) for c in str(B)] + [EQ]
            sync(); toks += ans; cls += [MUL] * len(ans)
        else:
            sn = rng.randrange(2, nmax + 1); sd = rng.randrange(1, sn)
            N = rng.randrange(10 ** (sn - 1), 10 ** sn)
            D = rng.randrange(10 ** (sd - 1), 10 ** sd) if sd > 1 else rng.randrange(1, 10)
            q, r = divmod(N, D)
            ans = [dig(c) for c in str(q)] + [dig(c) for c in str(r).zfill(sd)] + [DEND]
            toks += [dig(c) for c in str(N)] + [DSL2] + [dig(c) for c in str(D)] + [EQ]
            sync(); toks += ans; cls += [DBIG] * len(ans)
    toks += text_tokens(rng, rng.randrange(5, 10)); sync()
    return toks, cls

class RopeTF(nn.Module):
    def __init__(s, d=64, L=2, H=4):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.pos_gain = nn.Parameter(torch.ones(1))
        s.ln_f = nn.LayerNorm(d); s.head = nn.Linear(d, V, bias=False)
        s.gate = nn.Linear(d, 6)
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
        h = s.ln_f(h); return s.head(h) * s.pos_gain, s.gate(h.detach())  # DETACHED (L-HEAD-DECOUPLE)

def train_host(seed=0, steps=2500):
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
        tm = vm & (g == TEXT)
        loss = (ce * tm).sum() / tm.sum() + F.cross_entropy(
            gt.reshape(-1, 6)[vm.reshape(-1)], g.reshape(-1)[vm.reshape(-1)])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0: print(f"[host s{seed}] {step}/{steps} loss {loss.item():.4f}", flush=True)
    return mdl

def make_rows(rng, kind, w, sd_, B=40):
    rows = []
    for i in range(B):
        pre = text_tokens(rng, 8)
        if kind == "div":
            d = rng.randrange(2, 13); N = rng.randrange(10 ** (w - 1), 10 ** w)
            q = [dig(c) for c in str(N // d).zfill(w)]
            quest = [dig(c) for c in str(N)] + [SLASH] + [dig(c) for c in str(d)] + [EQ]
            tape = div_tape(str(N), d); want = q + [dig(c) for c in f"{N % d:02d}"] + [DEND]
        else:
            N = rng.randrange(10 ** (w - 1), 10 ** w)
            D = rng.randrange(10 ** (sd_ - 1), 10 ** sd_) if sd_ > 1 else rng.randrange(1, 10)
            q, r = divmod(N, D)
            quest = [dig(c) for c in str(N)] + [DSL2] + [dig(c) for c in str(D)] + [EQ]
            tape = bigdiv_tape(N, D)
            want = [dig(c) for c in str(q)] + [dig(c) for c in str(r).zfill(sd_)] + [DEND]
        cut = len(pre) + len(quest)
        toks = pre + quest + want + text_tokens(rng, 6)
        cls = [TEXT] * cut + [2] * len(want) + [TEXT] * 6
        rows.append((toks, cls, cut, tape, want))
    return rows

@torch.no_grad()
def complete(mdl, rows):
    ok = 0
    for toks, cls, cut, tape, want in rows:
        seq = list(toks[:cut]); k = 0
        while k < len(tape) + 8:
            lg, _ = mdl(torch.tensor([seq]))
            g, _ = gate_route(seq)
            nxt = tape[k] if (g[-1] in (2, 3, 4, 5) and k < len(tape)) else int(lg[0, -1].argmax())
            seq.append(nxt); k += 1
            if nxt == DEND: break
        ok += len(seq) > cut and seq[cut:cut + len(want)] == want
    return ok / len(rows)

out = []
for seed in (1, 2):
    mdl = train_host(seed)
    div150 = complete(mdl, make_rows(random.Random(7000 + seed), "div", 150, None))
    db4035 = complete(mdl, make_rows(random.Random(8000 + seed), "dbig", 40, 35))
    print(f"[seed {seed}] div@150 {div150:.1%} · dbig 40x35 {db4035:.1%}", flush=True)
    out.append(dict(seed=seed, div150=div150, dbig4035=db4035))
allc = all(r["div150"] == 1.0 and r["dbig4035"] == 1.0 for r in out)
res = dict(tag="ARC2-C7-CTRL-B", certified=bool(allc), seeds=out,
           wall_s=round(time.time() - t0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print("RESULT " + json.dumps(res), flush=True)
open("log.jsonl", "a").write(json.dumps(res) + "\n")
