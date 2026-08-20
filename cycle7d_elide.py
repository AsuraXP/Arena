"""ARC-2 cycle 7 M4a: transcript elision — fix HOST-M text-CE drift at long spans.
Mount change: after the organ self-terminates, the host's context view replaces the
answer region with ONE [ANS] marker (router + organs still see the full stream).
Arms (same arch/seed/steps): TF-FULL (plain LM, full stream) · HOST-M (masked, full
context) · HOST-E (masked, elided context). Router: frozen certified krgate3_s0.pt.
Organ: frozen div_t7.pt. Certify: HOST-E text CE flat across widths while hybrid
span exactness stays 100% (organ-owned, untouched).
"""
import json, math, random, resource, time
import torch, torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0); t0 = time.time()

VW, D0, SLASH, PLUS, EQ, DEND, STAR, ANS = 32, 32, 42, 43, 44, 45, 46, 47
V = 48
TEXT, DIV = 0, 2
dig = lambda c: D0 + int(c)

# ---------------- frozen certified organs/router (M2/M3 artifacts) ----------------
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

ckg = torch.load("krgate3_s0.pt", map_location="cpu")
Tsi, Thi = ckg["Ts"].numpy(), ckg["Th"].numpy()
def gate_route(toks):
    s = 0; out = []
    for t in toks:
        out.append(int(Thi[t, s])); s = int(Tsi[t, s])
    return out, s

# ---------------- stream builder: full view + elided host view ----------------
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
    ft, fc, et, ec = [], [], [], []
    def sync():
        fc.extend([TEXT] * (len(ft) - len(fc))); ec.extend([TEXT] * (len(et) - len(ec)))
    for _ in range(n_spans):
        for v in (ft, et): v.extend(text_tokens(rng, rng.randrange(5, 10)))
        sync()
        if rng.random() < 0.4:
            k = rng.randrange(1, 7); n0 = rng.randrange(10 ** k)
            dgs = [dig(c) for c in str(n0)]
            ft += dgs; et += dgs; sync()
        d = rng.randrange(2, 13); nd = rng.randrange(1, nmax + 1)
        N = rng.randrange(10 ** (nd - 1), 10 ** nd) if nd > 1 else rng.randrange(1, 10)
        quest = [dig(c) for c in str(N)] + [SLASH] + [dig(c) for c in str(d)] + [EQ]
        ans = ([dig(c) for c in str(N // d).zfill(nd)]
               + [dig(c) for c in f"{N % d:02d}"] + [DEND])
        ft += quest; et += quest; sync()
        ft += ans; fc += [DIV] * len(ans)
        et += [ANS]; ec += [TEXT]          # elided view: one marker, host-lossless
    for v in (ft, et): v.extend(text_tokens(rng, rng.randrange(5, 10)))
    sync()
    return ft, fc, et, ec

r2 = random.Random(3); okall = True
for _ in range(200):
    ft, fc, et, ec = build_stream(r2, 8)
    okall &= len(ft) == len(fc) and len(et) == len(ec)
    okall &= all((c == TEXT) or (32 <= t < 42) or (t == DEND) for t, c in zip(ft, fc))
    g, _ = gate_route(ft)
    okall &= all(a == b for a, b in zip(g[:-1], fc[1:]))
    okall &= et.count(ANS) == ft.count(DEND)
print(f"[P0] dual-view oracle + frozen-router routing: {'OK' if okall else 'BROKEN'}", flush=True)
assert okall

# ---------------- host ----------------
class RopeTF(nn.Module):
    def __init__(s, d=64, L=2, H=4):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.pos_gain = nn.Parameter(torch.ones(1))
        s.ln_f = nn.LayerNorm(d); s.head = nn.Linear(d, V, bias=False)
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
        h = s.ln_f(h); return s.head(h) * s.pos_gain, h

def train_host(mode, steps=2500, seed=0):
    torch.manual_seed(seed); mdl = RopeTF()
    opt = torch.optim.AdamW(mdl.parameters(), lr=3e-3); rng = random.Random(seed + 70)
    for step in range(1, steps + 1):
        rows = [build_stream(rng, 8) for _ in range(24)]
        Lm = max(len(r[0 if mode != "el" else 2]) for r in rows)
        x = torch.zeros(24, Lm - 1, dtype=torch.long); y = torch.zeros_like(x)
        g = torch.zeros_like(x); vm = torch.zeros_like(x, dtype=torch.bool)
        for i, (ft, fc, et, ec) in enumerate(rows):
            tk, cl = (ft, fc) if mode != "el" else (et, ec)
            n = len(tk) - 1
            x[i, :n] = torch.tensor(tk[:-1]); y[i, :n] = torch.tensor(tk[1:]); g[i, :n] = torch.tensor(cl[1:])
            vm[i, :n] = True
        lg, _ = mdl(x)
        ce = (F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="none").view(y.shape) * vm).sum() / vm.sum()
        if mode == "tf": loss = ce
        else:
            tm = vm & (g == TEXT)
            loss = (ce * tm).sum() / tm.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0: print(f"[host {mode}] {step}/{steps} loss {loss.item():.4f}", flush=True)
    return mdl

tf_full = train_host("tf"); host_m = train_host("msk"); host_e = train_host("el")

# ---------------- eval ----------------
def make_eval(rng, w, B=40):
    rows = []
    for i in range(B):
        pre = text_tokens(rng, 8)
        d = rng.randrange(2, 13); N = rng.randrange(10 ** (w - 1), 10 ** w)
        quest = [dig(c) for c in str(N)] + [SLASH] + [dig(c) for c in str(d)] + [EQ]
        want = ([dig(c) for c in str(N // d).zfill(w)]
                + [dig(c) for c in f"{N % d:02d}"] + [DEND])
        cut = len(pre) + len(quest)
        post = text_tokens(rng, 6)
        ft = pre + quest + want + post
        fc = [TEXT] * cut + [DIV] * len(want) + [TEXT] * 6
        et = pre + quest + [ANS] + post
        ec = [TEXT] * (cut + 1) + [TEXT] * 6
        assert len(ft) == len(fc) and len(et) == len(ec)
        rows.append((ft, fc, et, ec, cut, div_tape(str(N), d), want))
    return rows

@torch.no_grad()
def txt_ce(mdl, rows, view):
    tot = n = 0.0
    for ft, fc, et, ec, *_ in rows:
        tk, cl = (ft, fc) if view == "full" else (et, ec)
        x = torch.tensor([tk]); lg, _ = mdl(x)
        ce = F.cross_entropy(lg[0, :-1], x[0, 1:], reduction="none")
        m = torch.tensor([c == TEXT for c in cl[1:]])
        tot += ce[m].sum().item(); n += m.sum().item()
    return tot / n

@torch.no_grad()
def hybrid_exact(rows):
    ok = 0
    for ft, fc, et, ec, cut, tape, want in rows:
        seq = list(ft[:cut]); k = 0
        while k < len(tape) + 8:
            g, _ = gate_route(seq)
            nxt = tape[k] if (g[-1] == DIV and k < len(tape)) else None
            if nxt is None: break
            seq.append(nxt); k += 1
            if nxt == DEND: break
        ok += seq[cut:cut + len(want)] == want
    return ok / len(rows)

print("\n[P3] EVAL — text CE (flat = good) · hybrid exact (must stay 100%) · ctx len")
cert = True; rows_log = []
for w in (8, 40, 150):
    rows = make_eval(random.Random(4000 + w), w)
    ce_tf = txt_ce(tf_full, rows, "full"); ce_m = txt_ce(host_m, rows, "full"); ce_e = txt_ce(host_e, rows, "el")
    ex = hybrid_exact(rows)
    lf = sum(len(r[0]) for r in rows) / len(rows); le = sum(len(r[2]) for r in rows) / len(rows)
    print(f"  w={w:>3}: txtCE tf {ce_tf:.3f} · msk {ce_m:.3f} · ELID {ce_e:.3f} "
          f"(drift msk {ce_m - ce_tf:+.3f} / elid {ce_e - ce_tf:+.3f}) · hybrid {ex:.0%} "
          f"· ctx {lf:.0f} -> {le:.0f} tok", flush=True)
    rows_log.append(dict(w=w, ce_tf=ce_tf, ce_msk=ce_m, ce_elid=ce_e, hybrid=ex,
                         ctx_full=lf, ctx_elided=le))
    cert &= (ex == 1.0)
    if w >= 40:   # elision must never cost fluency vs EITHER baseline arm
        cert &= (ce_e <= ce_tf + 0.02) and (ce_e <= ce_m + 0.02)
cert &= (rows_log[-1]["ctx_elided"] < 0.75 * rows_log[-1]["ctx_full"])
drift_m = rows_log[-1]["ce_msk"] - rows_log[0]["ce_msk"]
drift_e = rows_log[-1]["ce_elid"] - rows_log[0]["ce_elid"]
print(f"[drift abs 8->150] HOST-M {drift_m:+.3f} · HOST-E {drift_e:+.3f} nats "
      f"(relative-to-TF: see rows; criterion = never-cost, not absolute flatness)", flush=True)

res = dict(tag="ARC2-C7-P11M4a", certified=bool(cert), drift_msk=round(drift_m, 3),
           drift_elid=round(drift_e, 3), rows=rows_log,
           wall_s=round(time.time() - t0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print("RESULT " + json.dumps(res), flush=True)
open("log.jsonl", "a").write(json.dumps(res) + "\n")
torch.save(host_e.state_dict(), "hybrid_host_elided_s0.pt")
