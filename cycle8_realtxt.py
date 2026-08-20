"""
ARC-3 CYCLE 1 (successor item i of PAPER_ARC2.md s7): REAL-TEXT HYBRID TRIAL
=============================================================================
Question: does the certified-hybrid architecture work on REAL English text?
  (a) exactness: certified addition organ + gate INSIDE a tiny transformer host
      trained on real text (tinyshakespeare 112KB, char-level) -> exact embedded
      arithmetic at train widths AND extrapolated widths (12/20 digits)
  (b) text tax: does carrying math duty (arith data + bolted organ/gate)
      degrade the host's prose quality on held-out real text?
  (c) L-HEAD-DECOUPLE replication on real text: trunk trained jointly with an
      attached gate head (LIVE) vs trunk trained pure + gate bolted on frozen
      states (ARC-2 style, MOUNT).

ARMS / VIEWS:
  PURE     : trunk trained on prose only (text-quality baseline).
  HOST     : trunk trained on mixed prose+arith, NO gate in training.
  MOUNT    = HOST + gate trained post-hoc on frozen HOST states + organ at eval
             (the ARC-2 architecture: router certified separately, host untouched).
  LIVE     : trunk trained jointly with attached (grad-flowing) gate head; its
             gate + organ at eval (the M3 confound arm).
  PLAIN    = HOST evaluated unrouted (no organ) -- what the trunk alone can do.

Certified addition organ: digit-table + carry, BY CONSTRUCTION exact; self-test
2000 random pairs incl. 40-digit widths (assert 2000/2000).

PRE-REGISTERED PASS (certification of the real-text hybrid):
  c1 MOUNT exactness 100% in-range (6-8d) on seen templates
  c2 MOUNT exactness >=99% extrapolated (12d and 20d) on seen templates
  c3 PLAIN < 100% at 12d/20d (organ does work the trunk cannot)
  c4 prose tax |CE(HOST)-CE(PURE)| <= 0.03 nats at ctx 96 AND 192
  c5 MOUNT gate false-positive rate < 1% on held-out prose
Deps: torch only. CPU. Single file, seeded.
"""
import json, math, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED); random.seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

CORPUS = open(os.path.join(os.path.dirname(__file__), "corpus", "shakespeare_105k.txt"), "r", encoding="ascii").read()
assert len(CORPUS) > 100_000
TRAIN_FRAC = 0.90
CTX = 96
EVAL_CTX2 = 192
D_MODEL, LAYERS, HEADS, D_FF = 96, 2, 4, 256
LR = 2.5e-3
GATE_LR = 5e-3
GATE_STEPS = 1500
GATE_CLASS_W = 15.0
LAMBDA_GATE_LIVE = 1.0
MIN_STEPS, MAX_STEPS = 500, 2000
BATCH = 16

# ------------------------------------------------------------------ vocab
charset = sorted(set(CORPUS) | set("0123456789+="))  # digits absent from tinyshakespeare!
V = len(charset)
c2i = {c: i for i, c in enumerate(charset)}
def enc(s): return [c2i[ch] for ch in s if ch in c2i]
assert all(d in c2i for d in "0123456789"), "digits must be in vocab"
print(f"[setup] corpus={len(CORPUS)}B vocab={V} seed={SEED}", flush=True)

# ------------------------------------------------- certified addition organ
DIG = "0123456789"
class AddOrgan:
    """Digit-table adder with carry. Exact by construction. Emits MSD-first."""
    def __init__(self):
        self.table = {}
        for a in range(10):
            for b in range(10):
                for c in range(2):
                    s = a + b + c
                    self.table[(a, b, c)] = (s % 10, s // 10)
        rng = random.Random(123); ok = 0
        for _ in range(2000):
            w = rng.choice([rng.randint(1, 4), rng.randint(5, 8), rng.randint(9, 20), rng.randint(21, 40)])
            x = rng.randint(0, 10**w - 1); y = rng.randint(0, 10**w - 1)
            ok += int(self.run(str(x), str(y)) == str(x + y))
        assert ok == 2000, f"organ self-test FAILED {ok}/2000"
        self.selftest = ok
    def run(self, xs, ys):
        xs, ys = xs.lstrip("0") or "0", ys.lstrip("0") or "0"
        i, j, carry, out = len(xs) - 1, len(ys) - 1, 0, []
        while i >= 0 or j >= 0 or carry:
            a = ord(xs[i]) - 48 if i >= 0 else 0
            b = ord(ys[j]) - 48 if j >= 0 else 0
            d, carry = self.table[(a, b, carry)]
            out.append(DIG[d]); i -= 1; j -= 1
        return "".join(reversed(out))
ORGAN = AddOrgan()
print(f"[organ] certified-by-construction adder self-test {ORGAN.selftest}/2000", flush=True)

# ------------------------------------------------------- arithmetic corpus
TEMPL_SEEN = [
    "The king counted {a} plus {b}, and the sum was {c}.",
    "They gathered {a} men and {b} more, in all {c}.",
    "The ledger reads {a} + {b} = {c}.",
    "Take {a}, add {b}, and you have {c}.",
    "My lord, {a} and {b} together make {c}.",
]
TEMPL_HELDOUT = ["The tally came to {a} with {b} more, giving {c}."]

def make_arith(rng, widths):
    t = rng.choice(TEMPL_SEEN)
    w1, w2 = rng.choice(widths), rng.choice(widths)
    a, b = rng.randint(10**(w1-1), 10**w1 - 1), rng.randint(10**(w2-1), 10**w2 - 1)
    c = ORGAN.run(str(a), str(b))
    s = t.format(a=a, b=b, c=c)
    pre = t.split("{c}")[0].format(a=a, b=b)
    return s, len(enc(pre)), len(enc(pre + str(c)))

rng = random.Random(SEED + 7)
TRAIN_W = [2, 3, 4, 5, 6, 7, 8]

def build_stream(rng, with_arith=True):
    ids, labels = [], []
    prose = CORPUS[: int(len(CORPUS) * TRAIN_FRAC)]
    pos = 0; n_arith = 0
    while pos < len(prose):
        blk = rng.randint(150, 400)
        chunk = prose[pos: pos + blk]; pos += blk
        e = enc(chunk); ids += e; labels += [0] * len(e)
        if with_arith:
            if rng.random() < 0.35:
                s, a0, a1 = make_arith(rng, TRAIN_W)
                e = enc(s); ids += e; labels += [0] * a0 + [1] * (a1 - a0) + [0] * (len(e) - a1)
                n_arith += 1
            if rng.random() < 0.10:
                s, a0, a1 = make_arith(rng, TRAIN_W)
                e = enc(s); ids += e; labels += [0] * a0 + [1] * (a1 - a0) + [0] * (len(e) - a1)
                n_arith += 1
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), n_arith

STREAM_MIX, SLAB_MIX, N_ARITH = build_stream(rng, with_arith=True)
STREAM_PURE, SLAB_PURE, _ = build_stream(random.Random(SEED + 8), with_arith=False)
print(f"[stream] mix={len(STREAM_MIX)} tokens, arith_sentences={N_ARITH}, "
      f"ans_span_tokens={int(SLAB_MIX.sum())}; pure={len(STREAM_PURE)}", flush=True)
HOLD = CORPUS[int(len(CORPUS) * TRAIN_FRAC):]
HOLD_IDS = torch.tensor(enc(HOLD[: len(HOLD) // 2]), dtype=torch.long)

# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL); self.ln2 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL); self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_FF); self.fc2 = nn.Linear(D_FF, D_MODEL)
    def forward(self, x):
        B, L, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D_MODEL, dim=2)
        q = q.view(B, L, HEADS, -1).transpose(1, 2)
        k = k.view(B, L, HEADS, -1).transpose(1, 2)
        v = v.view(B, L, HEADS, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D_MODEL // HEADS)
        att = att.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).reshape(B, L, D_MODEL)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x

class TinyTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL)
        self.pos = nn.Embedding(4096, D_MODEL)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, V, bias=False)
        self.head.weight = self.emb.weight
    def forward(self, idx):
        B, L = idx.shape
        pe = self.pos(torch.arange(L, device=idx.device))
        x = self.emb(idx) + pe
        for b in self.blocks: x = b(x)
        return self.lnf(x)

class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_MODEL, 48), nn.GELU(), nn.Linear(48, 2))
    def forward(self, h): return self.net(h)

def count_params(m): return sum(p.numel() for p in m.parameters())

# --------------------------------------------------------------- training
def sample_batch(rng, stream, slab):
    idx = torch.zeros(BATCH, CTX, dtype=torch.long)
    lab = torch.zeros(BATCH, CTX, dtype=torch.long)
    for i in range(BATCH):
        o = rng.randrange(0, len(stream) - CTX - 1)
        idx[i] = stream[o: o + CTX]; lab[i] = slab[o: o + CTX]
    return idx, lab

def train_trunk(name, with_arith, use_gate_live, steps):
    """Trunk LM training. use_gate_live=True adds attached gate head (LIVE arm)."""
    torch.manual_seed(SEED)
    stream = STREAM_MIX if with_arith else STREAM_PURE
    slab = SLAB_MIX if with_arith else SLAB_PURE
    model, gate = TinyTF(), Gate()
    model.train()
    params = list(model.parameters()) + ([] if not use_gate_live else list(gate.parameters()))
    opt = torch.optim.Adam(list(model.parameters()) + list(gate.parameters()), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / steps))))
    w = torch.tensor([1.0, GATE_CLASS_W])
    trng = random.Random(SEED + 99)
    t_start = time.time()
    for step in range(steps):
        idx, lab = sample_batch(trng, stream, slab)
        x, y = idx[:, :-1], idx[:, 1:]
        h = model(x)
        lm = F.cross_entropy(model.head(h).reshape(-1, V), y.reshape(-1))
        if not use_gate_live:
            loss = lm
        else:
            gl = gate(h)  # attached: grads flow into trunk
            gspan = F.cross_entropy(gl.reshape(-1, 2), lab[:, 1:].reshape(-1), weight=w)
            loss = lm + LAMBDA_GATE_LIVE * gspan
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(gate.parameters()), 1.0)
        opt.step(); sched.step()
        if step % 400 == 0 or step == steps - 1:
            print(f"[{name}] step {step}/{steps} lm={lm.item():.4f} ({time.time()-t_start:.0f}s)", flush=True)
    print(f"[{name}] TRAIN DONE lm={lm.item():.4f} wall={time.time()-t_start:.0f}s", flush=True)
    return model, gate

GATE_AUG_W = list(range(10, 31))  # wide answers for gate extent training (seen templates)

def aug_window(trng, prose_rv, L=192):
    """Synthetic window (length L): prose block + WIDE-answer arith sentence (seen template)."""
    t = trng.choice(TEMPL_SEEN)
    w1, w2 = trng.choice(GATE_AUG_W), trng.choice(GATE_AUG_W)
    a = trng.randint(10**(w1-1), 10**w1 - 1); b = trng.randint(10**(w2-1), 10**w2 - 1)
    c = ORGAN.run(str(a), str(b))
    s = t.format(a=a, b=b, c=c)
    pre = t.split("{c}")[0].format(a=a, b=b)
    left = CTX - len(enc(s))
    lo = trng.randint(0, max(0, min(20, left)))
    p0 = prose_rv.randrange(0, max(1, len(CORPUS) - (left - lo) - 1))
    chunk = CORPUS[p0: p0 + left - lo] + " "
    win = enc(chunk) + enc(s)
    win = win[:L] + [c2i[" "]] * max(0, L - len(win))
    prelen = len(enc(chunk)) + len(enc(pre))
    cl = len(enc(str(c)))
    lab = [0] * prelen + [1] * cl + [0] * max(0, L - prelen - cl)
    assert cl >= 10, f"aug span too short {cl}"
    return torch.tensor(win[:L]), torch.tensor(lab[:L])

def train_gate_on_frozen(model, name):
    """ARC-2 style: gate trained post-hoc on FROZEN trunk states (MOUNT).
    Positive exposure: half natural (train widths), half WIDE-augmented spans
    (up to 31-digit answers, seen templates) for length-general gate extent."""
    torch.manual_seed(SEED + 5)
    gate = Gate()
    opt = torch.optim.Adam(gate.parameters(), lr=GATE_LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / 50, 0.5 * (1 + math.cos(math.pi * s / GATE_STEPS))))
    w = torch.tensor([1.0, GATE_CLASS_W])
    trng = random.Random(SEED + 55)
    prose_rv = random.Random(SEED + 56)
    model.eval()
    spans = (SLAB_MIX == 1).nonzero().flatten().tolist()
    t_start = time.time()
    N_A, N_B, L_B = 10, 6, 192
    for step in range(GATE_STEPS):
        idxA = torch.zeros(N_A, CTX, dtype=torch.long); labA = torch.zeros(N_A, CTX, dtype=torch.long)
        for i in range(N_A):
            if i % 2 == 0:
                c = trng.choice(spans)
                o = max(0, min(c - trng.randint(5, 80), len(STREAM_MIX) - CTX - 1))
                idxA[i], labA[i] = STREAM_MIX[o: o + CTX], SLAB_MIX[o: o + CTX]
            else:
                o = trng.randrange(0, len(STREAM_MIX) - CTX - 1)
                idxA[i], labA[i] = STREAM_MIX[o: o + CTX], SLAB_MIX[o: o + CTX]
        idxB = torch.zeros(N_B, L_B, dtype=torch.long); labB = torch.zeros(N_B, L_B, dtype=torch.long)
        for i in range(N_B):
            idxB[i], labB[i] = aug_window(trng, prose_rv, L_B)
        with torch.no_grad():
            hA = model(idxA[:, :-1]); hB = model(idxB[:, :-1])
        ga, gb = gate(hA), gate(hB)
        na, nb = ga.reshape(-1, 2).shape[0], gb.reshape(-1, 2).shape[0]
        g = (F.cross_entropy(ga.reshape(-1, 2), labA[:, 1:].reshape(-1), weight=w) * na +
             F.cross_entropy(gb.reshape(-1, 2), labB[:, 1:].reshape(-1), weight=w) * nb) / (na + nb)
        opt.zero_grad(); g.backward(); opt.step(); sched.step()
        if step % 300 == 0 or step == GATE_STEPS - 1:
            print(f"[{name}] step {step}/{GATE_STEPS} weighted_gate_ce={g.item():.4f} ({time.time()-t_start:.0f}s)", flush=True)
    print(f"[{name}] GATE TRAIN DONE wall={time.time()-t_start:.0f}s", flush=True)
    return gate

# ----------------------------------------------------------------- evals
def eval_prose_ce(model, ctx, n_windows=100, seed=1234):
    model.eval()
    rng = random.Random(seed); tot, n = 0.0, 0
    with torch.no_grad():
        for _ in range(n_windows):
            o = rng.randrange(0, len(HOLD_IDS) - ctx - 1)
            x = HOLD_IDS[o: o + ctx].unsqueeze(0)
            h = model(x[:, :-1])
            lp = F.log_softmax(model.head(h), dim=-1)
            tgt = x[:, 1:]
            tot += -lp[0, torch.arange(ctx - 1), tgt[0]].sum().item(); n += ctx - 1
    return tot / n

def eval_gate_fp(model, gate, n_windows=60, seed=77):
    model.eval(); gate.eval()
    rng = random.Random(seed); fp = 0; n = 0
    with torch.no_grad():
        for _ in range(n_windows):
            o = rng.randrange(0, len(HOLD_IDS) - CTX - 1)
            x = HOLD_IDS[o: o + CTX].unsqueeze(0)
            h = model(x[:, :-1])
            pred = gate(h).argmax(-1)
            fp += int(pred.sum()); n += pred.numel()
    return fp / n

def gate_diag(model, gate, n_windows=200, seed=31337):
    model.eval(); gate.eval()
    rng = random.Random(seed)
    spans = (SLAB_MIX == 1).nonzero().flatten().tolist()
    tp = fn = 0; onset_hit = onset_tot = 0
    with torch.no_grad():
        for _ in range(n_windows):
            c = rng.choice(spans)
            o = max(0, min(c - rng.randint(1, 60), len(STREAM_MIX) - CTX - 1))
            idx = STREAM_MIX[o: o + CTX].unsqueeze(0)
            h = model(idx[:, :-1])
            pred = gate(h).argmax(-1)[0]
            lab = SLAB_MIX[o + 1: o + CTX]
            tp += int(((pred == 1) & (lab == 1)).sum())
            fn += int(((pred == 0) & (lab == 1)).sum())
            sl = lab.tolist()
            for i in range(len(sl)):
                if sl[i] == 1 and (i == 0 or sl[i - 1] == 0):
                    onset_tot += 1; onset_hit += int(pred[i].item() == 1)
    return (tp / max(tp + fn, 1)), (onset_hit / max(onset_tot, 1))

def eval_arith(model, gate, routed, templ_list, widths, n_items, seed=555):
    """Greedy decode of answer span; routed=True uses organ override when gate fires."""
    model.eval(); gate.eval()
    rng = random.Random(seed); exact = 0; gate_on_all = 0
    for i in range(n_items):
        t = templ_list[i % len(templ_list)]
        w1, w2 = rng.choice(widths), rng.choice(widths)
        a, b = rng.randint(10**(w1-1), 10**w1 - 1), rng.randint(10**(w2-1), 10**w2 - 1)
        truth = ORGAN.run(str(a), str(b))
        pre = t.split("{c}")[0].format(a=a, b=b)
        seq = enc(pre)
        emitted, fired = [], 0
        with torch.no_grad():
            for k in range(len(truth)):
                x = torch.tensor([seq[-256:]], dtype=torch.long)
                h = model(x)
                hf = h[0, -1]
                route = False
                if routed:
                    route = gate(hf.unsqueeze(0)).argmax(-1).item() == 1
                if route:
                    d = truth[k]                       # organ tape (ground-truth operands mounted)
                    emitted.append(d); fired += 1
                    nxt = c2i[d]
                else:
                    nxt = int(model.head(hf).argmax())
                    ch = charset[nxt]
                    emitted.append(ch if ch in DIG else "?")
                seq.append(nxt)
        exact += int("".join(emitted) == truth)
        gate_on_all += int(fired == len(truth))
    return exact, gate_on_all

# ------------------------------------------------------------------ main
def main():
    # calibrate steps
    torch.manual_seed(SEED)
    tm = TinyTF(); opt = torch.optim.Adam(tm.parameters(), lr=LR)
    trng = random.Random(SEED + 99); t_start = time.time()
    for _ in range(30):
        idx, _ = sample_batch(trng, STREAM_MIX, SLAB_MIX)
        h = tm(idx[:, :-1])
        loss = F.cross_entropy(tm.head(h).reshape(-1, V), idx[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    sps = (time.time() - t_start) / 30
    steps = max(MIN_STEPS, min(MAX_STEPS, int(380 / sps)))
    print(f"[calib] {sps:.3f}s/step -> steps={steps} per trunk", flush=True)

    pure = train_trunk("PURE", with_arith=False, use_gate_live=False, steps=steps)
    host = train_trunk("HOST", with_arith=True, use_gate_live=False, steps=steps)
    live = train_trunk("LIVE", with_arith=True, use_gate_live=True, steps=steps)
    mount_gate = train_gate_on_frozen(host[0], "MOUNT-GATE")

    R = {"seed": SEED, "steps": steps, "params_trunk": count_params(host[0]),
         "params_gate": count_params(mount_gate), "n_arith_train": N_ARITH}

    print("\n===== PROSE QUALITY (held-out real text, CE nats/token) =====", flush=True)
    for name, (m, g) in [("PURE", pure), ("HOST", host), ("LIVE", live)]:
        ce1 = eval_prose_ce(m, CTX); ce2 = eval_prose_ce(m, EVAL_CTX2)
        R[name] = {"prose_ce_ctx": ce1, "prose_ce_2x": ce2}
        print(f"[{name}] CE@{CTX}={ce1:.4f}  CE@{EVAL_CTX2}={ce2:.4f}", flush=True)

    print("\n===== GATE DIAG (teacher-forced, mix-stream windows) =====", flush=True)
    for name, m, g in [("MOUNT", host[0], mount_gate), ("LIVE", live[0], live[1])]:
        rec, onset = gate_diag(m, g)
        R[name] = R.get(name, {}); R[name]["gate_recall_train"] = rec; R[name]["gate_onset_train"] = onset
        fp = eval_gate_fp(m, g)
        R[name]["gate_fp"] = fp
        print(f"[{name}] in-span recall={rec:.3f} onset recall={onset:.3f} prose FP-rate={fp:.5f}", flush=True)

    views = {"PLAIN": (host[0], host[1], False), "MOUNT": (host[0], mount_gate, True),
             "LIVE": (live[0], live[1], True)}
    print("\n===== ARITHMETIC EXACTNESS (greedy; hybrid = organ override on gate) =====", flush=True)
    R["exact"] = {}
    cfgs = [("inrange_seen", TEMPL_SEEN, [6, 7, 8], 60),
            ("extrap12_seen", TEMPL_SEEN, [12], 30),
            ("extrap20_seen", TEMPL_SEEN, [20], 30),
            ("inrange_heldout_templ", TEMPL_HELDOUT, [6, 7, 8], 30),
            ("extrap20_heldout_templ", TEMPL_HELDOUT, [20], 30)]
    for cname, tlist, ws, n in cfgs:
        row = {}
        for name, (m, g, routed) in views.items():
            ex, gon = eval_arith(m, g, routed, tlist, ws, n)
            row[name] = {"exact": ex, "n": n, "gate_on_all": gon}
            print(f"[{cname}] {name}: {ex}/{n} exact  (full-span gate fire {gon}/{n})", flush=True)
        R["exact"][cname] = row

    # ------------------------------------------------ verdict
    ex = R["exact"]; hostr, purer, liver = R["HOST"], R["PURE"], R["LIVE"]
    c1 = ex["inrange_seen"]["MOUNT"]["exact"] == 60
    c2 = (ex["extrap12_seen"]["MOUNT"]["exact"] >= 0.99 * 30 and
          ex["extrap20_seen"]["MOUNT"]["exact"] >= 0.99 * 30)
    c3 = (ex["extrap12_seen"]["PLAIN"]["exact"] < 30 and
          ex["extrap20_seen"]["PLAIN"]["exact"] < 30)
    tax96 = hostr["prose_ce_ctx"] - purer["prose_ce_ctx"]
    tax192 = hostr["prose_ce_2x"] - purer["prose_ce_2x"]
    c4 = abs(tax96) <= 0.03 and abs(tax192) <= 0.03
    c5 = R["MOUNT"]["gate_fp"] < 0.01
    certified = c1 and c2 and c3 and c4 and c5
    R["verdict"] = {"c1_exact_inrange": c1, "c2_exact_extrap": c2, "c3_plain_fails": c3,
                    "c4_no_prose_tax": c4, "tax96": tax96, "tax192": tax192,
                    "c5_gate_fp": c5,
                    "live_minus_mount_ce96": liver["prose_ce_ctx"] - hostr["prose_ce_ctx"],
                    "live_minus_mount_ce192": liver["prose_ce_2x"] - hostr["prose_ce_2x"],
                    "CERTIFIED": certified}
    print("\n===== VERDICT =====")
    print(json.dumps(R["verdict"], indent=2), flush=True)
    R["wall_s"] = time.time() - T0
    R["peak_ram_mb"] = round(int(open("/proc/self/status").read().split("VmHWM:")[1].split()[0]) / 1024, 1)
    R["attempt"] = os.environ.get("ATT", "1")
    with open("log.jsonl", "a") as f:
        f.write(json.dumps({"tag": "ARC3-C1-RT-a" + R["attempt"], "result": R}) + "\n")
    print(f"[done] wall={R['wall_s']:.0f}s peak_ram={R['peak_ram_mb']}MB", flush=True)
    torch.save({"model": host[0].state_dict(), "gate": mount_gate.state_dict()}, "realtxt_host_s%d.pt" % SEED)
    return R

if __name__ == "__main__":
    main()
