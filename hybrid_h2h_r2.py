"""
ROUND 2 - HYBRID HEAD-TO-HEAD: Transformer vs Register-Machine vs Hybrid (Transformer+Coprocessor)
=========================================================================================
Self-contained single-file experiment. Only dependency: torch (preinstalled on
Colab/Kaggle). Auto-detects GPU (full run) vs CPU (smoke run).

Benchmark: mixed two-language corpus, each with an ANALYTIC ORACLE (exact best
possible cross-entropy), so we measure dCE = model CE - oracle CE (nats/token):
  FUZZY      : order-2 Markov chain over 16 tokens (statistical prediction -
               the transformer's home turf; no algorithmic structure).
  STRUCTURED : stochastic bounded-depth Dyck-2 (requires exact stack state -
               the machine's home turf; transformers decay with length).
Training at L=64; evaluation at L = 64 / 512 / 2048 / 4096 (64x extrapolation).

Architectures (matched scale):
  TF     : causal Transformer LM (2 layers, sinusoidal PE).
  MACHINE: register machine LM - 12 slots, 16 fixed permutation instructions
           (block/full cyclic shifts x write-bit), learned dispatch; transitions
           snap to discrete at eval ("hard mode"); decode continuous.
  HYBRID : TF backbone + machine coprocessor, late fusion into the logits.

Also reports the label-free crystallization gate for machine/hybrid arms
(hard-vs-soft CE divergence across lengths) and peak GPU/RAM usage.

USAGE: python hybrid_h2h.py           (or run the cell in Colab/Kaggle)
At the end it prints a results table and a JSON block marked PASTE-BACK.
"""
import bisect, json, math, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FULL = DEVICE == "cuda" or os.environ.get("FORCE_FULL") == "1"
SMOKE = not FULL
CFG = dict(
    seeds=[0, 1, 2] if FULL else [0],
    tf_steps=6000 if FULL else 200,
    mc_steps=9000 if FULL else 200,
    batch=64 if FULL else 16,
    train_len=64,
    eval_lens=[64, 512, 2048, 4096] if FULL else [64, 256],
    eval_reps=3 if FULL else 2,
    d_model=64 if FULL else 32,
    k=12, d_slot=16,
)
print(f"[setup] device={DEVICE} mode={'FULL' if FULL else 'SMOKE'} cfg={CFG}")
torch.manual_seed(1234)

# ---------------------------------------------------------------- benchmark data
V_MARKOV, V_DYCK = 16, 4          # union vocab: markov 0-15, dyck 16-19
VOCAB = V_MARKOV + V_DYCK
_g = torch.Generator().manual_seed(777)          # fixed world seed
MARKOV_T = F.softmax(1.5 * torch.randn(V_MARKOV, V_MARKOV, V_MARKOV,
                                       generator=_g), dim=-1)  # P(next|t-2,t-1)
MARKOV_P = MARKOV_T.tolist()                     # fast python access
MARKOV_CUM = MARKOV_T.cumsum(-1).tolist()

def gen_markov(batch, length, rng):
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        seq = [rng.randrange(V_MARKOV), rng.randrange(V_MARKOV)]
        nll = [math.log(V_MARKOV), math.log(V_MARKOV)]
        while len(seq) < length + 1:
            a, b = seq[-2], seq[-1]
            nxt = min(bisect.bisect(MARKOV_CUM[a][b], rng.random()), V_MARKOV - 1)
            seq.append(nxt)
            nll.append(-math.log(max(MARKOV_P[a][b][nxt], 1e-12)))
        xs.append(seq[:length]); ys.append(seq[1:length + 1])
        os_.append(nll[1:length + 1])
    return (torch.tensor(xs), torch.tensor(ys), torch.tensor(os_))

def gen_dyck(batch, length, rng, D=6):
    L2, L4 = math.log(2.0), math.log(4.0)
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x, nll, stack = [], [], []
        for _ in range(length + 1):
            d = len(stack)
            if d == 0:
                t = rng.randrange(2); stack.append(t)
                x.append(2 * t); nll.append(L2)
            elif d == D:
                t = stack.pop(); x.append(2 * t + 1); nll.append(0.0)
            else:
                if rng.random() < 0.5:
                    t = rng.randrange(2); stack.append(t)
                    x.append(2 * t); nll.append(L4)
                else:
                    t = stack.pop(); x.append(2 * t + 1); nll.append(L2)
        xs.append([v + V_MARKOV for v in x[:length]])
        ys.append([v + V_MARKOV for v in x[1:length + 1]])
        os_.append(nll[1:length + 1])
    return (torch.tensor(xs), torch.tensor(ys), torch.tensor(os_))

TASKS = {"fuzzy": gen_markov, "structured": gen_dyck}

def gen_mixed(batch, length, rng):
    xs, ys = [], []
    for _ in range(batch):
        gen = gen_markov if rng.random() < 0.5 else gen_dyck
        x, y, _ = gen(1, length, rng)
        xs.append(x[0]); ys.append(y[0])
    return torch.stack(xs), torch.stack(ys)

# ---------------------------------------------------------------- architectures
def role_basis(k):
    def shift(block, d):
        P = torch.zeros(k, k)
        m = {i: i for i in range(k)}
        for idx, i in enumerate(block):
            m[i] = block[(idx + d) % len(block)]
        for i in range(k):
            P[m[i], i] = 1.0
        return P
    A, B, Fu = list(range(6)), list(range(6, 12)), list(range(k))
    I = torch.eye(k)
    return torch.stack([I, shift(A, 1), shift(A, -1), shift(B, 1), shift(B, -1),
                        shift(A, 1) @ shift(B, 1), shift(Fu, 1), shift(Fu, -1)])

def st_onehot(p):
    h = torch.zeros_like(p).scatter_(-1, p.argmax(-1, keepdim=True), 1.0)
    return h + p - p.detach()

class Coprocessor(nn.Module):
    """Register machine: 12 slots, 16 fixed instructions (8 role perms x write-bit).
    Discrete transitions (snap at eval); returns per-position state reads."""
    def __init__(self, d_model, k=12, d_slot=16):
        super().__init__()
        self.k, self.d_slot, self.hard = k, d_slot, False
        self.register_buffer("PH", torch.stack([role_basis(k)[o % 8]
                                                for o in range(16)]))
        self.register_buffer("gbits", (torch.arange(16) < 8).float())
        self.alpha = nn.Linear(d_model, 16)
        nn.init.zeros_(self.alpha.bias); self.alpha.bias.data[8] = 2.0
        self.readq = nn.Linear(d_model, k)
        self.beta = nn.Linear(d_model, 8)
        self.vcode = nn.Parameter(torch.randn(8, d_slot))
        self.wlog = nn.Parameter(torch.randn(16, k))
        self.S0 = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.outdim = d_slot + k * d_slot

    def forward(self, h):                                   # h: (B,L,d_model)
        B, L, _ = h.shape
        a = F.softmax(self.alpha(h), -1)
        q = F.softmax(self.readq(h), -1)
        beta = F.softmax(self.beta(h), -1)
        w = F.softmax(self.wlog, -1)
        if self.hard:
            a, q, beta, w = st_onehot(a), st_onehot(q), st_onehot(beta), st_onehot(w)
        gb = self.gbits.view(-1, 1, 1)
        Mo = self.PH - gb * torch.einsum("oij,oj,ol->oil", self.PH, w, w)
        u = self.gbits.unsqueeze(-1) * torch.einsum("oij,oj->oi", self.PH, w)
        A = torch.einsum("blo,oij->blij", a, Mo)
        uv = torch.einsum("blo,oi->bli", a, u)
        v = beta @ self.vcode
        b = uv.unsqueeze(-1) * v.unsqueeze(-2)
        S = self.S0.expand(B, -1, -1)
        outs = []
        for t in range(L):
            S = torch.bmm(A[:, t], S) + b[:, t]
            r = torch.einsum("bk,bkd->bd", q[:, t], S)
            outs.append(torch.cat([r, S.reshape(B, -1)], -1))
        return torch.stack(outs, 1)                         # (B,L,outdim)

class TransformerLM(nn.Module):
    def __init__(self, d_model, max_len=4200):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(1e4) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        enc = nn.TransformerEncoderLayer(d_model, 4, 4 * d_model,
                                         batch_first=True, norm_first=True,
                                         dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, 2)
        self.head = nn.Linear(d_model, VOCAB)

    def backbone(self, x):
        B, L = x.shape
        h = self.emb(x) + self.pe[:L]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), 1)
        return self.tr(h, mask=mask)

    def forward(self, x):
        return self.head(self.backbone(x))

class MachineLM(nn.Module):
    def __init__(self, d_model, k, d_slot):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model)
        self.cop = Coprocessor(d_model, k, d_slot)
        self.head = nn.Sequential(nn.Linear(d_model + self.cop.outdim, 2 * d_model),
                                  nn.ReLU(), nn.Linear(2 * d_model, VOCAB))

    def forward(self, x):
        h = self.emb(x)
        return self.head(torch.cat([h, self.cop(h)], -1))

class HybridLM(nn.Module):
    """Transformer backbone + coprocessor, late fusion into logits."""
    def __init__(self, d_model, k, d_slot):
        super().__init__()
        self.tf = TransformerLM(d_model)
        self.cop = Coprocessor(d_model, k, d_slot)
        self.fuse = nn.Sequential(nn.Linear(self.cop.outdim, d_model), nn.ReLU(),
                                  nn.Linear(d_model, VOCAB))

    def forward(self, x):
        h = self.tf.emb(x)                                  # cop reads raw embeddings
        return self.tf(x) + self.fuse(self.cop(h))

def set_hard(model, hard):
    for m in model.modules():
        if isinstance(m, Coprocessor):
            m.hard = hard

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ---------------------------------------------------------------- train / eval
def train(model, steps, rng, tag, curriculum=False, structured_only=False):
    model.train().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    t0 = time.time()
    for step in range(1, steps + 1):
        if structured_only or (curriculum and step <= int(0.4 * steps)):
            x, y, _ = gen_dyck(CFG["batch"], CFG["train_len"], rng)
        else:
            x, y = gen_mixed(CFG["batch"], CFG["train_len"], rng)
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % max(1, steps // 4) == 0:
            print(f"  [{tag}] step {step}/{steps} CE {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model

@torch.no_grad()
def eval_dce(model, task, L, reps):
    model.eval()
    bs = max(2, min(16, 8192 // L))
    ce = orc = n = 0.0
    for i in range(reps):
        rng = random.Random(50_000 + L + i)
        x, y, o = TASKS[task](bs, L, rng)
        x, y = x.to(DEVICE), y.to(DEVICE)
        lp = F.log_softmax(model(x), -1)
        ce += -lp.gather(-1, y.unsqueeze(-1)).sum().item()
        orc += o.sum().item(); n += y.numel()
    return (ce - orc) / n

def evaluate(model, has_cop):
    res = {}
    modes = [("hard", True), ("soft", False)] if has_cop else [("", None)]
    for mname, hard in modes:
        if hard is not None:
            set_hard(model, hard)
        for task in TASKS:
            for L in CFG["eval_lens"]:
                key = f"{task}@{L}" + (f"[{mname}]" if mname else "")
                res[key] = round(eval_dce(model, task, L, CFG["eval_reps"]), 4)
    if has_cop:                                  # label-free crystallization gate
        L0, L1 = CFG["eval_lens"][0], CFG["eval_lens"][-1]
        d0 = res[f"structured@{L0}[hard]"] - res[f"structured@{L0}[soft]"]
        d1 = res[f"structured@{L1}[hard]"] - res[f"structured@{L1}[soft]"]
        res["crystallized_gate"] = bool(d0 < 0.05 and d1 <= 0.02)
        set_hard(model, True)                    # report hard mode as canonical
    return res

# ---------------------------------------------------------------- experiment
ARCHS = {
    "transformer":    (lambda: TransformerLM(CFG["d_model"]), {}),
    "machine_mixed":  (lambda: MachineLM(CFG["d_model"], CFG["k"], CFG["d_slot"]), {}),
    "machine_structonly": (lambda: MachineLM(CFG["d_model"], CFG["k"], CFG["d_slot"]),
                           dict(structured_only=True)),
    "hybrid_curric":  (lambda: HybridLM(CFG["d_model"], CFG["k"], CFG["d_slot"]),
                       dict(curriculum=True)),
}
ALL = {}
t_start = time.time()
for seed in CFG["seeds"]:
    for name, (ctor, tkw) in ARCHS.items():
        torch.manual_seed(seed)
        model = ctor()
        steps = CFG["tf_steps"] if name == "transformer" else CFG["mc_steps"]
        rng = random.Random(seed * 1000 + 17)
        print(f"[run] arch={name} seed={seed} params={n_params(model)}", flush=True)
        train(model, steps, rng, f"{name}-s{seed}", **tkw)
        ALL[f"{name}_s{seed}"] = evaluate(model, has_cop="machine" in name
                                          or "hybrid" in name)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

mem = (round(torch.cuda.max_memory_allocated() / 2**20, 1)
       if DEVICE == "cuda" else None)
summary = dict(round=2, device=DEVICE, mode="FULL" if FULL else "SMOKE", cfg=CFG,
               results=ALL, peak_gpu_mb=mem,
               wall_min=round((time.time() - t_start) / 60, 1))

print("\n" + "=" * 78)
print("RESULTS  (dCE = excess cross-entropy over analytic oracle, nats/token;"
      " lower=better, 0=perfect)")
print("=" * 78)
hdr = f"{'run':<22}" + "".join(f"{t}@{L:<6}"[:13].ljust(13)
                               for t in TASKS for L in CFG["eval_lens"])
print(hdr)
for run, r in ALL.items():
    cells = []
    for t in TASKS:
        for L in CFG["eval_lens"]:
            k = f"{t}@{L}[hard]" if f"{t}@{L}[hard]" in r else f"{t}@{L}"
            cells.append(f"{r[k]:<13}")
    gate = r.get("crystallized_gate", "-")
    print(f"{run:<26}" + "".join(cells) + f"  gate={gate}")
print("=" * 78)
print("\n########## PASTE-BACK BLOCK (copy everything between the hashes) ##########")
print(json.dumps(summary))
print("############################ END PASTE-BACK ###############################")
