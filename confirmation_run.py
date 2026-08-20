"""
CONFIRMATION RUN: Machine vs Transformer vs RoPE-Transformer on real English
=============================================================================
3 seeds x 3 architectures, matched scale, trained from scratch on TinyStories.
Arms:
  tf_sin  : standard Transformer (sinusoidal PE)   - round-1 baseline
  tf_rope : modern Transformer (rotary embeddings) - the strong/fair baseline
  machine : attention-free permutation-register recurrence (ours)
Reports per-seed and aggregated (mean/min/max) held-out CE at 256/1024/4096,
plus generated samples (seed 0). Single file, torch only, GPU=FULL / CPU=smoke.
Estimated FULL runtime on T4: ~2h. Progress prints throughout.
"""
import json, math, os, random, re, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FULL = DEVICE == "cuda" or os.environ.get("FORCE_FULL") == "1"
CFG = dict(
    corpus_mb=60 if FULL else 3,
    vocab=8192 if FULL else 2000,
    ctx=256 if FULL else 96,
    eval_lens=[256, 1024, 4096] if FULL else [96, 192],
    batch=32 if FULL else 8,
    d_model=256 if FULL else 64,
    n_layers=4 if FULL else 1,
    n_heads=8 if FULL else 4,
    tf_steps=4000 if FULL else 100,
    mc_steps=3000 if FULL else 100,
    k=16, d_slot=32, cop_layers=2,
    gen_tokens=150 if FULL else 25,
    seeds=[0, 1, 2] if FULL else [0],
)
print(f"[setup] device={DEVICE} mode={'FULL' if FULL else 'SMOKE'} "
      f"seeds={CFG['seeds']}", flush=True)

# ------------------------------------------------------------------ corpus
URLS = [
    ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
     "TinyStoriesV2-GPT4-train.txt", CFG["corpus_mb"] * 2**20),
    ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
     "TinyStories-valid.txt", None),
]
def fetch_corpus():
    for url, nbytes in URLS:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "research-script",
                **({"Range": f"bytes=0-{nbytes-1}"} if nbytes else {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            txt = data.decode("utf-8", errors="ignore")
            print(f"[corpus] {len(txt)/2**20:.1f}MB from {url.split('/')[-1]}",
                  flush=True)
            return txt
        except Exception as e:
            print(f"[corpus] {url.split('/')[-1]} failed ({e})", flush=True)
    print("[corpus] downloads failed - synthetic fallback", flush=True)
    rng = random.Random(0)
    ns = ["tom", "lily", "max", "anna"]; ts = ["dog", "ball", "tree", "cake"]
    vs = ["saw", "found", "liked", "took"]
    return "\n".join(
        f"one day {rng.choice(ns)} {rng.choice(vs)} a {rng.choice(ts)} . the end ."
        for _ in range(40000))

def tokenize(txt):
    words = re.findall(r"[a-z']+|[.,!?\"]", txt.lower())
    from collections import Counter
    cnt = Counter(words)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(CFG["vocab"] - 1)]
    idx = {w: i for i, w in enumerate(vocab)}
    ids = torch.tensor([idx.get(w, 0) for w in words], dtype=torch.long)
    print(f"[corpus] {len(words)/1e6:.1f}M words, vocab {len(vocab)}, "
          f"coverage {1.0 - (ids == 0).float().mean().item():.3f}", flush=True)
    return ids, vocab

TXT = fetch_corpus()
IDS, VOCAB_LIST = tokenize(TXT)
V = len(VOCAB_LIST)
n_val = max(CFG["eval_lens"][-1] * 20, len(IDS) // 20)
TRAIN_IDS, VAL_IDS = IDS[:-n_val], IDS[-n_val:]
print(f"[corpus] train {len(TRAIN_IDS)/1e6:.1f}M tok, val {len(VAL_IDS)/1e3:.0f}K",
      flush=True)

def get_batch(source, bsz, L, rng):
    ix = [rng.randrange(0, len(source) - L - 1) for _ in range(bsz)]
    x = torch.stack([source[i:i + L] for i in ix])
    y = torch.stack([source[i + 1:i + L + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ------------------------------------------------------------------ machine
def role_basis(k):
    def shift(block, d):
        P = torch.zeros(k, k)
        m = {i: i for i in range(k)}
        for idx, i in enumerate(block):
            m[i] = block[(idx + d) % len(block)]
        for i in range(k):
            P[m[i], i] = 1.0
        return P
    h = k // 2
    A, B, Fu = list(range(h)), list(range(h, k)), list(range(k))
    return torch.stack([torch.eye(k), shift(A, 1), shift(A, -1), shift(B, 1),
                        shift(B, -1), shift(A, 1) @ shift(B, 1),
                        shift(Fu, 1), shift(Fu, -1)])

class CopLayer(nn.Module):
    def __init__(self, d_model, k, d_slot):
        super().__init__()
        self.k, self.d_slot = k, d_slot
        self.register_buffer("PH", torch.stack([role_basis(k)[o % 8]
                                                for o in range(16)]))
        self.register_buffer("gbits", (torch.arange(16) < 8).float())
        self.alpha = nn.Linear(d_model, 16)
        nn.init.zeros_(self.alpha.bias); self.alpha.bias.data[8] = 2.0
        self.readq = nn.Linear(d_model, k)
        self.beta = nn.Linear(d_model, 16)
        self.vcode = nn.Parameter(torch.randn(16, d_slot))
        self.wlog = nn.Parameter(torch.randn(16, k))
        self.S0 = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.out = nn.Linear(d_slot + k * d_slot, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h):
        B, L, _ = h.shape
        hn = self.norm(h)
        a = F.softmax(self.alpha(hn), -1)
        q = F.softmax(self.readq(hn), -1)
        beta = F.softmax(self.beta(hn), -1)
        w = F.softmax(self.wlog, -1)
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
        return h + self.out(torch.stack(outs, 1))

class MachineLM(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(V, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.layers = nn.ModuleList([CopLayer(d, CFG["k"], CFG["d_slot"])
                                     for _ in range(CFG["cop_layers"])])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.head.weight = self.emb.weight

    def forward(self, x):
        h = self.emb(x)
        for l in self.layers:
            h = l(h)
        return self.head(self.norm(h))

# ------------------------------------------------------------------ transformers
class SinTransformerLM(nn.Module):
    def __init__(self, max_len=4300):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(V, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(1e4) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        enc = nn.TransformerEncoderLayer(d, CFG["n_heads"], 4 * d,
                                         batch_first=True, norm_first=True,
                                         dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, CFG["n_layers"])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.head.weight = self.emb.weight

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x) + self.pe[:L]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), 1)
        return self.head(self.norm(self.tr(h, mask=mask)))

def rope_cache(L, dh, device):
    pos = torch.arange(L, device=device).float()
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=device).float() / dh))
    ang = pos[:, None] * inv[None, :]
    return ang.cos(), ang.sin()                    # (L, dh/2)

def apply_rope(x, cos, sin):                       # x: (B,H,L,dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                       dim=-1).flatten(-2)

class RoPEBlock(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh, self.dh = nh, d // nh
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, h, cos, sin):
        B, L, d = h.shape
        q, k, v = self.qkv(self.n1(h)).chunk(3, -1)
        q = q.view(B, L, self.nh, self.dh).transpose(1, 2)
        k = k.view(B, L, self.nh, self.dh).transpose(1, 2)
        v = v.view(B, L, self.nh, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.proj(o.transpose(1, 2).reshape(B, L, d))
        return h + self.mlp(self.n2(h))

class RoPETransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(V, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([RoPEBlock(d, CFG["n_heads"])
                                     for _ in range(CFG["n_layers"])])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.head.weight = self.emb.weight

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x)
        cos, sin = rope_cache(L, CFG["d_model"] // CFG["n_heads"], x.device)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.head(self.norm(h))

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ------------------------------------------------------------------ train/eval
def train(model, steps, tag, seed):
    model.train().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
    rng = random.Random(42 + seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(TRAIN_IDS, CFG["batch"], CFG["ctx"], rng)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(); sch.step()
        if step % max(1, steps // 4) == 0:
            print(f"  [{tag}-s{seed}] {step}/{steps} CE {loss.item():.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model

@torch.no_grad()
def eval_ce(model, L, reps=4):
    model.eval()
    rng = random.Random(999)
    bs = max(1, min(8, 4096 // L))
    tot = n = 0.0
    for _ in range(reps):
        x, y = get_batch(VAL_IDS, bs, L, rng)
        lp = F.log_softmax(model(x), -1)
        tot += -lp.gather(-1, y.unsqueeze(-1)).sum().item()
        n += y.numel()
    return tot / n

@torch.no_grad()
def generate(model, prompt="once upon a time there was a", temp=0.8):
    model.eval()
    idx = {w: i for i, w in enumerate(VOCAB_LIST)}
    seq = [idx.get(w, 0) for w in prompt.split()]
    for _ in range(CFG["gen_tokens"]):
        x = torch.tensor([seq[-CFG["ctx"]:]], device=DEVICE)
        logits = model(x)[0, -1] / temp
        seq.append(torch.multinomial(F.softmax(logits, -1), 1).item())
    return " ".join(VOCAB_LIST[i] for i in seq)

# ------------------------------------------------------------------ experiment
ARCHS = {"tf_sin": SinTransformerLM, "tf_rope": RoPETransformerLM,
         "machine": MachineLM}
RES, SAMPLES = {}, {}
t_start = time.time()
for seed in CFG["seeds"]:
    for name, ctor in ARCHS.items():
        torch.manual_seed(seed)
        model = ctor()
        steps = CFG["mc_steps"] if name == "machine" else CFG["tf_steps"]
        print(f"[run] {name} seed={seed} params={n_params(model)/1e6:.2f}M "
              f"({(time.time()-t_start)/60:.0f}min elapsed)", flush=True)
        train(model, steps, name, seed)
        RES[f"{name}_s{seed}"] = {f"ce@{L}": round(eval_ce(model, L), 4)
                                  for L in CFG["eval_lens"]}
        if seed == CFG["seeds"][0]:
            SAMPLES[name] = generate(model)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

AGG = {}
for name in ARCHS:
    AGG[name] = {}
    for L in CFG["eval_lens"]:
        vals = [RES[f"{name}_s{s}"][f"ce@{L}"] for s in CFG["seeds"]]
        AGG[name][f"ce@{L}"] = dict(mean=round(sum(vals) / len(vals), 4),
                                    min=min(vals), max=max(vals))

print("\n" + "=" * 78)
print("CONFIRMATION RESULTS - held-out CE (mean [min-max] over "
      f"{len(CFG['seeds'])} seeds)")
print("=" * 78)
for name in ARCHS:
    row = f"{name:<10}"
    for L in CFG["eval_lens"]:
        a = AGG[name][f"ce@{L}"]
        row += f"  @{L}: {a['mean']} [{a['min']}-{a['max']}]"
    print(row)
print("=" * 78)
for name, s in SAMPLES.items():
    print(f"\n--- SAMPLE [{name}] (seed 0) ---\n{s}")
mem = (round(torch.cuda.max_memory_allocated() / 2**20, 1)
       if DEVICE == "cuda" else None)
summary = dict(experiment="confirmation_run", device=DEVICE,
               mode="FULL" if FULL else "SMOKE", cfg=CFG, per_seed=RES,
               aggregate=AGG, samples=SAMPLES, peak_gpu_mb=mem,
               wall_min=round((time.time() - t_start) / 60, 1))
print("\n########## PASTE-BACK BLOCK ##########")
print(json.dumps(summary))
print("############## END ###################")
