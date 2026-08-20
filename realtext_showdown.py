"""
REAL-TEXT SHOWDOWN: Transformer vs Register-Machine vs Hybrid on real English
==============================================================================
The mission test: can the permutation-register architecture match/beat a matched
Transformer ON THE TRANSFORMER'S HOME TURF - real natural language - and win at
long context? Trains all three FROM SCRATCH on TinyStories (real English), then:
  1. held-out cross-entropy at context 256 (in-distribution fluency)
  2. length extrapolation: CE at 1024 / 4096 (trained only at 256)
  3. generated text samples from each model (read them yourself)
Single file. Deps: torch only. Auto: GPU=FULL, CPU=smoke. Corpus: streamed slice
of TinyStories via HTTP range request; falls back to smaller file, then synthetic.
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
    tf_layers=4 if FULL else 1,
    tf_steps=4000 if FULL else 120,
    mc_steps=3000 if FULL else 120,
    k=16, d_slot=32, cop_layers=2,
    gen_tokens=150 if FULL else 30,
    seeds=[0] if FULL else [0],       # one seed per arm; per-arm compare is paired
)
print(f"[setup] device={DEVICE} mode={'FULL' if FULL else 'SMOKE'}", flush=True)

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
            print(f"[corpus] downloaded {len(txt)/2**20:.1f}MB from {url.split('/')[-1]}",
                  flush=True)
            return txt
        except Exception as e:
            print(f"[corpus] {url.split('/')[-1]} failed ({e}); trying next", flush=True)
    print("[corpus] all downloads failed - using synthetic fallback", flush=True)
    rng = random.Random(0)
    names = ["tom", "lily", "max", "anna", "ben", "mia"]
    things = ["dog", "ball", "tree", "cake", "bird", "car", "book", "star"]
    verbs = ["saw", "found", "liked", "took", "lost", "made", "wanted", "gave"]
    out = []
    for _ in range(40000):
        n, t, v = rng.choice(names), rng.choice(things), rng.choice(verbs)
        out.append(f"one day {n} {v} a {t} . {n} was very happy . the end .")
    return "\n".join(out)

def tokenize(txt):
    words = re.findall(r"[a-z']+|[.,!?\"]", txt.lower())
    from collections import Counter
    cnt = Counter(words)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(CFG["vocab"] - 1)]
    idx = {w: i for i, w in enumerate(vocab)}
    ids = torch.tensor([idx.get(w, 0) for w in words], dtype=torch.long)
    cover = 1.0 - (ids == 0).float().mean().item()
    print(f"[corpus] {len(words)/1e6:.1f}M words, vocab {len(vocab)}, "
          f"coverage {cover:.3f}", flush=True)
    return ids, vocab

TXT = fetch_corpus()
IDS, VOCAB_LIST = tokenize(TXT)
V = len(VOCAB_LIST)
n_val = max(CFG["eval_lens"][-1] * 20, len(IDS) // 20)
TRAIN_IDS, VAL_IDS = IDS[:-n_val], IDS[-n_val:]
print(f"[corpus] train {len(TRAIN_IDS)/1e6:.1f}M tok, val {len(VAL_IDS)/1e3:.0f}K tok",
      flush=True)

def get_batch(source, bsz, L, rng):
    ix = [rng.randrange(0, len(source) - L - 1) for _ in range(bsz)]
    x = torch.stack([source[i:i + L] for i in ix])
    y = torch.stack([source[i + 1:i + L + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ------------------------------------------------------------------ models
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
    """Recurrent permutation-register layer with residual read (soft mode)."""
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
        self.head.weight = self.emb.weight          # weight tying

    def forward(self, x):
        h = self.emb(x)
        for l in self.layers:
            h = l(h)
        return self.head(self.norm(h))

class TransformerLM(nn.Module):
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
        enc = nn.TransformerEncoderLayer(d, 8, 4 * d, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, CFG["tf_layers"])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.head.weight = self.emb.weight

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x) + self.pe[:L]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), 1)
        return self.head(self.norm(self.tr(h, mask=mask)))

class HybridLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tf = TransformerLM()
        d = CFG["d_model"]
        self.cop = nn.ModuleList([CopLayer(d, CFG["k"], CFG["d_slot"])])
        self.fuse = nn.Linear(d, V, bias=False)
        nn.init.zeros_(self.fuse.weight)            # starts as pure transformer

    def forward(self, x):
        h = self.tf.emb(x)
        for l in self.cop:
            h = l(h)
        return self.tf(x) + self.fuse(h - self.tf.emb(x))

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ------------------------------------------------------------------ train/eval
def train(model, steps, tag):
    model.train().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
    rng = random.Random(42)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(TRAIN_IDS, CFG["batch"], CFG["ctx"], rng)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(); sch.step()
        if step % max(1, steps // 5) == 0:
            print(f"  [{tag}] {step}/{steps} CE {loss.item():.3f} "
                  f"ppl {math.exp(min(loss.item(), 20)):.1f} "
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
def generate(model, prompt="once upon a time there was a", n_tok=None, temp=0.8):
    model.eval()
    idx = {w: i for i, w in enumerate(VOCAB_LIST)}
    seq = [idx.get(w, 0) for w in prompt.split()]
    for _ in range(n_tok or CFG["gen_tokens"]):
        x = torch.tensor([seq[-CFG["ctx"]:]], device=DEVICE)
        logits = model(x)[0, -1] / temp
        seq.append(torch.multinomial(F.softmax(logits, -1), 1).item())
    return " ".join(VOCAB_LIST[i] for i in seq)

# ------------------------------------------------------------------ experiment
ARCHS = {"transformer": TransformerLM, "machine": MachineLM, "hybrid": HybridLM}
RES, SAMPLES = {}, {}
t_start = time.time()
for name, ctor in ARCHS.items():
    torch.manual_seed(0)
    model = ctor()
    steps = CFG["tf_steps"] if name == "transformer" else CFG["mc_steps"]
    print(f"[run] {name} params={n_params(model)/1e6:.2f}M steps={steps}", flush=True)
    train(model, steps, name)
    RES[name] = {f"ce@{L}": round(eval_ce(model, L), 4) for L in CFG["eval_lens"]}
    RES[name]["ppl@" + str(CFG["eval_lens"][0])] = round(
        math.exp(RES[name][f"ce@{CFG['eval_lens'][0]}"]), 1)
    SAMPLES[name] = generate(model)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print("\n" + "=" * 78)
print("REAL-TEXT RESULTS (cross-entropy nats/token on held-out TinyStories;"
      " lower=better)")
print("=" * 78)
print(f"{'model':<14}" + "".join(f"ce@{L:<9}" for L in CFG['eval_lens'])
      + f"ppl@{CFG['eval_lens'][0]}")
for name, r in RES.items():
    print(f"{name:<14}" + "".join(f"{r[f'ce@{L}']:<12}" for L in CFG["eval_lens"])
          + f"{r['ppl@' + str(CFG['eval_lens'][0])]}")
print("=" * 78)
for name, s in SAMPLES.items():
    print(f"\n--- SAMPLE [{name}] ---\n{s}")
mem = (round(torch.cuda.max_memory_allocated() / 2**20, 1)
       if DEVICE == "cuda" else None)
summary = dict(experiment="realtext_showdown", device=DEVICE,
               mode="FULL" if FULL else "SMOKE", cfg={k: v for k, v in CFG.items()},
               results=RES, samples=SAMPLES, peak_gpu_mb=mem,
               wall_min=round((time.time() - t_start) / 60, 1))
print("\n########## PASTE-BACK BLOCK ##########")
print(json.dumps(summary))
print("############## END ###################")
