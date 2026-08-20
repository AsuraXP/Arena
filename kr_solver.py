"""ARC-2 cycle 2: generalized KR automaton solver for frozen suite T1/T2/T4.
Generic transition basis {id, const_j, shift+1, transpositions}; learned dispatch
+ contextual output table; hard (vertex-snapped) inference; certify-then-judge."""
import json, math, random, re, resource, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)

def basis(M):
    mats = [torch.eye(M)]
    for j in range(M):                       # const_j (reset)
        C = torch.zeros(M, M); C[j, :] = 1.0; mats.append(C)
    S = torch.zeros(M, M)                    # shift+1
    for i in range(M): S[(i + 1) % M, i] = 1.0
    mats.append(S)
    for i in range(M):                       # transpositions
        for j in range(i + 1, M):
            T = torch.eye(M); T[i, i] = T[j, j] = 0.0; T[i, j] = T[j, i] = 1.0
            mats.append(T)
    return torch.stack(mats)

def st_onehot(p):
    h = torch.zeros_like(p).scatter_(-1, p.argmax(-1, keepdim=True), 1.0)
    return h + p - p.detach()

class KRA(nn.Module):
    def __init__(self, vin, M, vout, out_pre=False):
        super().__init__()
        self.M, self.out_pre, self.hard = M, out_pre, False
        self.register_buffer("TB", basis(M))
        self.mdisp = nn.Parameter(0.3 * torch.randn(vin, self.TB.shape[0]))
        self.table = nn.Parameter(0.3 * torch.randn(vin, M, vout))
    def forward(self, x):
        B, L = x.shape
        md = F.softmax(self.mdisp, -1)[x]
        if self.hard: md = st_onehot(md)
        T = torch.einsum("blj,jmn->blmn", md, self.TB)
        m = torch.zeros(B, self.M); m[:, 0] = 1.0
        outs = []
        for t in range(L):
            m_pre = m
            m = torch.bmm(T[:, t], m.unsqueeze(-1)).squeeze(-1)
            if self.hard: m = st_onehot(m)
            mu = m_pre if self.out_pre else m
            outs.append(torch.einsum("bm,bmv->bv", mu, self.table[x[:, t]]))
        return torch.stack(outs, 1)

# ---------------------------------------------------------------- task data
def gen_t2(B, L, rng):
    x = torch.tensor([[rng.randrange(2) for _ in range(L)] for _ in range(B)])
    y = torch.cumsum(x, 1) % 2
    return x, y

SWAPS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
SWAP_ID = {p: 5 + k for k, p in enumerate(SWAPS)}     # tokens 5..14; init 0..4
def gen_t4(B, L, rng):
    xs, ys = [], []
    for _ in range(B):
        p = rng.randrange(5); x = [p]; y = [p]
        for _ in range(L - 1):
            a, b = rng.sample(range(5), 2); a, b = min(a, b), max(a, b)
            x.append(SWAP_ID[(a, b)])
            if p == a: p = b
            elif p == b: p = a
            y.append(p)
        xs.append(x); ys.append(y)
    return torch.tensor(xs), torch.tensor(ys)

END = 100
def gen_t1(B, n, rng):
    xs, ys = [], []
    for _ in range(B):
        x, y, c = [], [], 0
        for _ in range(n):
            a, b = rng.randrange(10), rng.randrange(10)
            x.append(10 * a + b)
            s = a + b + c
            y.append(s % 10); c = s // 10
        x.append(END); y.append(10 + c)
        xs.append(x); ys.append(y)
    return torch.tensor(xs), torch.tensor(ys)

TASKS = {
    "t2": dict(vin=2, M=2, vout=2, out_pre=False, train_len=64,
               gen=gen_t2, cert_len=2000, steps=2500),
    "t4": dict(vin=15, M=5, vout=5, out_pre=False, train_len=80,
               gen=gen_t4, cert_len=350, steps=3500),
    "t1": dict(vin=101, M=2, vout=12, out_pre=True, train_len=8,
               gen=gen_t1, cert_len=120, steps=3500),
}

def acc(model, gen, L, rng, B=16, reps=4):
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(reps):
            x, y = gen(B, L, rng)
            p = model(x).argmax(-1)
            c += (p == y).sum().item(); t += y.numel()
    model.train()
    return c / t

def solve(task, max_seeds=6):
    cfg = TASKS[task]
    for seed in range(max_seeds):
        torch.manual_seed(seed)
        model = KRA(cfg["vin"], cfg["M"], cfg["vout"], cfg["out_pre"])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        rng = random.Random(seed + 50)
        for step in range(1, cfg["steps"] + 1):
            x, y = cfg["gen"](64, cfg["train_len"], rng)
            loss = F.cross_entropy(model(x).reshape(-1, cfg["vout"]),
                                   y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        model.hard = True
        a_short = acc(model, cfg["gen"], cfg["train_len"], random.Random(900))
        a_long = acc(model, cfg["gen"], cfg["cert_len"], random.Random(901))
        certified = a_short == 1.0 and a_long == 1.0
        print(f"[{task}] seed {seed}: hard@train {a_short:.4f} "
              f"hard@{cfg['cert_len']} {a_long:.4f} "
              f"{'CERTIFIED' if certified else 'restart'}", flush=True)
        if certified:
            return model, seed, a_long
    return model, -1, a_long

# ---------------------------------------------------------------- judge cards
KEY = json.load(open("answer_key.json"))
CARDS = open("JUDGE_CARDS.md").read()
def item(tid):
    m = re.search(rf"## {re.escape(tid)}\n```\n(.*?)\n```", CARDS, re.S)
    return m.group(1)

@torch.no_grad()
def answer_t2(model, tid):
    bits = re.search(r"\n([01]{100,})", item(tid)).group(1)
    x = torch.tensor([[int(b) for b in bits]])
    cls = model(x).argmax(-1)[0, -1].item()
    return "odd" if cls == 1 else "even"

@torch.no_grad()
def answer_t4(model, tid):
    p = item(tid)
    init = int(re.search(r"position (\d)", p).group(1))
    ops = re.findall(r"swap (\d) (\d)", p)
    toks = [init] + [SWAP_ID[(min(int(a), int(b)), max(int(a), int(b)))]
                     for a, b in ops]
    return str(model(torch.tensor([toks])).argmax(-1)[0, -1].item())

@torch.no_grad()
def answer_t1(model, tid):
    p = item(tid).replace("\n", " ")
    A, B = re.search(r"(\d{20,})\s*\+\s*(\d{20,})", p).groups()
    ra, rb = A[::-1], B[::-1]
    n = max(len(ra), len(rb))
    ra, rb = ra.ljust(n, "0"), rb.ljust(n, "0")
    toks = [10 * int(ra[i]) + int(rb[i]) for i in range(n)] + [END]
    out = model(torch.tensor([toks])).argmax(-1)[0]
    digits = "".join(str(out[i].item()) for i in range(n))[::-1]
    return ("1" if out[n].item() == 11 else "") + digits

t0 = time.time()
models, seeds, rows = {}, {}, []
for task, ans_fn in (("t2", answer_t2), ("t4", answer_t4), ("t1", answer_t1)):
    models[task], seeds[task], _ = solve(task)
    models[task].hard = True; models[task].eval()
    for tid in sorted(k for k in KEY if k.lower().startswith(task[:2])):
        pred = ans_fn(models[task], tid)
        ok = pred.strip() == KEY[tid].strip()
        rows.append((tid, ok, pred))
        print(f"[judge] {tid}: {'PASS' if ok else 'FAIL'}", flush=True)

n_ok = sum(1 for _, ok, _ in rows if ok)
lines = ["# ARC-2 SCOREBOARD — sandbox-trained KR automata vs frozen judge suite",
         "", f"Machine total: **{n_ok}/{len(rows)}** exact-match", "",
         "| item | machine | frontier-LLM (operator to fill) |", "|---|---|---|"]
for tid, ok, pred in rows:
    lines.append(f"| {tid} | {'PASS' if ok else 'FAIL'} | _pending_ |")
lines += ["", f"Certification seeds used: {seeds}",
          f"Total params (3 models): "
          f"{sum(sum(p.numel() for p in m.parameters()) for m in models.values())}",
          f"Wall: {time.time()-t0:.0f}s · peak RAM "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB · 1 CPU"]
open("SCOREBOARD.md", "w").write("\n".join(lines))
res = dict(tag="ARC2-C2-SOLVE", score=f"{n_ok}/{len(rows)}", seeds=seeds,
           items={tid: ("PASS" if ok else "FAIL") for tid, ok, _ in rows},
           wall_s=round(time.time() - t0, 1))
print("RESULT " + json.dumps(res))
open("log.jsonl", "a").write(json.dumps(res) + "\n")
for t, m in models.items():
    torch.save(m.state_dict(), f"kra_{t}.pt")
