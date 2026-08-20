"""ARC-2 cycle 6: T7 streaming long division N / d (d in 2..12).
Tokens: digits 0-9; divisor tokens 10+d-2 (d=2..12 -> 10..20); END=21.
Outputs: quotient digit per digit token; remainder class (22+r) at END."""
import json, random, re, resource, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
t0 = time.time()
DMIN, DMAX = 2, 12
NDIV = DMAX - DMIN + 1
ENDT = 10 + NDIV                       # 21
RMAX = DMAX                            # r in 0..11

def ref_rows(N_str, d, rows=None):
    toks = [10 + d - DMIN] + [int(c) for c in N_str] + [ENDT]
    q, r = [], 0
    dd = 0
    for tok in toks:
        old_didx = dd - DMIN if dd else 0
        old_r = r
        if 10 <= tok < ENDT:
            out = -1; dd = tok - 10 + DMIN; nr = 0
        elif tok == ENDT:
            out = 10 + r; nr = r
        else:
            t = r * 10 + tok
            out = t // dd; nr = t % dd
        if rows is not None:
            rows.append((tok, old_didx, old_r, out if out >= 0 else 0,
                         nr, 1 if out >= 0 else 0))
        if 0 <= out <= 9: q.append(out)
        r = nr
    return q, r

# oracle
rng = random.Random(5); ok = True
for _ in range(300):
    d = rng.randrange(DMIN, DMAX + 1)
    N = rng.randrange(1, 10 ** rng.randrange(1, 30))
    q, r = ref_rows(str(N), d)
    qs = "".join(map(str, q)).lstrip("0") or "0"
    if int(qs) != N // d or r != N % d: ok = False
print(f"[oracle] division semantics: {'OK' if ok else 'BROKEN'}", flush=True)
assert ok

# learned factored tables
Td = nn.Parameter(0.1 * torch.randn(22, NDIV, NDIV))          # divisor register
Tr = nn.Parameter(0.1 * torch.randn(22, NDIV, RMAX, RMAX))    # remainder
Hq = nn.Parameter(0.1 * torch.randn(22, NDIV, RMAX, 22))      # output (q or 10+r)
Hm = nn.Parameter(0.1 * torch.randn(22, 2))                   # emit mask
opt = torch.optim.AdamW([Td, Tr, Hq, Hm], lr=2e-2)
rng = random.Random(1)
for step in range(1, 1501):
    rows = []
    for _ in range(64):
        d = rng.randrange(DMIN, DMAX + 1)
        N = rng.randrange(0, 10 ** rng.randrange(1, 8))
        ref_rows(str(N), d, rows)
    r_ = torch.tensor(rows)
    tok, dreg, rreg, out, nr, msk = [r_[:, i] for i in range(6)]
    ndreg = torch.where((tok >= 10) & (tok < ENDT), tok - 10, dreg)
    loss = (F.cross_entropy(Td[tok, dreg], ndreg)
            + F.cross_entropy(Tr[tok, dreg, rreg], nr))
    m = msk.bool()
    loss = loss + F.cross_entropy(Hq[tok, dreg, rreg][m], out[m])
    loss = loss + F.cross_entropy(Hm[tok], msk)
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 500 == 0:
        print(f"[train] {step}/1500 CE {loss.item():.5f}", flush=True)

Tdi = Td.argmax(-1).numpy(); Tri = Tr.argmax(-1).numpy()
Hqi = Hq.argmax(-1).numpy(); Hmi = Hm.argmax(-1).numpy()
def machine_div(N_str, d):
    toks = [10 + d - DMIN] + [int(c) for c in N_str] + [ENDT]
    dd, r, q, rem = 0, 0, [], None
    for tok in toks:
        if int(Hmi[tok]) == 1:
            o = int(Hqi[tok, dd, r])
            if o <= 9: q.append(o)
            else: rem = o - 10
        ndd = int(Tdi[tok, dd]); nr = int(Tri[tok, dd, r])
        dd, r = ndd, nr
    return ("".join(map(str, q)).lstrip("0") or "0"), rem

rng = random.Random(902); fails = 0
for _ in range(200):
    d = rng.randrange(DMIN, DMAX + 1)
    nd = rng.randrange(80, 151)
    N = rng.randrange(10 ** (nd - 1), 10 ** nd)
    qs, rem = machine_div(str(N), d)
    if int(qs) != N // d or rem != N % d: fails += 1
cert = fails == 0
print(f"[certify] 80-150-digit / d(2..12), trained <=7 digits: "
      f"{200-fails}/200 ({'CERTIFIED' if cert else 'FAILED'})", flush=True)

# freeze + answer T7 judge items
key = json.load(open("answer_key.json"))
cards = open("JUDGE_CARDS.md").read()
jr = random.Random(20260822)
items = []
for i, d in enumerate([7, 11, 12]):
    nd = jr.choice([100, 110, 120])
    N = jr.randrange(10 ** (nd - 1), 10 ** nd)
    tid = f"T7-{i+1}"
    if f"## {tid}" not in cards:
        cards += (f"## {tid}\n```\nCompute exactly: {N} divided by {d}. "
                  "Answer as: quotient digits, then the word 'remainder', "
                  "then the remainder.\n```\n\n")
    key[tid] = f"{N // d} remainder {N % d}"
    items.append((tid, N, d))
open("JUDGE_CARDS.md", "w").write(cards)
json.dump(key, open("answer_key.json", "w"), indent=1)

sb = open("SCOREBOARD.md").read().rstrip()
n_pass = 0
for tid, N, d in items:
    qs, rem = machine_div(str(N), d)
    pred = f"{qs} remainder {rem}"
    okj = pred == key[tid]; n_pass += okj
    row = f"| {tid} | {'PASS' if okj else 'FAIL'} | _pending_ |"
    if f"| {tid} " in sb: sb = re.sub(rf"\| {tid} \|[^\n]*", row, sb)
    else: sb = sb.replace("Certification seeds used",
                          row + "\n\nCertification seeds used", 1)
    print(f"[judge] {tid} ({len(str(N))} digits / {d}): "
          f"{'PASS' if okj else 'FAIL'}", flush=True)
sb = re.sub(r"\*\*\d+/\d+\*\*", f"**{26 + n_pass}/29**", sb, count=1)
open("SCOREBOARD.md", "w").write(sb + "\n")
res = dict(tag="ARC2-C6-T7", certified=bool(cert), judge=f"{n_pass}/3",
           wall_s=round(time.time() - t0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1))
print("RESULT " + json.dumps(res), flush=True)
open("log.jsonl", "a").write(json.dumps(res) + "\n")
torch.save({"Td": Td, "Tr": Tr, "Hq": Hq, "Hm": Hm}, "div_t7.pt")
