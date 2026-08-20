"""ARC-2 cycle 7 M5a: LEARN the big/big division per-pass transducers (T3 division
of labor: python driver iterates; tables learn each pass's FST).
Pass types (all direct-supervision, L-DIRECT-GRADIENT):
  SUB   (tok=(r,d), b)   -> b', out_r=(r-d-b)%10, out_d=d      [borrow LSB->MSB]
  ADD   (tok=(r,d), c)   -> c', out_r=(r+d+c)%10, out_d=d      [carry add-back]
  SHIFT (tok=(r,d), hld) -> hld'=r, out=(lag): pair(hld_prev, d); NULL at first,
                            ENDT flushes (last r, 0)           [d shifts down one]
Driver: M4b oracle semantics (certified 511/511): rounds of SUB*->ADD->SHIFT with
counter, alignment s=lenN-lenD..0, remainder emitted at s<0. Certify end-to-end.
"""
import json, random, resource, time
import torch, torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0); t0 = time.time()

# ---------------- oracle driver (M4b, certified) with pluggable passes ----------
def drive(N, D, sub, add, shift, max_passes=6000):
    sn, sd = len(str(N)), len(str(D)); L = sn + 1
    rd = [int(c) for c in str(N)[::-1]] + [0] * (L - sn)
    dd = [int(c) for c in str(D)][::-1]
    s = sn - sd
    d_at = lambda i, sh: dd[i - sh] if 0 <= i - sh < len(dd) else 0
    mode, cnt = "SUB", 0; qd, rem = [], None
    for _ in range(max_passes):
        if mode == "DONE": break
        if mode == "SUB":
            rd, b = sub(rd, [d_at(i, s) for i in range(L)])
            if b == 0: cnt += 1
            else: mode = "ADD"
        elif mode == "ADD":
            rd, _ = add(rd, [d_at(i, s) for i in range(L)])
            mode = "SHIFT"
        elif mode == "SHIFT":
            rd = shift(rd)
            qd.append(cnt); s -= 1
            if s < 0: rem = rd[:]; mode = "DONE"
            else: mode, cnt = "SUB", 0
    q = "".join(map(str, qd)).lstrip("0") or "0"
    r = "".join(map(str, rem[::-1])).lstrip("0") or "0"
    return q, r

# reference passes (semantics the tables must learn)
def ref_sub(rd, dseq):
    b = 0; out = []
    for r, d in zip(rd, dseq):
        t = r - d - b; out.append(t % 10); b = 1 if t < 0 else 0
    return out, b
def ref_add(rd, dseq):
    c = 0; out = []
    for r, d in zip(rd, dseq):
        u = r + d + c; out.append(u % 10); c = u // 10
    return out, c
# ORACLE at driver level (M4b semantics re-verified inline, SUB/ADD via ref passes)
def ref_shift_pairs(pairs):
    out = []; hld = None
    for (r, d) in pairs:
        if hld is not None: out.append((hld, d))
        hld = r
    out.append((hld, 0))
    return out[1:] if len(out) > len(pairs) else out

def drive_ref(N, D):
    sn, sd = len(str(N)), len(str(D)); L = sn + 1
    rd = [int(c) for c in str(N)[::-1]] + [0] * (L - sn)
    dd = [int(c) for c in str(D)][::-1]; s = sn - sd
    d_at = lambda i, sh: dd[i - sh] if 0 <= i - sh < len(dd) else 0
    mode, cnt = "SUB", 0; qd, rem = [], None
    for _ in range(6000):
        if mode == "DONE": break
        if mode == "SUB":
            rd, b = ref_sub(rd, [d_at(i, s) for i in range(L)])
            if b == 0: cnt += 1
            else: mode = "ADD"
        elif mode == "ADD":
            rd, _ = ref_add(rd, [d_at(i, s) for i in range(L)])
            mode = "SHF"
        else:
            pairs = [(rd[i], d_at(i, s)) for i in range(L)]
            pairs = ref_shift_pairs(pairs)
            rd = [p[0] for p in pairs]
            qd.append(cnt); s -= 1
            if s < 0: rem = rd[:]; mode = "DONE"
            else: mode, cnt = "SUB", 0
    q = "".join(map(str, qd)).lstrip("0") or "0"
    r = "".join(map(str, rem[::-1])).lstrip("0") or "0"
    return q, r

rng = random.Random(5); bad = 0
for _ in range(300):
    sn = rng.randrange(1, 41); sd = rng.randrange(1, sn + 1)
    N = rng.randrange(10 ** (sn - 1), 10 ** sn)
    D = rng.randrange(10 ** (sd - 1), 10 ** sd) if sd > 1 else rng.randrange(1, 10)
    if D > N: N, D = D, N
    qq, rr = divmod(N, D)
    bad += tuple(map(int, drive_ref(N, D))) != (qq, rr)
print(f"[P0] driver+lagged-shift reference: {300-bad}/300 exact", flush=True)
assert bad == 0

# ---------------- learned per-pass tables (pair-token space) ----------------
# SUB/ADD: Tb[100,2]->b', To[100,2]->out_r; out_d=d implicit (no table needed).
# SHIFT: Th[100,10]? held is r (0..9): rows (paircode, hld) -> out paircode (lag).
def train_tables(seed=0, steps=1500):
    torch.manual_seed(seed)
    Tbs = nn.Parameter(0.1 * torch.randn(100, 2, 2))    # SUB borrow next
    Tas = nn.Parameter(0.1 * torch.randn(100, 2, 2))    # ADD carry next
    Tos = nn.Parameter(0.1 * torch.randn(100, 2, 10))   # SUB out digit
    Toa = nn.Parameter(0.1 * torch.randn(100, 2, 10))   # ADD out digit
    Ts = nn.Parameter(0.1 * torch.randn(100, 10, 100))  # SHIFT: (pair, hld) -> out pair
    opt = torch.optim.AdamW([Tbs, Tas, Tos, Toa, Ts], lr=2e-2); rng = random.Random(seed + 9)
    for step in range(1, steps + 1):
        rows_b = []; rows_o = []; rows_s = []
        for _ in range(48):
            for op in (0, 1):                            # 0=SUB, 1=ADD
                b = 0
                for _ in range(rng.randrange(1, 9)):
                    r, d = rng.randrange(10), rng.randrange(10)
                    if op == 0:
                        t = r - d - b; nb = 1 if t < 0 else 0; od = t % 10
                    else:
                        u = r + d + b; nb = u // 10; od = u % 10
                    code = r * 10 + d
                    if op == 0: rows_b.append((code, b, nb)); rows_o.append((code, b, od))
                    else: rows_b.append((code + 100, b, nb)); rows_o.append((code + 100, b, od))
                    b = nb
            pairs = [(rng.randrange(10), rng.randrange(10)) for _ in range(rng.randrange(1, 9))]
            hld = rng.randrange(10)
            for (r, d) in pairs:                         # lag write: out=(hld, d); hld<-r
                rows_s.append((r * 10 + d, hld, hld * 10 + d)); hld = r
            rows_s.append((pairs[-1][0] * 10 + pairs[-1][1], hld, hld * 10 + 0))  # ENDT flush
        rb = torch.tensor(rows_b); ro = torch.tensor(rows_o); rs_ = torch.tensor(rows_s)
        mb = rb[:, 0] >= 100; mo = ro[:, 0] >= 100
        loss = (F.cross_entropy(Tbs[rb[~mb, 0] % 100, rb[~mb, 1]], rb[~mb, 2])
                + F.cross_entropy(Tas[rb[mb, 0] % 100, rb[mb, 1]], rb[mb, 2])
                + F.cross_entropy(Tos[ro[~mo, 0] % 100, ro[~mo, 1]], ro[~mo, 2])
                + F.cross_entropy(Toa[ro[mo, 0] % 100, ro[mo, 1]], ro[mo, 2])
                + F.cross_entropy(Ts[rs_[:, 0], rs_[:, 1]], rs_[:, 2]))
        loss.backward(); opt.step(); opt.zero_grad()
        if step % 500 == 0: print(f"[train] {step}/{steps} CE {loss.item():.5f}", flush=True)
    return (Tbs.argmax(-1).numpy(), Tas.argmax(-1).numpy(),
            Tos.argmax(-1).numpy(), Toa.argmax(-1).numpy(), Ts.argmax(-1).numpy())

Tbs_i, Tas_i, Tos_i, Toa_i, Ts_i = train_tables()

def learned_sub(rd, dseq):
    b = 0; out = []
    for r, d in zip(rd, dseq):
        code = r * 10 + d
        out.append(int(Tos_i[code, b])); b = int(Tbs_i[code, b])
    return out, b
def learned_add(rd, dseq):
    c = 0; out = []
    for r, d in zip(rd, dseq):
        code = r * 10 + d
        out.append(int(Toa_i[code, c])); c = int(Tas_i[code, c])
    return out, c

# SHIFT in learned pipeline operates on PAIRS via Ts_i (driver keeps d-sequence):
def learned_drive(N, D):
    sn, sd = len(str(N)), len(str(D)); L = sn + 1
    rd = [int(c) for c in str(N)[::-1]] + [0] * (L - sn)
    dd = [int(c) for c in str(D)][::-1]; s = sn - sd
    d_at = lambda i, sh: dd[i - sh] if 0 <= i - sh < len(dd) else 0
    mode, cnt = "SUB", 0; qd, rem = [], None
    for _ in range(6000):
        if mode == "DONE": break
        dseq = [d_at(i, s) for i in range(L)]
        if mode == "SUB":
            rd, b = learned_sub(rd, dseq)
            if b == 0: cnt += 1
            else: mode = "ADD"
        elif mode == "ADD":
            rd, _ = learned_add(rd, dseq)
            mode = "SHF"
        else:
            pairs = [(rd[i], dseq[i]) for i in range(L)]
            np_ = []; hld = 0
            for (r, d) in pairs:                        # lag via learned table
                np_.append(int(Ts_i[r * 10 + d, hld])); hld = r
            np_.append(int(Ts_i[pairs[-1][0] * 10 + pairs[-1][1], hld]))
            np_ = np_[1:]
            rd = [p // 10 for p in np_]
            qd.append(cnt); s -= 1
            if s < 0: rem = rd[:]; mode = "DONE"
            else: mode, cnt = "SUB", 0
    if rem is None: return "-1", "-1"
    q = "".join(map(str, qd)).lstrip("0") or "0"
    r = "".join(map(str, rem[::-1])).lstrip("0") or "0"
    return q, r

rng = random.Random(999); fails = 0; cases = []
for sn, sd in [(8, 4), (20, 10), (40, 20), (40, 35), (40, 1), (1, 1)]:
    k = 0; f = 0
    for _ in range(50):
        N = rng.randrange(10 ** (sn - 1), 10 ** sn)
        D = rng.randrange(10 ** (sd - 1), 10 ** sd) if sd > 1 else rng.randrange(1, 10)
        if D > N: N, D = D, N
        qq, rr = divmod(N, D)
        got = tuple(map(int, learned_drive(N, D)))
        f += got != (qq, rr); k += 1
    fails += f; cases.append(f"{sn}/{sd}:{k-f}/{k}")
print(f"[certify] learned big/big pipeline: {cases} "
      f"({'CERTIFIED' if fails == 0 else 'FAILED ' + str(fails)}), trained on <=8-digit rows", flush=True)

res = dict(tag="ARC2-C7-P11M5a", certified=bool(fails == 0), per_size=cases,
           wall_s=round(time.time() - t0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print("RESULT " + json.dumps(res), flush=True)
open("log.jsonl", "a").write(json.dumps(res) + "\n")
torch.save({"Tbs": torch.tensor(Tbs_i), "Tas": torch.tensor(Tas_i),
            "Tos": torch.tensor(Tos_i), "Toa": torch.tensor(Toa_i),
            "Ts": torch.tensor(Ts_i)}, "bigdiv_s0.pt")
