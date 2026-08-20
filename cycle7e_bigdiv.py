"""ARC-2 cycle 7 M4b (stretch): BIG/BIG division via IFT compare-subtract — Phase A:
reference encoding semantics, oracle-first (no learning until 500/500).

Tape (LSB-first): [MODE][CNT c][SH hi][SL lo][PAIR(r0,d0)]...[END]
  MODE in {SUB, REPAIR, DONE}; CNT = current quotient-digit counter; s = 10*hi+lo =
  current alignment shift of D (starts at lenN-lenD, decrements per round).
  R = N zero-padded to L = lenN+1 digits; pair_i = (r_i, d_{i-s} or 0).

SUB pass:     r_i' = (r_i - d_i - b) mod 10, borrow propagates LSB->MSB.
              ENDT: no final borrow -> CNT+1 (committed subtraction of D*10^s);
                    final borrow  -> MODE=REPAIR (rollback next pass; value in tape
                    is R - D*10^s + 10^L, i.e. two's-complement garbage).
REPAIR pass:  r_i' = (r_i + d_i + carry) mod 10 (adds D*10^s back: R + 10^L = R
              mod 10^L, final carry drops); simultaneously shifts D down one
              (new alignment s-1). ENDT: EMIT_Q(CNT) [MSB-first quotient order];
              s -= 1; s < 0 -> EMIT_R all r digits (LSB-first) + MODE=DONE;
              else MODE=SUB, CNT=0.
DONE pass:    no output (drain/halt).

Decode: quotient = concatenated EMIT_Q stream (already MSB-first), lstrip zeros;
        remainder = reverse(EMIT_R stream), lstrip zeros.
"""
import json, random, resource, time
t0 = time.time()

NSUB, NREP, NDONE = 0, 1, 2
def run_bigdiv(N, D, max_passes=4000):
    assert N >= D >= 1
    sn, sd = len(str(N)), len(str(D))
    L = sn + 1
    rd = [int(c) for c in str(N)[::-1]] + [0] * (L - sn)
    dd = [int(c) for c in str(D)[::-1]]
    s = sn - sd
    def d_at(i, sh): return dd[i - sh] if 0 <= i - sh < len(dd) else 0
    tape_mode, cnt = NSUB, 0
    emitted = []
    for _ in range(max_passes):
        if tape_mode == NDONE:
            break
        if tape_mode == NSUB:
            b = 0
            for i in range(L):
                t = rd[i] - d_at(i, s) - b
                rd[i] = t % 10
                b = 1 if t < 0 else 0
            if b == 0:
                cnt += 1
            else:
                tape_mode = NREP
        else:  # REPAIR: add back + shift down
            carry = 0
            for i in range(L):
                u = rd[i] + d_at(i, s) + carry
                rd[i] = u % 10
                carry = u // 10
            emitted.append(cnt)                    # quotient digit, MSB-first order
            s -= 1
            if s < 0:
                emitted += ["R"] + rd[:]           # remainder LSB-first marker
                tape_mode = NDONE
            else:
                tape_mode, cnt = NSUB, 0
    q = "".join(map(str, [e for e in emitted[:emitted.index("R")] if isinstance(e, int)])) \
        if "R" in emitted else "".join(map(str, emitted))
    try:
        ri = emitted.index("R")
        r = "".join(map(str, emitted[ri + 1:][::-1]))
    except ValueError:
        r = ""
    q = q.lstrip("0") or "0"; r = r.lstrip("0") or "0"
    return q, r

# ---------------- oracle: encoding existence proof ----------------
rng = random.Random(77); bad = 0; cases = 0
def check(N, D):
    global bad, cases
    q, r = run_bigdiv(N, D)
    qq, rr = divmod(N, D)
    bad += (int(q), int(r)) != (qq, rr); cases += 1

for _ in range(500):                                 # random: N 1..40 digits, D 1..N
    sn = rng.randrange(1, 41); sd = rng.randrange(1, sn + 1)
    N = rng.randrange(10 ** (sn - 1), 10 ** sn)
    D = rng.randrange(10 ** (sd - 1), 10 ** sd) if sd > 1 else rng.randrange(1, 10)
    if D > N: N, D = D, N
    check(N, D)
for N, D in [(1, 1), (10, 1), (100, 10), (99, 99), (100, 99), (7, 7), (120, 20),
             (10**19, 3), (10**19 - 1, 10**9), (123456789, 987654321 // 7),
             (10**39, 10**19), (999999999, 1)]:
    if N >= D: check(N, D)
print(f"[oracle] big/big division encoding: {cases-bad}/{cases} exact "
      f"({'OK — encoding exists' if bad == 0 else 'BROKEN'})", flush=True)
assert bad == 0

# pass-count statistics (work complexity of the iteration)
rng = random.Random(78); stats = []
for sn, sd in [(8, 4), (20, 10), (40, 20), (40, 35)]:
    ps = []
    for _ in range(20):
        N = rng.randrange(10 ** (sn - 1), 10 ** sn); D = rng.randrange(10 ** (sd - 1), 10 ** sd)
        import contextlib
        with contextlib.redirect_stdout(None):
            pass
        # count passes directly
        def count_passes(N, D):
            tot = 0; sn2, sd2 = len(str(N)), len(str(D)); L = sn2 + 1
            rd = [int(c) for c in str(N)[::-1]] + [0] * (L - sn2)
            dd = [int(c) for c in str(D)][::-1]; s = sn2 - sd2
            d_at = lambda i, sh: dd[i - sh] if 0 <= i - sh < len(dd) else 0
            mode, cnt = NSUB, 0
            for _ in range(4000):
                tot += 1
                if mode == NDONE: return tot - 1
                if mode == NSUB:
                    b = 0
                    for i in range(L):
                        t = rd[i] - d_at(i, s) - b; rd[i] = t % 10; b = 1 if t < 0 else 0
                    if b == 0: cnt += 1
                    else: mode = NREP
                else:
                    carry = 0
                    for i in range(L):
                        u = rd[i] + d_at(i, s) + carry; rd[i] = u % 10; carry = u // 10
                    s -= 1
                    if s < 0: mode = NDONE
                    else: mode, cnt = NSUB, 0
            return 4000
        ps.append(count_passes(N, D))
    stats.append(dict(N_digits=sn, D_digits=sd, passes_mean=sum(ps) / len(ps),
                      passes_max=max(ps)))
print("[stats] passes to fixpoint:", json.dumps(stats), flush=True)

res = dict(tag="ARC2-C7-P11M4b-ORACLE", encoding_exact=bool(bad == 0), cases=cases,
           pass_stats=stats, wall_s=round(time.time() - t0, 1))
print("RESULT " + json.dumps(res), flush=True)
open("log.jsonl", "a").write(json.dumps(res) + "\n")
