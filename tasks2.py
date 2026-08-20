"""track5: fused NC^1 state-tracking + associative recall benchmark.
Stream of events over 5 logical registers holding values from an 8-symbol alphabet:
  G_i   (5 tokens): apply S5 generator i -> permutes register CONTENTS (cup shuffle)
  STO_jv(40 tokens): overwrite register j with value v
  Q_j   (5 tokens): emit value currently in register j (only supervised positions)
Answering Q_j after many G's requires exact permutation composition (NC^1) applied to
values stored arbitrarily far back (associative recall). Targets elsewhere = -100.
Variant 'far': stores only in prefix, then a long pure-shuffle span, queries at the end
(stress-tests long-range retention + deep composition; used for eval only).
"""
import torch

N_REG, N_VAL, N_GEN = 5, 8, 5
GENS = [(1,0,2,3,4), (1,2,3,4,0), (0,1,2,3,4), (0,2,1,3,4), (4,3,2,1,0)]
TOK_G = 0                      # G tokens: [0,5)
TOK_STO = N_GEN                # STO tokens: [5, 5+40)
TOK_Q = TOK_STO + N_REG*N_VAL  # Q tokens: [45, 50)
VOCAB_IN = TOK_Q + N_REG       # 50
VOCAB_OUT = N_VAL              # 8

def _apply_gen(val, gi):
    # contents permute: new_val[p[j]] = val[j]
    p = GENS[gi]
    return [val[p.index(j)] for j in range(N_REG)]

def _make_seq(length, rng, far=False, pg=0.5, ps=0.3, aux=False):
    x, y, ya = [], [], []
    val = [j % N_VAL for j in range(N_REG)]          # known initial contents
    for t in range(length):
        if far:
            if t < 12:            ev = 'S' if rng.random() < 0.7 else 'G'
            elif t < length - 6:  ev = 'G'
            else:                 ev = 'Q'
        else:
            r = rng.random()
            ev = 'G' if r < pg else ('S' if r < pg + ps else 'Q')
        if ev == 'G':
            gi = rng.randrange(N_GEN)
            val = _apply_gen(val, gi)
            x.append(TOK_G + gi); y.append(-100)
        elif ev == 'S':
            j, v = rng.randrange(N_REG), rng.randrange(N_VAL)
            val[j] = v
            x.append(TOK_STO + j * N_VAL + v); y.append(-100)
        else:
            j = rng.randrange(N_REG)
            x.append(TOK_Q + j); y.append(val[j])
        ya.append(list(val))
    return (x, y, ya) if aux else (x, y)

def gen_track5(batch, length, g=None, far=False, pg=0.5, ps=0.3, aux=False):
    import random
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else random.randrange(2**31)
    rng = random.Random(seed)
    seqs = [_make_seq(length, rng, far, pg, ps, aux) for _ in range(batch)]
    if aux:
        xs, ys, yas = zip(*seqs)
        return (torch.tensor(xs), torch.tensor(ys), torch.tensor(yas),
                VOCAB_IN, VOCAB_OUT)
    xs, ys = zip(*seqs)
    return torch.tensor(xs), torch.tensor(ys), VOCAB_IN, VOCAB_OUT

def gen_track5_far(batch, length, g=None):
    return gen_track5(batch, length, g, far=True)

def gen_recall5(batch, length, g=None):   # D1: no shuffles
    return gen_track5(batch, length, g, pg=0.0, ps=0.6)

def gen_shuffle5(batch, length, g=None):  # D2: no stores
    return gen_track5(batch, length, g, pg=0.7, ps=0.0)

TASKS2 = {"track5": gen_track5, "track5far": gen_track5_far,
          "recall5": gen_recall5, "shuffle5": gen_shuffle5}
