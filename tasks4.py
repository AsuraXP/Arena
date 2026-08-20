"""anbncn streams: blocks a^n b^n c^n, n in [1,nmax]. tokens a=0 b=1 c=2.
Dense per-token target = obligation class after this token:
0=FREE (next a|b), 1=forced B, 2=forced C, 3=block complete (next must be 'a')."""
import random, torch

def _seq(length, rng, nmax=5):
    x, y = [], []
    while len(x) < length:
        n = rng.randint(1, nmax)
        for i in range(n):            # a-phase
            x.append(0); y.append(0)
        for j in range(1, n + 1):     # b-phase
            x.append(1); y.append(1 if j < n else 2)
        for j in range(1, n + 1):     # c-phase
            x.append(2); y.append(2 if j < n else 3)
    return x[:length], y[:length]

def gen_abc(batch, length, g=None, nmax=5):
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else random.randrange(2**31)
    rng = random.Random(seed)
    xs, ys = zip(*[_seq(length, rng, nmax) for _ in range(batch)])
    return torch.tensor(xs), torch.tensor(ys), 3, 4

def _seq_p(length, rng, nmax=5, p_probe=0.15):
    # tokens: 0=a 1=b 2=c 3=PROBE(countA). outputs: 0..3 obligations, 4..9 = countA 0..5
    x, y = [], []
    ca = 0; phase = 'a'; n = 0; j = 0
    def emit_probe():
        x.append(3); y.append(4 + ca)
    while len(x) < length:
        n = rng.randint(1, nmax)
        ca = 0
        for i in range(n):
            if rng.random() < p_probe: emit_probe()
            x.append(0); ca += 1; y.append(0)
        for j in range(1, n + 1):
            if rng.random() < p_probe: emit_probe()
            x.append(1); ca -= 1; y.append(1 if j < n else 2)
        for j in range(1, n + 1):
            if rng.random() < p_probe: emit_probe()
            x.append(2); y.append(2 if j < n else 3)
    return x[:length], y[:length]

def gen_abcp(batch, length, g=None, nmax=5, p_probe=0.15):
    import random as _r
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else _r.randrange(2**31)
    rng = _r.Random(seed)
    xs, ys = zip(*[_seq_p(length, rng, nmax, p_probe) for _ in range(batch)])
    return torch.tensor(xs), torch.tensor(ys), 4, 10
