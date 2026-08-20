"""Polyglot oracle-LM: union vocab = dyck(0-3) | morph(4-9) | abc(10-12)."""
import math, random, torch
from tasks10 import gen_lm
from tasks12 import gen_morph

def gen_abc_lm(batch, length, g=None, nmax=5):
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else random.randrange(2**31)
    rng = random.Random(seed)
    L2 = math.log(2.0)
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x, nll = [], []
        while len(x) <= length:
            n = 1
            x.append(0); nll.append(0.0 if len(x) == 1 else 0.0)   # first 'a' forced
            while n < nmax and rng.random() < 0.5:
                x.append(0); nll.append(L2); n += 1
            # transition a->b was the .5 choice; first b:
            if n < nmax: nll[-0:] = nll[-0:]
            x.append(1); nll.append(0.0 if n == nmax else L2)      # first b
            for j in range(1, n): x.append(1); nll.append(0.0)
            for j in range(n): x.append(2); nll.append(0.0)
        xs.append(x[:length]); ys.append([t for t in x[1:length + 1]])
        os_.append(nll[1:length + 1])
    X = torch.tensor(xs) + 10
    Y = torch.tensor(ys) + 10
    return X, Y, torch.tensor(os_)

FAMS = {
    "dyck":  lambda b, L, g=None: (lambda x, y, o: (x, y, o))(*gen_lm(b, L, g)),
    "morph": lambda b, L, g=None: (lambda x, y, o: (x + 4, y + 4, o))(*gen_morph(b, L, g)),
    "abc":   gen_abc_lm,
}

def gen_poly(batch, length, g=None):
    xs, ys, os_ = [], [], []
    keys = list(FAMS)
    for _ in range(batch):
        f = keys[int(torch.randint(0, 3, (1,), generator=g).item())]
        x, y, o = FAMS[f](1, length, g)
        xs.append(x[0]); ys.append(y[0]); os_.append(o[0])
    return torch.stack(xs), torch.stack(ys), torch.stack(os_)
