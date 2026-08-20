"""w#wR streams: blocks of random word w (a/b, len 1..nmax), '#', then reversed w.
Targets (obligation after token): free=2 during w; at '#' and during wR: forced next
symbol (0/1); after final wR symbol: 3 (new block). vocab_in=3, vocab_out=4."""
import random, torch

def _seq(length, rng, nmax=8):
    x, y = [], []
    while len(x) < length:
        n = rng.randint(1, nmax)
        w = [rng.randrange(2) for _ in range(n)]
        for i, s in enumerate(w):
            x.append(s); y.append(2)
        x.append(2); y.append(w[-1])                 # '#': next forced
        for j in range(n):
            x.append(w[n - 1 - j])
            y.append(w[n - 2 - j] if j < n - 1 else 3)
    return x[:length], y[:length]

def gen_wwr(batch, length, g=None, nmax=8):
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else random.randrange(2**31)
    rng = random.Random(seed)
    xs, ys = zip(*[_seq(length, rng, nmax) for _ in range(batch)])
    return torch.tensor(xs), torch.tensor(ys), 3, 4
