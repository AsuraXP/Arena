"""Stochastic-Dyck language modeling with analytic oracle.
Policy (depth d, D=6): d=0 -> push type~U2; d=D -> forced matching close;
else push .5 (type U2) / matching close .5. Tokens 0='(' 1=')' 2='[' 3=']'.
Returns x, y_next, oracle_nll (per position, last position ignored)."""
import math, random, torch

def gen_lm(batch, length, g=None, D=6):
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else random.randrange(2**31)
    rng = random.Random(seed)
    L2, L4 = math.log(2.0), math.log(4.0)
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x, nll, stack = [], [], []
        for _ in range(length + 1):
            d = len(stack)
            if d == 0:
                t = rng.randrange(2); stack.append(t); x.append(2 * t); nll.append(L2)
            elif d == D:
                t = stack.pop(); x.append(2 * t + 1); nll.append(0.0)
            else:
                r = rng.random()
                if r < 0.5:
                    t = rng.randrange(2); stack.append(t); x.append(2 * t)
                    nll.append(L4)
                else:
                    t = stack.pop(); x.append(2 * t + 1); nll.append(L2)
        xs.append(x[:length]); ys.append(x[1:length + 1]); os_.append(nll[1:length + 1])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)
