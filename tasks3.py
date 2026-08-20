"""dyck2: bounded-depth Dyck-2 stream. Per-token target = stack-top TYPE after the
token (0=empty, 1='(', 2='['). Well-formed prefixes, depth <= max_depth.
Tokens: 0='(' 1=')' 2='[' 3=']'. vocab_in=4, vocab_out=3.
"""
import random, torch

def _seq(length, rng, max_depth=6):
    x, y, stack = [], [], []
    for _ in range(length):
        if not stack:                push = True
        elif len(stack) >= max_depth: push = False
        else:                        push = rng.random() < 0.5
        if push:
            t = rng.randrange(2)             # 0 -> '(', 1 -> '['
            stack.append(t)
            x.append(0 if t == 0 else 2)
        else:
            t = stack.pop()
            x.append(1 if t == 0 else 3)
        y.append(0 if not stack else stack[-1] + 1)
    return x, y

def gen_dyck2(batch, length, g=None, max_depth=6):
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else random.randrange(2**31)
    rng = random.Random(seed)
    xs, ys = zip(*[_seq(length, rng, max_depth) for _ in range(batch)])
    return torch.tensor(xs), torch.tensor(ys), 4, 3

def _seq_p(length, rng, max_depth=6, p_peek=0.2):
    # tokens: 0='(' 1=')' 2='[' 3=']' 4=PEEK0 5=PEEK1 6=PEEK2
    x, y, stack = [], [], []
    for _ in range(length):
        if rng.random() < p_peek:
            d = rng.randrange(3)
            x.append(4 + d)
            y.append(0 if len(stack) <= d else stack[-1 - d] + 1)
            continue
        if not stack:                 push = True
        elif len(stack) >= max_depth: push = False
        else:                         push = rng.random() < 0.5
        if push:
            t = rng.randrange(2); stack.append(t)
            x.append(0 if t == 0 else 2)
        else:
            t = stack.pop()
            x.append(1 if t == 0 else 3)
        y.append(0 if not stack else stack[-1] + 1)
    return x, y

def gen_dyck2p(batch, length, g=None, max_depth=6, p_peek=0.2):
    import random as _r
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else _r.randrange(2**31)
    rng = _r.Random(seed)
    xs, ys = zip(*[_seq_p(length, rng, max_depth, p_peek) for _ in range(batch)])
    return torch.tensor(xs), torch.tensor(ys), 7, 3

def _seq3(length, rng, max_depth=8, p_peek=0.2):
    # tokens: 0/2/4 = open ( [ { ; 1/3/5 = close ) ] } ; 6,7,8 = PEEK0..2
    x, y, stack = [], [], []
    for _ in range(length):
        if rng.random() < p_peek:
            d = rng.randrange(3)
            x.append(6 + d)
            y.append(0 if len(stack) <= d else stack[-1 - d] + 1)
            continue
        if not stack:                 push = True
        elif len(stack) >= max_depth: push = False
        else:                         push = rng.random() < 0.5
        if push:
            t = rng.randrange(3); stack.append(t); x.append(2 * t)
        else:
            t = stack.pop(); x.append(2 * t + 1)
        y.append(0 if not stack else stack[-1] + 1)
    return x, y

def gen_dyck3p(batch, length, g=None, max_depth=8, p_peek=0.2):
    import random as _r
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else _r.randrange(2**31)
    rng = _r.Random(seed)
    xs, ys = zip(*[_seq3(length, rng, max_depth, p_peek) for _ in range(batch)])
    return torch.tensor(xs), torch.tensor(ys), 9, 4

def gen_dyck3(batch, length, g=None, max_depth=8):
    return gen_dyck3p(batch, length, g, max_depth, p_peek=0.0)
