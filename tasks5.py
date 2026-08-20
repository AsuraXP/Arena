"""Nested subject-verb agreement streams (Lakretz-style, synthetic).
tokens: 0=S_sg 1=S_pl (open clause, push feature) 2=V (close clause, pop; must agree)
        3=N_sg 4=N_pl (distractor nouns, no-op). depth <= max_depth.
targets: at V -> feature popped (0=sg 1=pl); elsewhere -> 2 (n/a). Dense."""
import random, torch

def _seq(length, rng, max_depth=5, p_noun=0.35):
    x, y, stack = [], [], []
    for _ in range(length):
        r = rng.random()
        if r < p_noun:
            x.append(3 + rng.randrange(2)); y.append(2); continue
        if not stack: push = True
        elif len(stack) >= max_depth: push = False
        else: push = rng.random() < 0.5
        if push:
            f = rng.randrange(2); stack.append(f)
            x.append(f); y.append(2)
        else:
            f = stack.pop()
            x.append(2); y.append(f)
    return x, y

def gen_agree(batch, length, g=None, max_depth=5):
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else random.randrange(2**31)
    rng = random.Random(seed)
    xs, ys = zip(*[_seq(length, rng, max_depth) for _ in range(batch)])
    return torch.tensor(xs), torch.tensor(ys), 5, 3
