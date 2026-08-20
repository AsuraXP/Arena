"""modal-dyck: mode tokens M0/M1 relabel bracket semantics (swap ( <-> [ while M1).
tokens: 0=M0 1=M1 2='(' 3=')' 4='[' 5=']'. Target = EFFECTIVE stack-top type
(0=empty 1=P 2=B) after token. depth<=5."""
import random, torch

def _seq(length, rng, max_depth=5, p_mode=0.1, p_probe=0.0):
    x, y, stack, mode = [], [], [], 0
    for _ in range(length):
        if p_probe > 0 and rng.random() < p_probe:
            x.append(6); y.append(3 + mode)          # MP token: target = current mode
            continue
        if rng.random() < p_mode:
            mode = rng.randrange(2)
            x.append(mode); y.append(0 if not stack else stack[-1] + 1)
            continue
        if not stack: push = True
        elif len(stack) >= max_depth: push = False
        else: push = rng.random() < 0.5
        if push:
            eff = rng.randrange(2)                   # effective type to push
            raw = eff if mode == 0 else 1 - eff      # symbol whose eff-type = eff
            stack.append(eff); x.append(2 + 2 * raw)
        else:
            eff = stack.pop()
            raw = eff if mode == 0 else 1 - eff
            x.append(3 + 2 * raw)
        y.append(0 if not stack else stack[-1] + 1)
    return x, y

def gen_modal(batch, length, g=None, max_depth=5, p_mode=0.1, p_probe=0.0):
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else random.randrange(2**31)
    rng = random.Random(seed)
    xs, ys = zip(*[_seq(length, rng, max_depth, p_mode, p_probe) for _ in range(batch)])
    if p_probe > 0:
        return torch.tensor(xs), torch.tensor(ys), 7, 5
    return torch.tensor(xs), torch.tensor(ys), 6, 3

def gen_modalp(batch, length, g=None, max_depth=5):
    return gen_modal(batch, length, g, max_depth, p_probe=0.15)
