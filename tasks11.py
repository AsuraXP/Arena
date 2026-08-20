"""Stochastic modal-dyck LM with analytic oracle.
tokens: 0=M0 1=M1 2='(' 3=')' 4='[' 5=']'. mode swaps bracket semantics.
Policy: p_mode=0.1 (uniform M0/M1); else depth policy (D=5): d=0 push (eff type U2);
d=D forced close (raw closer for eff top under mode); else push .5 / close .5."""
import math, random, torch

def gen_modal_lm(batch, length, g=None, D=5, p_mode=0.1):
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else random.randrange(2**31)
    rng = random.Random(seed)
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x, nll, stack, mode = [], [], [], 0
        for _ in range(length + 1):
            d = len(stack)
            if p_mode > 0 and rng.random() < p_mode:
                mode = rng.randrange(2)
                x.append(mode); nll.append(-math.log(p_mode / 2))
                continue
            SC = 1.0 - p_mode                       # oracle mass after mode reservation
            if d == 0:
                eff = rng.randrange(2); stack.append(eff)
                raw = eff if mode == 0 else 1 - eff
                x.append(2 + 2 * raw); nll.append(-math.log(SC * 0.5))
            elif d == D:
                eff = stack.pop(); raw = eff if mode == 0 else 1 - eff
                x.append(3 + 2 * raw); nll.append(-math.log(SC))
            else:
                if rng.random() < 0.5:
                    eff = rng.randrange(2); stack.append(eff)
                    raw = eff if mode == 0 else 1 - eff
                    x.append(2 + 2 * raw); nll.append(-math.log(SC * 0.25))
                else:
                    eff = stack.pop(); raw = eff if mode == 0 else 1 - eff
                    x.append(3 + 2 * raw); nll.append(-math.log(SC * 0.5))
        xs.append(x[:length]); ys.append(x[1:length + 1]); os_.append(nll[1:length + 1])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)
