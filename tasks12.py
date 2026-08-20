"""Morphology-LM: stochastic nested agreement with analytic oracle.
tokens: 0=S_sg 1=S_pl (open clause, push feature) 2=V_sg 3=V_pl (verb closes clause,
form MUST agree with top) 4=N_sg 5=N_pl (distractor nouns). D=5.
Policy: p_noun=.3 (forms uniform); of remaining .7: empty->open only (each .35);
depth=D->verb only (correct form mass .7); else open .35 (each .175)/verb .35."""
import math, random, torch

def gen_morph(batch, length, g=None, D=5):
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) if g is not None \
           else random.randrange(2**31)
    rng = random.Random(seed)
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x, nll, stack = [], [], []
        for _ in range(length + 1):
            d = len(stack)
            r = rng.random()
            if r < 0.3:
                f = rng.randrange(2); x.append(4 + f); nll.append(-math.log(0.15))
                continue
            if d == 0:
                f = rng.randrange(2); stack.append(f)
                x.append(f); nll.append(-math.log(0.35))
            elif d == D:
                f = stack.pop(); x.append(2 + f); nll.append(-math.log(0.7))
            else:
                if rng.random() < 0.5:
                    f = rng.randrange(2); stack.append(f)
                    x.append(f); nll.append(-math.log(0.175))
                else:
                    f = stack.pop(); x.append(2 + f); nll.append(-math.log(0.35))
        xs.append(x[:length]); ys.append(x[1:length + 1]); os_.append(nll[1:length + 1])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)
