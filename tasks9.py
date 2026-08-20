"""mixed3: three families in one stream space.
tokens: 0-6 dyck2p | 7-11 agree(+7) | 12-15 abcp(+12); outputs: 0-2 dyck | 3-5 agree |
6-15 abcp(+6)."""
import torch
from tasks3 import gen_dyck2p
from tasks5 import gen_agree
from tasks4 import gen_abcp

def gen_mixed3(batch, length, g=None):
    xs, ys = [], []
    for _ in range(batch):
        pick = int(torch.randint(0, 3, (1,), generator=g).item())
        if pick == 0:
            x, y, _, _ = gen_dyck2p(1, length, g); xs.append(x[0]); ys.append(y[0])
        elif pick == 1:
            x, y, _, _ = gen_agree(1, length, g); xs.append(x[0] + 7); ys.append(y[0] + 3)
        else:
            x, y, _, _ = gen_abcp(1, length, g); xs.append(x[0] + 12); ys.append(y[0] + 6)
    return torch.stack(xs), torch.stack(ys), 16, 16

def fam3(f):
    def gen(batch, length, g=None):
        if f == "dyck":
            x, y, _, _ = gen_dyck2p(batch, length, g); return x, y, 16, 16
        if f == "agree":
            x, y, _, _ = gen_agree(batch, length, g); return x + 7, y + 3, 16, 16
        x, y, _, _ = gen_abcp(batch, length, g); return x + 12, y + 6, 16, 16
    return gen
