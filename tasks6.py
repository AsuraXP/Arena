"""Mixed-skill streams: dyck2p (tokens 0-6, classes 0-2) U agree (tokens 7-11 -> +7,
classes 3-5). Each sequence is one family."""
import torch
from tasks3 import gen_dyck2p
from tasks5 import gen_agree

def gen_mixed(batch, length, g=None):
    xs, ys = [], []
    for i in range(batch):
        pick = int(torch.randint(0, 2, (1,), generator=g).item())
        if pick == 0:
            x, y, _, _ = gen_dyck2p(1, length, g)
            xs.append(x[0]); ys.append(y[0])
        else:
            x, y, _, _ = gen_agree(1, length, g)
            xs.append(x[0] + 7); ys.append(y[0] + 3)
    return torch.stack(xs), torch.stack(ys), 12, 6

def gen_mixed_family(fam):
    def gen(batch, length, g=None):
        if fam == "dyck":
            x, y, _, _ = gen_dyck2p(batch, length, g); return x, y, 12, 6
        x, y, _, _ = gen_agree(batch, length, g); return x + 7, y + 3, 12, 6
    return gen
