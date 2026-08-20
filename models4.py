"""CycleOpPRAM: ops are block-cyclic rotations with LEARNED OFFSET DISTRIBUTIONS.
P_o = blockdiag_b( sum_d soft(off[o,b])_d * R_b^d ).  Hard mode: argmax offsets."""
import torch, torch.nn as nn, torch.nn.functional as F
from models2 import st_onehot

def _rot_basis(s):
    B = torch.zeros(s, s, s)
    for d in range(s):
        for i in range(s): B[d, (i + d) % s, i] = 1.0
    return B                                            # B[d] = R^d

class CycleOpLayer(nn.Module):
    def __init__(self, d_model, blocks=(6, 6), d_slot=16, n_ops=16, n_vals=3,
                 hard=False, use_scan=True):
        super().__init__()
        self.blocks, self.k = blocks, sum(blocks)
        self.n_ops, self.hard, self.use_scan, self.d_slot = n_ops, hard, use_scan, d_slot
        self.off = nn.ParameterList([nn.Parameter(torch.randn(n_ops, s)) for s in blocks])
        for bi, s in enumerate(blocks):
            self.register_buffer(f"basis{bi}", _rot_basis(s))
        self.register_buffer("gbits", (torch.arange(n_ops) < n_ops // 2).float())
        self.wlog  = nn.Parameter(torch.randn(n_ops, self.k))
        self.S0    = nn.Parameter(0.5 * torch.randn(self.k, d_slot))
        self.alpha = nn.Linear(d_model, n_ops)
        self.beta  = nn.Linear(d_model, n_vals)
        self.vcode = nn.Parameter(torch.randn(n_vals, d_slot))
        self.readq = nn.Linear(d_model, self.k)
        self.out   = nn.Linear(d_slot + d_model, d_model)

    def op_perms(self):
        mats, pos = [], 0
        P = torch.zeros(self.n_ops, self.k, self.k, device=self.wlog.device)
        for bi, s in enumerate(self.blocks):
            off = F.softmax(self.off[bi], dim=-1)                  # (O,s)
            if self.hard: off = st_onehot(off)
            blk = torch.einsum("od,dij->oij", off, getattr(self, f"basis{bi}"))
            P[:, pos:pos + s, pos:pos + s] = blk
            pos += s
        return P

    def forward(self, h):
        B, L, _ = h.shape
        P = self.op_perms()
        w = F.softmax(self.wlog, dim=-1)
        a = F.softmax(self.alpha(h), dim=-1)
        q = F.softmax(self.readq(h), dim=-1)
        beta = F.softmax(self.beta(h), dim=-1)
        if self.hard:
            w, a, q, beta = st_onehot(w), st_onehot(a), st_onehot(q), st_onehot(beta)
        gb = self.gbits.view(-1, 1, 1)
        M = P - gb * torch.einsum("oij,oj,ol->oil", P, w, w)
        u = self.gbits.unsqueeze(-1) * torch.einsum("oij,oj->oi", P, w)
        A = torch.einsum("blo,oij->blij", a, M)
        uv = torch.einsum("blo,oi->bli", a, u)
        v = beta @ self.vcode
        b = uv.unsqueeze(-1) * v.unsqueeze(-2)
        S0 = self.S0.expand(B, -1, -1)
        if self.use_scan:
            Ac, bc, off_ = A, b, 1
            I = torch.eye(self.k, device=h.device)
            while off_ < L:
                Ap = F.pad(Ac, (0, 0, 0, 0, off_, 0))[:, :L]
                Ap[:, :off_] = I
                bp = F.pad(bc, (0, 0, 0, 0, off_, 0))[:, :L]
                bc = torch.matmul(Ac, bp) + bc
                Ac = torch.matmul(Ac, Ap)
                off_ *= 2
            S = torch.matmul(Ac, S0.unsqueeze(1)) + bc
        else:
            St, outs = S0, []
            for t in range(L):
                St = torch.matmul(A[:, t], St) + b[:, t]
                outs.append(St)
            S = torch.stack(outs, 1)
        r = torch.einsum("blk,blkd->bld", q, S)
        self.last_r = r
        return h + self.out(torch.cat([r, h], dim=-1))

class CycleOpPRAM(nn.Module):
    def __init__(self, vocab_in, vocab_out, d_model=32, blocks=(6, 6), d_slot=16,
                 n_ops=16, hard=False, use_scan=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_in, d_model)
        self.layers = nn.ModuleList([CycleOpLayer(d_model, blocks, d_slot, n_ops,
                                                  vocab_out, hard, use_scan)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_out)

    def forward(self, x):
        h = self.emb(x)
        for l in self.layers: h = l(h)
        lyr = self.layers[-1]
        return self.head(self.norm(h)) + lyr.last_r @ lyr.vcode.t()
