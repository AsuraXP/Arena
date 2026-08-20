"""RoleOpPRAM: each op selects from a FIXED role basis of permutations via learned
softmax (structure SELECTION, not discovery). Roles for k=12, blocks 6+6:
0=identity 1=A+1 2=A-1 3=B+1 4=B-1 5=both+1."""
import torch, torch.nn as nn, torch.nn.functional as F
from models2 import st_onehot

def role_basis(k=12):
    def shift(block, d):
        p = list(range(k))
        for i in block: p[block[(block.index(i) + d) % len(block)]] = i
        P = torch.zeros(k, k)
        for r in range(k): P[p.index(r) if False else 0, 0] = 0  # placeholder
        # build correctly: new[r] = old[pinv[r]] where item at lane i -> lane (i+d) in block
        P = torch.zeros(k, k)
        m = {i: i for i in range(k)}
        for idx, i in enumerate(block): m[i] = block[(idx + d) % len(block)]
        for i in range(k): P[m[i], i] = 1.0
        return P
    A, B, F_ = list(range(6)), list(range(6, 12)), list(range(k))
    I = torch.eye(k)
    return torch.stack([I, shift(A, 1), shift(A, -1), shift(B, 1), shift(B, -1),
                        shift(A, 1) @ shift(B, 1), shift(F_, 1), shift(F_, -1)])

class RoleOpLayer(nn.Module):
    def __init__(self, d_model, k=12, d_slot=16, n_ops=16, n_vals=3,
                 hard=False, use_scan=True, fixed_isa=False):
        super().__init__()
        self.k, self.n_ops, self.hard, self.use_scan = k, n_ops, hard, use_scan
        self.fixed_isa = fixed_isa
        self.register_buffer("roles", role_basis(k))          # (6,k,k)
        if fixed_isa:
            nr = self.roles.shape[0]
            self.n_ops = n_ops = 2 * nr            # ops = roles x {write, no-write}
            idx = torch.arange(2 * nr) % nr
            self.register_buffer("isa_onehot", F.one_hot(idx, nr).float())
        else:
            self.rlog = nn.Parameter(torch.randn(n_ops, self.roles.shape[0]))
        self.register_buffer("gbits", (torch.arange(n_ops) < n_ops // 2).float())
        self.wlog  = nn.Parameter(torch.randn(n_ops, k))
        self.S0    = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.alpha = nn.Linear(d_model, n_ops)
        self.beta  = nn.Linear(d_model, n_vals)
        self.vcode = nn.Parameter(torch.randn(n_vals, d_slot))
        self.readq = nn.Linear(d_model, k)
        self.out   = nn.Linear(d_slot + d_model, d_model)

    def forward(self, h):
        B, L, _ = h.shape
        rl = self.isa_onehot if self.fixed_isa else F.softmax(self.rlog, dim=-1)
        w = F.softmax(self.wlog, dim=-1)
        a = F.softmax(self.alpha(h), dim=-1)
        q = F.softmax(self.readq(h), dim=-1)
        beta = F.softmax(self.beta(h), dim=-1)
        if self.hard:
            if not self.fixed_isa: rl = st_onehot(rl)
            w, a, q, beta = st_onehot(w), st_onehot(a), st_onehot(q), st_onehot(beta)
        P = torch.einsum("or,rij->oij", rl, self.roles)
        gb = self.gbits.view(-1, 1, 1)
        M = P - gb * torch.einsum("oij,oj,ol->oil", P, w, w)
        u = self.gbits.unsqueeze(-1) * torch.einsum("oij,oj->oi", P, w)
        A = torch.einsum("blo,oij->blij", a, M)
        uv = torch.einsum("blo,oi->bli", a, u)
        v = beta @ self.vcode
        b = uv.unsqueeze(-1) * v.unsqueeze(-2)
        S0 = self.S0.expand(B, -1, -1)
        if self.use_scan:
            Ac, bc, off = A, b, 1
            I = torch.eye(self.k, device=h.device)
            while off < L:
                Ap = F.pad(Ac, (0, 0, 0, 0, off, 0))[:, :L]
                Ap[:, :off] = I
                bp = F.pad(bc, (0, 0, 0, 0, off, 0))[:, :L]
                bc = torch.matmul(Ac, bp) + bc
                Ac = torch.matmul(Ac, Ap)
                off *= 2
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

class RoleOpPRAM(nn.Module):
    def __init__(self, vocab_in, vocab_out, d_model=32, k=12, d_slot=16, n_ops=16,
                 hard=False, use_scan=True, fixed_isa=False, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_in, d_model)
        self.layers = nn.ModuleList([RoleOpLayer(d_model, k, d_slot, n_ops,
                                                 vocab_out, hard, use_scan,
                                                 fixed_isa=fixed_isa)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_out)

    def forward(self, x):
        h = self.emb(x)
        for l in self.layers: h = l(h)
        lyr = self.layers[-1]
        return self.head(self.norm(h)) + lyr.last_r @ lyr.vcode.t()
