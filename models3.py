"""PRAM-v3 'OpBook': gate-free program — one softmax selects a fused op
(perm, hardwired write-bit, per-op write slot). No sigmoids anywhere."""
import torch, torch.nn as nn, torch.nn.functional as F
from models import sinkhorn, hard_perm
from models2 import st_onehot

class OpPRAMLayer(nn.Module):
    def __init__(self, d_model, k=8, d_slot=16, tau=0.5, sink_iters=5, n_ops=16,
                 n_vals=3, hard=False, use_scan=True):
        super().__init__()
        self.k, self.d_slot, self.tau, self.sink_iters = k, d_slot, tau, sink_iters
        self.n_ops, self.hard, self.use_scan = n_ops, hard, use_scan
        self.protos = nn.Parameter(torch.randn(n_ops, k, k))
        self.register_buffer("gbits", (torch.arange(n_ops) < n_ops // 2).float())
        self.wlog  = nn.Parameter(torch.randn(n_ops, k))       # per-op write slot
        self.S0    = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.alpha = nn.Linear(d_model, n_ops)                 # THE op selector
        self.beta  = nn.Linear(d_model, n_vals)
        self.vcode = nn.Parameter(torch.randn(n_vals, d_slot))
        self.readq = nn.Linear(d_model, k)
        self.out   = nn.Linear(d_slot + d_model, d_model)

    def forward(self, h):
        B, L, _ = h.shape
        P = sinkhorn(self.protos / self.tau, self.sink_iters)  # (O,k,k)
        w = F.softmax(self.wlog, dim=-1)                       # (O,k)
        a = F.softmax(self.alpha(h), dim=-1)                   # (B,L,O)
        q = F.softmax(self.readq(h), dim=-1)                   # (B,L,k)
        beta = F.softmax(self.beta(h), dim=-1)                 # (B,L,V)
        if self.hard:
            P = hard_perm(P) + P - P.detach()
            w, a, q, beta = st_onehot(w), st_onehot(a), st_onehot(q), st_onehot(beta)
        gb = self.gbits.view(-1, 1, 1)
        M = P - gb * torch.einsum("oij,oj,ol->oil", P, w, w)   # P_o(I - g_o w_o w_o^T)
        u = self.gbits.unsqueeze(-1) * torch.einsum("oij,oj->oi", P, w)  # g_o P_o w_o
        A = torch.einsum("blo,oij->blij", a, M)                # (B,L,k,k)
        uv = torch.einsum("blo,oi->bli", a, u)                 # (B,L,k)
        v = beta @ self.vcode                                  # (B,L,d)
        b = uv.unsqueeze(-1) * v.unsqueeze(-2)                 # (B,L,k,d)
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

class OpPRAM(nn.Module):
    def __init__(self, vocab_in, vocab_out, d_model=32, k=8, d_slot=16, n_ops=16,
                 tau=0.5, hard=False, use_scan=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_in, d_model)
        self.layers = nn.ModuleList([OpPRAMLayer(d_model, k, d_slot, tau, 5, n_ops,
                                                 vocab_out, hard, use_scan)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_out)

    def forward(self, x):
        h = self.emb(x)
        for l in self.layers: h = l(h)
        lyr = self.layers[-1]
        return self.head(self.norm(h)) + lyr.last_r @ lyr.vcode.t()
