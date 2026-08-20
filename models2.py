"""PRAM: Permutation-Routed Associative Machine.
Affine recurrence  S_t = A_t S_{t-1} + b_t  with
  A_t = R_t (I - g_t w_t w_t^T)   (permute after optional delta-overwrite)
  b_t = g_t R_t w_t v_t^T
R_t: Sinkhorn-codebook (hard-perm ST), w_t: one-hot ST, g_t: binary ST.
Supports sequential loop or Hillis-Steele associative scan (O(log L) depth).
"""
import torch, torch.nn as nn, torch.nn.functional as F
from models import sinkhorn, hard_perm

def st_onehot(p):                       # straight-through argmax over last dim
    h = torch.zeros_like(p).scatter_(-1, p.argmax(-1, keepdim=True), 1.0)
    return h + p - p.detach()

def st_binary(p):                       # straight-through threshold
    return (p > 0.5).float() + p - p.detach()

class PRAMLayer(nn.Module):
    def __init__(self, d_model, k=6, d_slot=16, tau=0.5, sink_iters=5, n_proto=12,
                 hard=False, use_scan=False, n_vals=0):
        super().__init__()
        self.k, self.d_slot, self.tau, self.sink_iters = k, d_slot, tau, sink_iters
        self.hard, self.use_scan = hard, use_scan
        self.protos = nn.Parameter(torch.randn(n_proto, k, k))
        self.S0 = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.alpha = nn.Linear(d_model, n_proto)
        self.sel   = nn.Linear(d_model, k)          # write address
        self.n_vals = n_vals
        if n_vals > 0:                               # tied discrete value codebook
            self.beta  = nn.Linear(d_model, n_vals)
            self.vcode = nn.Parameter(torch.randn(n_vals, d_slot))
        else:
            self.val   = nn.Linear(d_model, d_slot)  # continuous write content
        self.gate  = nn.Linear(d_model, 1)
        nn.init.constant_(self.gate.bias, -3.0)     # writes opt-in
        self.rho   = nn.Linear(d_model, 1)          # routing opt-in gate
        nn.init.constant_(self.rho.bias, -2.0)
        self.readq = nn.Linear(d_model, k)          # addressed read
        self.readf = nn.Linear(k * d_slot, d_model) # full-state read
        self.out   = nn.Linear(d_slot + 2 * d_model, d_model)

    def _ops(self, h):
        B, L, _ = h.shape
        P = sinkhorn(self.protos / self.tau, self.sink_iters)
        a = F.softmax(self.alpha(h), dim=-1)
        w = F.softmax(self.sel(h), dim=-1)
        q = F.softmax(self.readq(h), dim=-1)
        g = torch.sigmoid(self.gate(h))
        rho = torch.sigmoid(self.rho(h))
        if self.hard:
            P = hard_perm(P) + P - P.detach()
            a, w, q, g = st_onehot(a), st_onehot(w), st_onehot(q), st_binary(g)
            rho = st_binary(rho)
        elif getattr(self, "hard_gates", False):
            g, rho = st_binary(g), st_binary(rho)
        R = torch.einsum("blm,mij->blij", a, P)                    # (B,L,k,k)
        I0 = torch.eye(self.k, device=h.device).expand_as(R)
        R = (1 - rho.unsqueeze(-1)) * I0 + rho.unsqueeze(-1) * R
        if self.n_vals > 0:
            beta = F.softmax(self.beta(h), dim=-1)
            if self.hard: beta = st_onehot(beta)
            v = beta @ self.vcode                                  # (B,L,d)
        else:
            v = torch.tanh(self.val(h))                            # (B,L,d)
        gw = g.unsqueeze(-1) * w.unsqueeze(-1)                     # (B,L,k,1)
        I = torch.eye(self.k, device=h.device)
        A = torch.matmul(R, I - gw * w.unsqueeze(-2))              # R(I - g w w^T)
        b = torch.matmul(R, gw * v.unsqueeze(-2))                  # g R w v^T
        return A, b, q

    def forward(self, h):
        B, L, _ = h.shape
        A, b, q = self._ops(h)
        S0 = self.S0.expand(B, -1, -1)
        if self.use_scan:                     # Hillis-Steele inclusive scan
            Ac, bc, off = A, b, 1
            while off < L:
                Ap = F.pad(Ac, (0, 0, 0, 0, off, 0))[:, :L]        # shift right, I-pad
                Ap[:, :off] = torch.eye(self.k, device=h.device)
                bp = F.pad(bc, (0, 0, 0, 0, off, 0))[:, :L]
                bc = torch.matmul(Ac, bp) + bc
                Ac = torch.matmul(Ac, Ap)
                off *= 2
            S = torch.matmul(Ac, S0.unsqueeze(1)) + bc             # (B,L,k,d)
        else:
            St, outs = S0, []
            for t in range(L):
                St = torch.matmul(A[:, t], St) + b[:, t]
                outs.append(St)
            S = torch.stack(outs, dim=1)
        r_addr = torch.einsum("blk,blkd->bld", q, S)
        self.last_r_addr = r_addr
        r_flat = self.readf(S.reshape(B, L, -1))
        if not getattr(self, "use_flat", True):
            r_flat = r_flat * 0.0
        return h + self.out(torch.cat([r_addr, r_flat, h], dim=-1))

class PRAM(nn.Module):
    def __init__(self, vocab_in, vocab_out, d_model=32, k=6, d_slot=16, n_layers=1,
                 tau=0.5, sink_iters=5, n_proto=12, hard=False, use_scan=False,
                 tie_vals=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_in, d_model)
        self.tie_vals = tie_vals
        self.layers = nn.ModuleList([PRAMLayer(d_model, k, d_slot, tau, sink_iters,
                                               n_proto, hard, use_scan,
                                               n_vals=vocab_out if tie_vals else 0)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_out)
        self.aux_head = nn.Linear(d_model, 5 * 8)   # dense state probe (train only)

    def forward(self, x, with_aux=False):
        h = self.emb(x)
        for l in self.layers: h = l(h)
        hn = self.norm(h)
        logits = self.head(hn)
        if self.tie_vals:
            lyr = self.layers[-1]
            logits = logits + lyr.last_r_addr @ lyr.vcode.t()
        if with_aux:
            return logits, self.aux_head(hn).view(*x.shape, 5, 8)
        return logits
