"""FBISA: feedback Krohn-Rhodes ISA machine (trainable).
Mode transition selected per (token, e-bit) from enumerated basis {id,c0,c1,shift} (M=2).
e-bit = ST-thresholded learned probe on previous read. Dispatch/read/value: linear heads
on [emb; mode-onehot; e]. Control state exposed to readout."""
import torch, torch.nn as nn, torch.nn.functional as F
from models2 import st_onehot, st_binary
from models5 import role_basis
from models6 import mode_basis

class FBISA(nn.Module):
    def __init__(self, vocab_in, vocab_out, d_model=32, k=12, d_slot=16, M=2,
                 hard=False):
        super().__init__()
        self.vin, self.vout, self.k, self.M, self.hard = vocab_in, vocab_out, k, M, hard
        self.register_buffer("PH", torch.stack([role_basis(k)[o % 8] for o in range(16)]))
        self.register_buffer("gbits", (torch.arange(16) < 8).float())
        self.register_buffer("TB", mode_basis(M))          # (M+2,M,M)
        self.emb = nn.Embedding(vocab_in, d_model)
        self.mtab = nn.Parameter(torch.randn(vocab_in, 2, M + 2))
        din = d_model + M + 1
        self.alpha = nn.Linear(din, 16)
        nn.init.zeros_(self.alpha.bias); self.alpha.bias.data[8] = 2.0
        self.readq = nn.Linear(din, k)
        self.beta  = nn.Linear(din, vocab_out)
        self.wlog  = nn.Parameter(torch.randn(16, k))
        self.eprobe = nn.Linear(d_slot, 1)
        self.S0    = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.vcode = nn.Parameter(torch.randn(vocab_out, d_slot))
        self.memb  = nn.Parameter(0.1 * torch.randn(M, d_model))
        self.out   = nn.Linear(d_slot + d_model, d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_out)

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x)
        w = F.softmax(self.wlog, dim=-1)
        if self.hard: w = st_onehot(w)
        gb = self.gbits.view(-1, 1, 1)
        Mo = self.PH - gb * torch.einsum("oij,oj,ol->oil", self.PH, w, w)
        u = self.gbits.unsqueeze(-1) * torch.einsum("oij,oj->oi", self.PH, w)
        S = self.S0.expand(B, -1, -1)
        m = torch.zeros(B, self.M, device=x.device); m[:, 0] = 1.0
        e = torch.ones(B, 1, device=x.device)
        reads, mouts = [], []
        for t in range(L):
            md = F.softmax(self.mtab[x[:, t]], dim=-1)          # (B,2,M+2)
            md = md[torch.arange(B), (e.squeeze(-1) > 0.5).long()] if self.hard \
                 else (1 - e).unsqueeze(-1).squeeze(-1) * md[:, 0] + \
                      e * md[:, 1]
            if self.hard: md = st_onehot(md)
            T = torch.einsum("bj,jmn->bmn", md, self.TB)
            m = torch.bmm(T, m.unsqueeze(-1)).squeeze(-1)
            if self.hard: m = st_onehot(m)
            hc = torch.cat([h[:, t], m, e], dim=-1)
            a = F.softmax(self.alpha(hc), dim=-1)
            q = F.softmax(self.readq(hc), dim=-1)
            beta = F.softmax(self.beta(hc), dim=-1)
            if self.hard:
                a, q, beta = st_onehot(a), st_onehot(q), st_onehot(beta)
            A = torch.einsum("bo,oij->bij", a, Mo)
            uv = torch.einsum("bo,oi->bi", a, u)
            v = beta @ self.vcode
            S = torch.bmm(A, S) + uv.unsqueeze(-1) * v.unsqueeze(1)
            r = torch.einsum("bk,bkd->bd", q, S)
            e = torch.sigmoid(self.eprobe(r))
            if self.hard: e = st_binary(e)
            reads.append(r); mouts.append(m)
        r = torch.stack(reads, 1)
        mm = torch.stack(mouts, 1)
        ho = h + self.out(torch.cat([r, h], -1)) + mm @ self.memb
        return self.head(self.norm(ho)) + r @ self.vcode.t()
