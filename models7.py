"""KR-ISA v2: linear contextual dispatch on [emb(token); mode-onehot].
Mode automaton: enumerated permutation-reset basis (as models6). Stage-2: 16-instruction
ISA with linear heads (embedding-coupled, per L-EMBED-COUPLING)."""
import torch, torch.nn as nn, torch.nn.functional as F
from models2 import st_onehot
from models5 import role_basis
from models6 import mode_basis

class KRISA2(nn.Module):
    def __init__(self, vocab_in, vocab_out, d_model=32, k=12, d_slot=16, M=4,
                 hard=False):
        super().__init__()
        self.vin, self.vout, self.k, self.M, self.hard = vocab_in, vocab_out, k, M, hard
        self.register_buffer("PH", torch.stack([role_basis(k)[o % 8] for o in range(16)]))
        self.register_buffer("gbits", (torch.arange(16) < 8).float())
        self.register_buffer("TB", mode_basis(M))
        self.emb = nn.Embedding(vocab_in, d_model)
        self.mdisp = nn.Parameter(torch.randn(vocab_in, M + 2))
        self.alpha = nn.Linear(d_model + M, 16)
        nn.init.zeros_(self.alpha.bias); self.alpha.bias.data[8] = 2.0   # no-op prior
        self.readq = nn.Linear(d_model + M, k)
        self.beta  = nn.Linear(d_model + M, vocab_out)
        self.wlog  = nn.Parameter(torch.randn(16, k))
        self.S0    = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.vcode = nn.Parameter(torch.randn(vocab_out, d_slot))
        self.out   = nn.Linear(d_slot + d_model, d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_out)

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x)
        md = F.softmax(self.mdisp, dim=-1)[x]
        if self.hard: md = st_onehot(md)
        T = torch.einsum("blj,jmn->blmn", md, self.TB)
        m = torch.zeros(B, self.M, device=x.device); m[:, 0] = 1.0
        modes = []
        for t in range(L):
            m = torch.bmm(T[:, t], m.unsqueeze(-1)).squeeze(-1)
            modes.append(m)
        m_t = torch.stack(modes, 1)
        if self.hard: m_t = st_onehot(m_t)
        hc = torch.cat([h, m_t], dim=-1)
        a = F.softmax(self.alpha(hc), dim=-1)
        q = F.softmax(self.readq(hc), dim=-1)
        beta = F.softmax(self.beta(hc), dim=-1)
        w = F.softmax(self.wlog, dim=-1)
        if self.hard:
            a, q, beta, w = st_onehot(a), st_onehot(q), st_onehot(beta), st_onehot(w)
        gb = self.gbits.view(-1, 1, 1)
        Mo = self.PH - gb * torch.einsum("oij,oj,ol->oil", self.PH, w, w)
        u = self.gbits.unsqueeze(-1) * torch.einsum("oij,oj->oi", self.PH, w)
        A = torch.einsum("blo,oij->blij", a, Mo)
        uv = torch.einsum("blo,oi->bli", a, u)
        v = beta @ self.vcode
        b = uv.unsqueeze(-1) * v.unsqueeze(-2)
        S = self.S0.expand(B, -1, -1); reads = []
        for t in range(L):
            S = torch.matmul(A[:, t], S) + b[:, t]
            reads.append(torch.einsum("bk,bkd->bd", q[:, t], S))
        r = torch.stack(reads, 1)
        ho = h + self.out(torch.cat([r, h], -1))
        return self.head(self.norm(ho)) + r @ self.vcode.t()
