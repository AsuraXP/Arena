"""KR-ISA: neural Krohn-Rhodes cascade.
Stage 1: mode automaton over M states; per-token transition selected (crisp softmax)
from enumerated permutation-reset basis {id, const_0..const_{M-1}, shift+1}.
Stage 2: ISA register machine (16 fixed role-instructions) with CONTEXTUAL tables:
dispatch/read/value logits indexed by (token, mode). All selections crisp-channel."""
import torch, torch.nn as nn, torch.nn.functional as F
from models2 import st_onehot
from models5 import role_basis

def mode_basis(M):
    mats = [torch.eye(M)]
    for j in range(M):
        C = torch.zeros(M, M); C[j, :] = 1.0
        mats.append(C)                                   # const_j (reset)
    S = torch.zeros(M, M)
    for i in range(M): S[(i + 1) % M, i] = 1.0
    mats.append(S)                                       # shift+1
    return torch.stack(mats)                             # (M+2, M, M)

class KRISA(nn.Module):
    def __init__(self, vocab_in, vocab_out, d_model=32, k=12, d_slot=16, M=4,
                 hard=False):
        super().__init__()
        self.vin, self.vout, self.k, self.M, self.hard = vocab_in, vocab_out, k, M, hard
        self.register_buffer("roles", role_basis(k))
        self.NOPS = 16
        self.register_buffer("gbits", (torch.arange(16) < 8).float())
        self.register_buffer("PH", torch.stack([role_basis(k)[o % 8] for o in range(16)]))
        self.register_buffer("TB", mode_basis(M))        # (M+2,M,M)
        self.emb = nn.Embedding(vocab_in, d_model)
        self.mdisp = nn.Parameter(torch.randn(vocab_in, M + 2))    # mode instr table
        self.ta = nn.Parameter(torch.randn(vocab_in, M, 16))       # contextual dispatch
        self.tq = nn.Parameter(torch.randn(vocab_in, M, k))        # contextual read
        self.tb = nn.Parameter(torch.randn(vocab_in, M, vocab_out))# contextual value
        self.wlog = nn.Parameter(torch.randn(16, k))
        self.S0 = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.vcode = nn.Parameter(torch.randn(vocab_out, d_slot))
        self.out = nn.Linear(d_slot + d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_out)

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x)
        md = F.softmax(self.mdisp, dim=-1)[x]            # (B,L,M+2)
        ta = F.softmax(self.ta, dim=-1)[x]               # (B,L,M,16)
        tq = F.softmax(self.tq, dim=-1)[x]
        tb = F.softmax(self.tb, dim=-1)[x]
        w = F.softmax(self.wlog, dim=-1)
        if self.hard:
            md, ta, tq, tb, w = (st_onehot(md), st_onehot(ta), st_onehot(tq),
                                 st_onehot(tb), st_onehot(w))
        T = torch.einsum("blj,jmn->blmn", md, self.TB)   # (B,L,M,M)
        m = torch.zeros(B, self.M, device=x.device); m[:, 0] = 1.0
        modes = []
        for t in range(L):
            m = torch.bmm(T[:, t], m.unsqueeze(-1)).squeeze(-1)
            modes.append(m)
        m_t = torch.stack(modes, 1)                      # (B,L,M)
        a = torch.einsum("blm,blmo->blo", m_t, ta)       # contextual dispatch
        q = torch.einsum("blm,blmk->blk", m_t, tq)
        beta = torch.einsum("blm,blmv->blv", m_t, tb)
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
