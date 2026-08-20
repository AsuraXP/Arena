"""SSR (Sinkhorn State Router) — novel architecture — and baseline micro-Transformer."""
import math, torch, torch.nn as nn, torch.nn.functional as F

def sinkhorn(logits, n_iter=5):
    # logits: (..., k, k) -> doubly-stochastic via log-space Sinkhorn-Knopp
    for _ in range(n_iter):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()

def hard_perm(P):
    """Greedy assignment rounding: (M,k,k) DS matrices -> exact permutation matrices."""
    M, k, _ = P.shape
    out = torch.zeros_like(P)
    cost = P.detach().clone()
    for m in range(M):
        c = cost[m]
        for _ in range(k):
            idx = torch.argmax(c).item()
            i, j = idx // k, idx % k
            out[m, i, j] = 1.0
            c[i, :] = -1.0; c[:, j] = -1.0
    return out

class SSRLayer(nn.Module):
    """v2: Permutation-Codebook Router. R_t = sum_m alpha_m(x_t) * Sinkhorn(P_m)."""
    def __init__(self, d_model, k=6, d_slot=16, tau=0.5, sink_iters=5, n_proto=8,
                 hard=False, use_write=True):
        super().__init__()
        self.k, self.d_slot, self.tau, self.sink_iters = k, d_slot, tau, sink_iters
        self.n_proto, self.hard, self.use_write = n_proto, hard, use_write
        self.protos = nn.Parameter(torch.randn(n_proto, k, k))
        self.S0 = nn.Parameter(0.5 * torch.randn(k, d_slot))     # learned initial state
        self.alpha = nn.Linear(d_model, n_proto)        # prototype selector
        self.sel   = nn.Linear(d_model, k)              # write-slot selector
        self.val   = nn.Linear(d_model, d_slot)         # write content
        self.gate  = nn.Linear(d_model, 1)              # write strength
        nn.init.constant_(self.gate.bias, -3.0)         # writes nearly off at init
        self.readf = nn.Linear(k * d_slot, d_model)     # full-state readout
        self.out   = nn.Linear(2 * d_model, d_model)

    def forward(self, h):                               # h: (B, L, d_model)
        B, L, _ = h.shape
        P = sinkhorn(self.protos / self.tau, self.sink_iters)     # (M,k,k)
        a = F.softmax(self.alpha(h), dim=-1)                      # (B,L,M)
        if self.hard:   # straight-through vertex snapping (fwd hard, bwd soft)
            P = hard_perm(P) + P - P.detach()
            a_h = torch.zeros_like(a).scatter_(-1, a.argmax(-1, keepdim=True), 1.0)
            a = a_h + a - a.detach()
        with torch.no_grad(): pass
        P_soft = sinkhorn(self.protos / self.tau, self.sink_iters)
        self.reg = (-(a.clamp_min(1e-9) * a.clamp_min(1e-9).log()).sum(-1).mean()
                    + (1.0 - P_soft.max(-1).values.mean()))
        R = torch.einsum("blm,mij->blij", a, P)                   # (B,L,k,k)
        w = F.softmax(self.sel(h), dim=-1)              # (B,L,k)
        v = torch.tanh(self.val(h))                     # (B,L,d_slot)
        g = torch.sigmoid(self.gate(h))                 # (B,L,1)
        S = self.S0.expand(B, -1, -1).contiguous()
        reads = []
        for t in range(L):
            S = torch.bmm(R[:, t], S)
            if self.use_write:
                S = S + g[:, t].unsqueeze(-1) * w[:, t].unsqueeze(-1) * v[:, t].unsqueeze(1)
            reads.append(S.reshape(B, -1))
        r = self.readf(torch.stack(reads, dim=1))       # (B,L,d_model)
        return h + self.out(torch.cat([r, h], dim=-1))

class SSRModel(nn.Module):
    def __init__(self, vocab_in, vocab_out, d_model=32, k=6, d_slot=16,
                 n_layers=1, tau=0.5, sink_iters=5, hard=False, n_proto=8, use_write=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_in, d_model)
        self.layers = nn.ModuleList(
            [SSRLayer(d_model, k, d_slot, tau, sink_iters, n_proto=n_proto, hard=hard, use_write=use_write) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_out)

    def forward(self, x):
        h = self.emb(x)
        for l in self.layers: h = l(h)
        return self.head(self.norm(h))

# ---------------- baseline micro-Transformer ----------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_in, vocab_out, d_model=32, n_heads=2, n_layers=2,
                 max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_in, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                         batch_first=True, norm_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Linear(d_model, vocab_out)

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x) + self.pe[:L]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), 1)
        return self.head(self.tr(h, mask=mask))

def count_params(m): return sum(p.numel() for p in m.parameters())
