"""LM-ISA: learn exact-state machine from raw LM objective. python3 phase33.py <seed>"""
import json, resource, sys, time, torch, torch.nn as nn, torch.nn.functional as F
from tasks10 import gen_lm
from models2 import st_onehot
from models5 import role_basis

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED)

class LMISA(nn.Module):
    def __init__(self, vin=4, vout=4, d_model=32, k=12, d_slot=16):
        super().__init__()
        self.k, self.hard = k, False
        self.register_buffer("PH", torch.stack([role_basis(k)[o % 8] for o in range(16)]))
        self.register_buffer("gbits", (torch.arange(16) < 8).float())
        self.emb = nn.Embedding(vin, d_model)
        self.alpha = nn.Linear(d_model, 16)
        nn.init.zeros_(self.alpha.bias); self.alpha.bias.data[8] = 2.0
        self.readq = nn.Linear(d_model, k)
        self.beta  = nn.Linear(d_model, vout)
        self.vcode = nn.Parameter(torch.randn(vout, d_slot))
        self.wlog  = nn.Parameter(torch.randn(16, k))
        self.S0    = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.flat  = nn.Linear(k * d_slot, d_model)     # continuous LM decode channel
        self.out   = nn.Linear(d_slot + 2 * d_model, d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vout)

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x)
        a = F.softmax(self.alpha(h), -1)
        q = F.softmax(self.readq(h), -1)
        beta = F.softmax(self.beta(h), -1)
        w = F.softmax(self.wlog, -1)
        if self.hard:                       # transitions snap; decode stays continuous
            a, q, beta, w = st_onehot(a), st_onehot(q), st_onehot(beta), st_onehot(w)
        gb = self.gbits.view(-1, 1, 1)
        Mo = self.PH - gb * torch.einsum("oij,oj,ol->oil", self.PH, w, w)
        u = self.gbits.unsqueeze(-1) * torch.einsum("oij,oj->oi", self.PH, w)
        A = torch.einsum("blo,oij->blij", a, Mo)
        uv = torch.einsum("blo,oi->bli", a, u)
        v = beta @ self.vcode
        b = uv.unsqueeze(-1) * v.unsqueeze(-2)
        S = self.S0.expand(B, -1, -1); reads, flats = [], []
        for t in range(L):
            S = torch.bmm(A[:, t], S) + b[:, t]
            reads.append(torch.einsum("bk,bkd->bd", q[:, t], S))
            flats.append(S.reshape(B, -1))
        r = torch.stack(reads, 1)
        fl = self.flat(torch.stack(flats, 1))
        ho = h + self.out(torch.cat([r, fl, h], -1))
        return self.head(self.norm(ho)) + r @ self.vcode.t()

model = LMISA()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
for step in range(1, 12001):
    L = 32 if step < 6000 else 64
    x, y, _ = gen_lm(32, L, g)
    loss = F.cross_entropy(model(x).reshape(-1, 4), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 6000 == 0: print(f"[lmisa-s{SEED}] {step} CE {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), f"lmisa_s{SEED}.pt")
model.eval(); res = {}
with torch.no_grad():
    for hard in (False, True):
        model.hard = hard
        for L in (64, 1024, 4096):
            ce = orc = n = 0.0
            for i in range(3):
                x, y, o = gen_lm(4 if L > 256 else 16, L,
                                 torch.Generator().manual_seed(9780 + L + i))
                lp = F.log_softmax(model(x), -1)
                ce += -lp.gather(-1, y.unsqueeze(-1)).sum().item()
                orc += o.sum().item(); n += y.numel()
            res[("hard" if hard else "soft") + str(L)] = round((ce - orc) / n, 5)
out = dict(tag=f"EXP107-LMISA-S{SEED}", dCE=res, wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
