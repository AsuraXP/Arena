"""LM-KR on modal-LM: learnable mode automaton from raw LM loss. python3 phase34.py <seed>"""
import json, resource, sys, time, torch, torch.nn as nn, torch.nn.functional as F
from tasks11 import gen_modal_lm
from models2 import st_onehot
from models5 import role_basis
from models6 import mode_basis
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
FREEZE = len(sys.argv) > 2 and sys.argv[2] == "freeze"
torch.manual_seed(SEED)

class LMKR(nn.Module):
    def __init__(self, vin=6, vout=6, d_model=32, k=12, d_slot=16, M=4):
        super().__init__()
        self.k, self.M, self.hard = k, M, False
        self.register_buffer("PH", torch.stack([role_basis(k)[o % 8] for o in range(16)]))
        self.register_buffer("gbits", (torch.arange(16) < 8).float())
        self.register_buffer("TB", mode_basis(M))
        self.emb = nn.Embedding(vin, d_model)
        self.mdisp = nn.Parameter(torch.randn(vin, M + 2))
        din = d_model + M
        self.alpha = nn.Linear(din, 16)
        nn.init.zeros_(self.alpha.bias); self.alpha.bias.data[8] = 2.0
        self.readq = nn.Linear(din, k)
        self.beta  = nn.Linear(din, 6)
        self.vcode = nn.Parameter(torch.randn(6, d_slot))
        self.wlog  = nn.Parameter(torch.randn(16, k))
        self.S0    = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.memb  = nn.Parameter(0.1 * torch.randn(M, d_model))
        self.flat  = nn.Linear(k * d_slot, d_model)
        self.out   = nn.Linear(d_slot + 2 * d_model, d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vout)

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x)
        md = F.softmax(self.mdisp, dim=-1)[x]
        if self.hard: md = st_onehot(md)
        T = torch.einsum("blj,jmn->blmn", md, self.TB)
        w = F.softmax(self.wlog, dim=-1)
        if self.hard: w = st_onehot(w)
        gb = self.gbits.view(-1, 1, 1)
        Mo = self.PH - gb * torch.einsum("oij,oj,ol->oil", self.PH, w, w)
        u = self.gbits.unsqueeze(-1) * torch.einsum("oij,oj->oi", self.PH, w)
        m = torch.zeros(B, self.M, device=x.device); m[:, 0] = 1.0
        S = self.S0.expand(B, -1, -1)
        reads, flats, ms = [], [], []
        for t in range(L):
            m = torch.bmm(T[:, t], m.unsqueeze(-1)).squeeze(-1)
            if self.hard: m = st_onehot(m)
            hc = torch.cat([h[:, t], m], -1)
            a = F.softmax(self.alpha(hc), -1)
            q = F.softmax(self.readq(hc), -1)
            beta = F.softmax(self.beta(hc), -1)
            if self.hard: a, q, beta = st_onehot(a), st_onehot(q), st_onehot(beta)
            A = torch.einsum("bo,oij->bij", a, Mo)
            uv = torch.einsum("bo,oi->bi", a, u)
            v = beta @ self.vcode
            S = torch.bmm(A, S) + uv.unsqueeze(-1) * v.unsqueeze(1)
            reads.append(torch.einsum("bk,bkd->bd", q, S))
            flats.append(S.reshape(B, -1)); ms.append(m)
        r = torch.stack(reads, 1)
        fl = self.flat(torch.stack(flats, 1))
        mm = torch.stack(ms, 1)
        ho = h + self.out(torch.cat([r, fl, h], -1)) + mm @ self.memb
        return self.head(self.norm(ho)) + r @ self.vcode.t()

model = LMKR()
if FREEZE:
    with torch.no_grad():
        model.mdisp.fill_(-10.0)
        for i, instr in enumerate([2, 3, 0, 0, 0, 0]):   # M0->c1, M1->c2, brackets->id
            model.mdisp[i, instr] = 10.0
    model.mdisp.requires_grad_(False)
    params = [p for n, p in model.named_parameters() if n != "mdisp"]
else:
    params = list(model.parameters())
opt = torch.optim.AdamW(params, lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
for step in range(1, 12001):
    L = 32 if step < 6000 else 64
    x, y, _ = gen_modal_lm(32, L, g)
    loss = F.cross_entropy(model(x).reshape(-1, 6), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 6000 == 0: print(f"[lmkr-s{SEED}] {step} CE {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), f"lmkr_frz_s{SEED}.pt")
model.eval(); res = {}
with torch.no_grad():
    mi = F.softmax(model.mdisp, -1)
    NAMES = ["id","c0","c1","c2","c3","sh"]; TOK = ["M0","M1","(",")","[","]"]
    for i in range(6):
        print(f"[dump-s{SEED}] {TOK[i]}: {NAMES[mi[i].argmax()]}(p={mi[i].max():.2f})",
              flush=True)
    for hard in (False, True):
        model.hard = hard
        for L in (64, 1024, 4096):
            ce = orc = n = 0.0
            for i in range(3):
                x, y, o = gen_modal_lm(4 if L > 256 else 16, L,
                                       torch.Generator().manual_seed(9770 + L + i))
                lp = F.log_softmax(model(x), -1)
                ce += -lp.gather(-1, y.unsqueeze(-1)).sum().item()
                orc += o.sum().item(); n += y.numel()
            res[("hard" if hard else "soft") + str(L)] = round((ce - orc) / n, 5)
out = dict(tag=f"EXP109-LMKR-FROZEN-S{SEED}", dCE=res, wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
