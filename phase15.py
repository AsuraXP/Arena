"""abcp count-probe training. python3 phase15.py <seed>"""
import json, sys, time, torch, torch.nn.functional as F
from tasks4 import gen_abcp
from models3 import OpPRAM
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED)
model = OpPRAM(4, 10, k=12, n_ops=16)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
for step in range(1, 12001):
    f = step / 12000
    nm = 2 if f < .2 else 3 if f < .4 else 4 if f < .6 else 5
    L = 32 if f < 0.5 else 64
    x, y, _, _ = gen_abcp(32, L, g, nmax=nm)
    loss = F.cross_entropy(model(x).reshape(-1, 10), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 4000 == 0: print(f"[abcp] {step} loss {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), f"oppram_abcp_s{SEED}.pt")
for l in model.layers: l.hard = True
model.eval()
with torch.no_grad():
    c = t = 0
    for i in range(3):
        x, y, _, _ = gen_abcp(16, 64, torch.Generator().manual_seed(8300 + i))
        p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
print(json.dumps(dict(tag=f"EXP066-ABCP-SOFT-S{SEED}", hard64=round(c/t,4),
      wall_s=round(time.time()-t0,1))), flush=True)
open("results.jsonl","a").write(json.dumps(dict(tag=f"EXP066-ABCP-SOFT-S{SEED}",
      hard64=round(c/t,4)))+"\n")
