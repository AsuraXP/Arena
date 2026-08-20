"""TRACK B: one OpPRAM, two grammars. python3 phase18.py <seed>"""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks6 import gen_mixed, gen_mixed_family
from models3 import OpPRAM
from models import count_params
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED)
model = OpPRAM(12, 6, k=12, n_ops=32)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
for step in range(1, 14001):
    f = step / 14000
    L = 32 if f < 0.5 else 64
    x, y, _, _ = gen_mixed(32, L, g)
    loss = F.cross_entropy(model(x).reshape(-1, 6), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 4000 == 0: print(f"[mixed] {step} loss {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), f"oppram_mixed_slack_s{SEED}.pt")
for l in model.layers: l.hard = True
model.eval(); res = {}
with torch.no_grad():
    for fam in ("dyck", "agree"):
        gen = gen_mixed_family(fam)
        for L in (64, 1024, 4096):
            bs = 16 if L <= 256 else 4
            c = t = 0
            for i in range(3):
                x, y, _, _ = gen(bs, L, torch.Generator().manual_seed(8800 + L + i))
                p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
            res[fam + str(L)] = round(c / t, 4)
out = dict(tag=f"EXP074-MIXED-SLACK-S{SEED}", params=count_params(model), acc=res,
           wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
