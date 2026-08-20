"""agree training. python3 phase16.py <model> <seed>"""
import json, sys, time, torch, torch.nn.functional as F, resource
from tasks5 import gen_agree
from models3 import OpPRAM
from models import TinyTransformer, count_params
MODEL = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
torch.manual_seed(SEED)
if MODEL == "oppram":
    model = OpPRAM(5, 3, k=8, n_ops=16)
else:
    model = TinyTransformer(5, 3, 32, 2, 2, max_len=4104)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
for step in range(1, 12001):
    f = step / 12000
    md = 1 if f < .15 else 2 if f < .3 else 3 if f < .5 else 4 if f < .7 else 5
    L = 32 if f < 0.5 else 64
    x, y, _, _ = gen_agree(32, L, g, max_depth=md)
    loss = F.cross_entropy(model(x).reshape(-1, 3), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 4000 == 0: print(f"[agree-{MODEL}] {step} loss {loss.item():.5f}", flush=True)
if MODEL == "oppram":
    torch.save(model.state_dict(), f"oppram_agree_s{SEED}.pt")
    for l in model.layers: l.hard = True
model.eval(); res = {}
with torch.no_grad():
    for L in (64, 256, 1024, 4096):
        bs = 16 if L <= 256 else (4 if MODEL == "oppram" else 2)
        c = t = cv = tv = 0
        for i in range(3):
            x, y, _, _ = gen_agree(bs, L, torch.Generator().manual_seed(8400 + L + i))
            p = model(x).argmax(-1)
            c += (p == y).sum().item(); t += y.numel()
            m = y != 2                                 # verb positions only
            cv += (p[m] == y[m]).sum().item(); tv += m.sum().item()
        res[str(L)] = round(c / t, 4); res["verb" + str(L)] = round(cv / tv, 4)
out = dict(tag=f"EXP067-AGREE-{MODEL.upper()}-S{SEED}", params=count_params(model),
           acc=res, wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
