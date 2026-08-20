"""M33: KR-ISA on modal+probes. python3 phase27.py <seed>"""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks8 import gen_modalp
from models6 import KRISA
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED)
model = KRISA(7, 5)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
t0 = time.time()
for step in range(1, 12001):
    f = step / 12000
    md = 1 if f < .15 else 2 if f < .3 else 3 if f < .5 else 4 if f < .7 else 5
    L = 32 if f < 0.5 else 64
    x, y, _, _ = gen_modalp(32, L, g, max_depth=md)
    loss = F.cross_entropy(model(x).reshape(-1, 5), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 4000 == 0: print(f"[m33] {step} loss {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), f"krisa_modalp_s{SEED}.pt")
with torch.no_grad():
    mi = F.softmax(model.mdisp, -1)
    NAMES = ["id","c0","c1","c2","c3","sh"]; TOK = ["M0","M1","(",")","[","]","MP"]
    for i in range(7):
        print(f"[dump] {TOK[i]}: {NAMES[mi[i].argmax()]}(p={mi[i].max():.2f})", flush=True)
model.eval(); model.hard = True; res = {}
with torch.no_grad():
    for L in (64, 1024):
        c = t = 0
        for i in range(3):
            x, y, _, _ = gen_modalp(8, L, torch.Generator().manual_seed(9820+L+i))
            p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
        res["hard" + str(L)] = round(c / t, 4)
out = dict(tag=f"EXP085-M33-RAW-S{SEED}", acc=res, wall_s=round(time.time()-t0,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
