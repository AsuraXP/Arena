"""Modal-dyck: 1-layer vs 2-layer ISA. python3 phase25.py <n_layers>"""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks8 import gen_modal
from models5 import RoleOpPRAM
from models import count_params
NL = int(sys.argv[1])
torch.manual_seed(0)
model = RoleOpPRAM(6, 3, fixed_isa=True, n_layers=NL)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(100)
t0 = time.time()
for step in range(1, 12001):
    f = step / 12000
    md = 1 if f < .15 else 2 if f < .3 else 3 if f < .5 else 4 if f < .7 else 5
    L = 32 if f < 0.5 else 64
    x, y, _, _ = gen_modal(32, L, g, max_depth=md)
    loss = F.cross_entropy(model(x).reshape(-1, 3), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 6000 == 0: print(f"[modal-{NL}L] {step} loss {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), f"isa_modal_{NL}L.pt")
soft_loss = loss.item()
res = {}
model.eval()
with torch.no_grad():
    for hard in (False, True):
        for l in model.layers: l.hard = hard
        for L in (64, 1024):
            bs = 16 if L <= 256 else 4
            c = t = 0
            for i in range(3):
                x, y, _, _ = gen_modal(bs, L, torch.Generator().manual_seed(9700+L+i))
                p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
            res[("hard" if hard else "soft") + str(L)] = round(c / t, 4)
out = dict(tag=f"EXP081-MODAL-{NL}L", params=count_params(model),
           final_loss=round(soft_loss, 5), acc=res,
           wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
