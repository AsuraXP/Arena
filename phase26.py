"""KR-ISA on modal-dyck. python3 phase26.py <seed>"""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks8 import gen_modal
from models6 import KRISA
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED)
model = KRISA(6, 3)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(SEED + 100)
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
    if step % 4000 == 0: print(f"[kr] {step} loss {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), f"krisa_modal_s{SEED}.pt")
model.eval(); res = {}
with torch.no_grad():
    # mode-instruction dump
    mi = F.softmax(model.mdisp, -1)
    NAMES = ["id", "c0", "c1", "c2", "c3", "sh"]
    TOK = ["M0", "M1", "(", ")", "[", "]"]
    for i in range(6):
        print(f"[dump] {TOK[i]}: mode-instr={NAMES[mi[i].argmax()]}(p={mi[i].max():.2f})",
              flush=True)
    for hard in (False, True):
        model.hard = hard
        for L in (64, 1024, 4096):
            bs = 16 if L <= 256 else 4
            c = t = 0
            for i in range(3):
                x, y, _, _ = gen_modal(bs, L, torch.Generator().manual_seed(9800+L+i))
                p = model(x).argmax(-1); c += (p == y).sum().item(); t += y.numel()
            res[("hard" if hard else "soft") + str(L)] = round(c / t, 4)
out = dict(tag=f"EXP082-KRISA-MODAL-S{SEED}",
           params=sum(p.numel() for p in model.parameters()),
           acc=res, certified=all(res[k] == 1.0 for k in res if k.startswith("hard")),
           wall_s=round(time.time()-t0,1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl","a").write(json.dumps(out) + "\n")
