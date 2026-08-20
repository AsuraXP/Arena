"""Zero-supervision LM daemon attempt: train -> label-free gate -> report.
python3 daemon.py <seed>"""
import json, resource, sys, time, torch, torch.nn.functional as F
from tasks10 import gen_lm
SEED = int(sys.argv[1])
torch.manual_seed(SEED)
exec(open("phase33.py").read().split("model = LMISA()")[0])
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
torch.save(model.state_dict(), f"lmisa_s{SEED}.pt")
model.eval()
def ce(hard, L, n=3):
    model.hard = hard
    tot = cnt = 0.0
    with torch.no_grad():
        for i in range(n):
            x, y, _ = gen_lm(8 if L > 256 else 16, L,
                             torch.Generator().manual_seed(9740 + L + i))
            lp = F.log_softmax(model(x), -1)
            tot += -lp.gather(-1, y.unsqueeze(-1)).sum().item(); cnt += y.numel()
    return tot / cnt
h64, s64 = ce(True, 64), ce(False, 64)
h1k, s1k = ce(True, 1024), ce(False, 1024)
gate = (h64 - s64 < 0.05) and (h1k <= s1k + 0.02)
res = dict(h64=round(h64, 5), s64=round(s64, 5), h1k=round(h1k, 5), s1k=round(s1k, 5))
verify = {}
if gate:                                    # oracle used ONLY post-gate
    model.hard = True
    with torch.no_grad():
        for L in (64, 1024, 4096):
            tot = orc = cnt = 0.0
            for i in range(3):
                x, y, o = gen_lm(4 if L > 256 else 16, L,
                                 torch.Generator().manual_seed(9730 + L + i))
                lp = F.log_softmax(model(x), -1)
                tot += -lp.gather(-1, y.unsqueeze(-1)).sum().item()
                orc += o.sum().item(); cnt += y.numel()
            verify[f"dCE{L}"] = round((tot - orc) / cnt, 5)
out = dict(tag=f"EXP114-DAEMON-S{SEED}", gate=bool(gate), gate_stats=res,
           oracle_verify=verify, wall_s=round(time.time() - t0, 1),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
print("RESULT " + json.dumps(out), flush=True)
open("results.jsonl", "a").write(json.dumps(out) + "\n")
