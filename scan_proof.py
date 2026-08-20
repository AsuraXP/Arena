"""Proof: PRAM recurrence is computable by associative scan.
1) exact equivalence scan vs sequential loop (fwd + grad)
2) wall-clock scaling: python loop O(L) depth vs Hillis-Steele O(log L) depth
"""
import time, torch
from models2 import PRAM

torch.manual_seed(0)
m_loop = PRAM(50, 8, use_scan=False)
m_scan = PRAM(50, 8, use_scan=True)
m_scan.load_state_dict(m_loop.state_dict())

x = torch.randint(0, 50, (4, 128))
y1, y2 = m_loop(x), m_scan(x)
print(f"[EQUIV fwd] max|loop-scan| = {(y1 - y2).abs().max().item():.3e}")
g1 = torch.autograd.grad(y1.pow(2).sum(), m_loop.parameters(), retain_graph=False)
g2 = torch.autograd.grad(y2.pow(2).sum(), m_scan.parameters(), retain_graph=False)
gd = max((a - b).abs().max().item() for a, b in zip(g1, g2))
print(f"[EQUIV grad] max param-grad diff = {gd:.3e}")

# hard mode equivalence too
for mm in (m_loop, m_scan):
    for l in mm.layers: l.hard = True
y1, y2 = m_loop(x), m_scan(x)
print(f"[EQUIV fwd HARD] max|loop-scan| = {(y1 - y2).abs().max().item():.3e}")
for mm in (m_loop, m_scan):
    for l in mm.layers: l.hard = False

print(f"{'L':>6} {'loop_ms':>9} {'scan_ms':>9} {'speedup':>8}")
for L in [64, 256, 1024, 4096]:
    x = torch.randint(0, 50, (1, L))
    for mm, name in ((m_loop, 'loop'), (m_scan, 'scan')):
        with torch.no_grad(): mm(x)  # warm
    with torch.no_grad():
        t0 = time.time(); m_loop(x); tl = (time.time() - t0) * 1e3
        t0 = time.time(); m_scan(x); ts = (time.time() - t0) * 1e3
    print(f"{L:>6} {tl:>9.1f} {ts:>9.1f} {tl/ts:>7.1f}x")
