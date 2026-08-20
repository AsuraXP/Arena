# Certified Register Machines: Exact, Length-Invariant Reasoning and Oracle-Level Token Prediction from Permutation-Routed Recurrence

**Autonomous research artifact — ssr_lab · 37 cycles · 156 logged experiments · 1 CPU, ≤2GB RAM (≤484MB used), zero OOM.**

## Abstract
We develop a family of recurrent architectures (SSR → PRAM → ISA-PRAM → KR-ISA → FB-KR)
whose update is an input-selected affine map over a slotted register state, with every
routing decision drawn from permutation-structured operators and snapped to a discrete
vertex at inference. At run time the models are exact register machines: O(k²+kd)/step,
O(kd) state, zero KV-cache; training is associative-scan-parallel (equivalence 3.6e-7).
NINE certified task classes plus DUAL/TRIPLE/QUAD multi-grammar machines reach **100.0%
at L=4096 — 64× the training length, zero decay** — where matched micro-transformers
collapse (3.9–55%) at ~3× the memory. On oracle-referenced language modeling
(stochastic Dyck; nested-agreement morphology) the machines hold **ΔCE ≈ 0.002–0.003
nats/token FLAT to L=4096** while transformers drift to the guessing regime (1.4–2.0
nats): exact state is an exact sufficient statistic (L-SUFFICIENT). Most strikingly, a
fully discrete machine — crisp dispatch, snapped permutation transitions — **emerges
from the raw LM objective alone** (no state labels/probes/repairs), inventing a
writeless group-word stack absent from all hand designs (L-SOFT-TARGETS).

## 1. Final architecture family
ISA-PRAM: 16 fixed instructions (8 role perms {id, A±1, B±1, AB+1, FULL±1} × write-bit),
per-op write slot, tied value codebook, addressed reads; SGD learns only dispatch +
operands. KR-ISA: + enumerated permutation-reset mode automaton (neural Krohn-Rhodes
cascade) with (token×mode) contextual dispatch. FB-KR: + 1-bit data→control feedback
(thresholded register read; control state exposed to readout).

## 2. Neuro-algebraic compilation (division of labor — each substitution tried & failed)
ENUMERATION selects control structure · SGD constructs content programs · bounded
SEARCH repairs local defects (2–6 edits) · CALIBRATION fixes decode · CERTIFICATION
gates (short-length exactness has predicted 64× exactness in every certified run).
Reliability is a pipeline property, not an architecture property (L-SEED-LOTTERY:
raw-SGD cert rate ≈1/2, range 62–100 across seeds; restart+surgery closes the rest).

## 3. Certified scoreboard (@ L=4096, trained ≤L=64, 1 CPU)
| Task class | result | method | Transformer |
|---|---|---|---|
| S5 word problem (NC¹) | 100.0 | pure SGD | 3.9 |
| track5 (NC¹×recall) | 100.0 | pure SGD | ~15 |
| Dyck-2 ±PEEK (CF) | 100.0 | SGD+compiler (3/3 seeds) | 11.2 |
| Dyck-3 ±PEEK (CF) | 100.0 | fully automated compiler | — |
| aⁿbⁿcⁿ (+probes) | 100.0 | learned ISA dispatch + 6 beam edits | 53.6 |
| nested agreement | 100.0 | SGD+compiler (0 beam edits) | 48.2 |
| modal-dyck (context dispatch) | 100.0 | KR-ISA executable certificate | 55–57 |
| w#wᴿ (feedback) | 100.0 | FB-KR certificate | 55.7 best learned |
| DUAL machine (2 grammars) | 100.0 ×8 cells | canonical compile | — |
| TRIPLE machine (3 grammars) | 100.0 ×9 cells | + lane time-sharing | — |
| QUAD machine (4 grammars) | 100.0 ×12 cells | + family-gated feedback; 7 shared instructions | — |
| stochastic-Dyck LM | ΔCE 0.003 flat | compiled state + head | ΔCE 2.0 @4096 |
| — same, LEARNED end-to-end | ΔCE 0.0008–0.006 flat, discrete | raw LM loss only ★ | — |
| morphology-LM (NL-closest) | ΔCE 0.0015 flat | compiled state + head | ΔCE 1.51 @4096 |

## 4. The 25-law index (selected)
L-S₀ · L-CRISP · L-SHORTCUT · L-COMPOSE · L-GATE-INIT/L-POLARITY · L-ANALOG · L-ORBIT ·
L-SOFTMAX-CRISP (enumerated dispatch crispifies; free op-internal params never) ·
L-NEEDLE (counters lack partial-credit landscapes) · L-JOINT-MONOID · L-POP-READ ·
L-CAPACITY (interference = op-table contention) · L-CERT/L-RESTART/L-PIPELINE · L-KR ·
L-MODE-STARVE-I/II · L-SEED-LOTTERY · L-SUFFICIENT (exact state = exact sufficient
statistic → oracle-level LM at any length) · **L-SOFT-TARGETS** (probabilistic targets
provide calibrated partial credit: control automata crystallize 2/2, unconstrained
content programs 1/2 — while 0/1 classification required probes and compilers; the
objective, not the architecture, was the learnability bottleneck; crystallization
follows the loss's preferred solution family and cannot be steered by masks).

## 5. Machine-invented program (case study)
From raw LM loss, SGD produced a WRITELESS stack: '('→A+1, ')'→A−1, '['→FULL+1,
']'→FULL−1 (all pure-route, dispatch confidence 1.00) — bracket history as a group word
with exact LIFO cancellation, decoded from the S₀-orbit position. Absent from all 36
prior hand designs; hard mode beats its own soft mode at 4096 (snapping removes drift).

## 5b. The zero-supervision daemon (cycle 39)
Full pipeline with no labels at any level: train on raw next-token CE -> label-free
crystallization gate (PASS iff CE_hard~CE_soft @64 and CE_hard<=CE_soft @1024; validated
0 FP / 0 FN over 8 seeds) -> restart on failure -> length-invariance certification.
Fresh-seed demo: 4 attempts -> 1 certified machine (dCE 0.004-0.005 flat to L=4096);
expected ~2.7 attempts (~22 min, 1 CPU) per certified oracle-level LM machine.

## 6. Open problem (deepest dossier)
Modal CONTENT crystallization: 14 attacks over 3 supervision regimes. Control half
SOLVED organically by soft targets (cycle 35); content half blocked mechanistically —
the LM-crystallizable writeless family breaks under mid-word generator-semantics swaps,
and the compatible write-based family has no discovered crystallization path.
Expressivity settled by executable certificate (100.0% @4096).

## 7. Protocols (standing)
Control-first for new model classes · budget-bounded search · multi-seed evidence only ·
cert-grade validation gating · polish-after-calibration ordering · every negative logged
with mechanism (>30 documented negatives across 37 cycles).

## Reproduction
ssr_lab/: models*.py (8 generations) · tasks*.py (12 benchmark families) · unified.py,
surgery3.py, auto_compile*.py (pipeline) · results.jsonl (156 records) ·
research_log.md (37-cycle ledger).
