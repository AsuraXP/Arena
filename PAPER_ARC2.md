# The Resource Inversion Program: Certified Register Organs Behind a Learned Router — Exact Unbounded Arithmetic Inside a Token-Prediction Host, from a 2GB Sandbox

**ARC-2 autonomous research artifact — 7 cycles · 12+ logged experiments/milestones · 1-2 CPU, ≤2GB RAM charter (peak observed 937MB) · zero training restarts across the entire program.**
Predecessor: ssr_lab (41 cycles, `paper.md`, `research_log.md`) — architecture family and 26-law index inherited.

## Abstract

We close the resource-inversion loop: machines trained in a 2GB-RAM/1-CPU sandbox
defeat frontier LLMs (gemini-3.1-pro 8/19, gemini-3.5-flash 7/18) on a frozen public
suite of 29 exact algorithmic tasks — **29/29 machine-side** — with victory
independently auditable by any operator pasting identical judge cards into any
frontier chatbot. We then mount the certified "organ" set (streaming addition,
streaming division, iterated multiplication, and **learned big/big division**) inside
a single token-prediction host (tiny RoPE transformer, ~72k params) behind a
**certified table-machine router** (9 states, 78605/78695→78605/78605 routing
decisions exact, 3 seeds). The hybrid completes exact arithmetic spans — up to
150-digit addition/division and 40-digit ÷ 35-digit long division — **mid-generation,
inside fluent text, at up to 19× the host's training operand width**, where the plain
host scores 0% and a neural-probe router collapses beyond 2–5×. Transcript elision
(replacing organ-owned answer regions with one marker in the host's context view) is
certified lossless-by-construction and saves ~48% context. Every trained component is
multi-seed verified; every defect across the program was caught by oracle/audit gates
before contaminating results.

## 1. The suite and the inversion (operator-auditable victory)

Seven task families, frozen at cycle 1 and extended as solved: T1 addition (50–100
digit), T2 parity (1000–1500 bit), T4 cup-chains (300 swaps), T3 multiplication
(35×35–40×40), T5 nested expressions (3–5 ops, 20–40-digit operands), T6 sorting
(200–250 elements), T7 division (100–120 digit ÷ 2..12).

| side | score | params | training cost |
|---|---|---|---|
| sandbox machines | **29/29 exact** | ~1.7M total across 7 machines | < 5 min wall, 1 CPU |
| gemini-3.1-pro (operator-judged) | 8/19 attempted, 11 actively wrong | ~10¹² | 100s-of-GPU runs |
| gemini-3.5-flash-thinking (operator-judged) | 7/18 attempted, 11 actively wrong | ~10¹² | 100s-of-GPU runs |

Notables from the judging (FRONTIER_RESULTS.md): pro solved all six additions up to
100 digits but failed 1500-bit parity (3/4 wrong), guessed on 300-swap cup chains,
and never truly attempted 35×35+ products; neither model answered the nested
expressions, 200-element sorts, or 120-digit divisions. Where two frontier models
disagree, at least one is wrong; against the python-verified key, both are wrong on
11 items each.

## 2. Machines (cycles 2–6)

All machines are table transducers over enumerated instruction/state bases — SGD
learns dispatch and output heads by **direct per-cell supervision**; inference is the
argmax (discrete) machine; certification is length-invariance testing at 8–30× the
training scale before any frozen item is touched.

| task | mechanism | trained | certified |
|---|---|---|---|
| parity / cups / addition (C2) | KR-automaton, generic basis (id/const/shift/transpositions) | ≤64 items / ≤8 digits | parity@2000, cups@350, add@120 (14/14) |
| multiplication (C3) | **IFT**: one learned FST iterated on its own tape to fixpoint | ≤6-digit operands | 200/200 @ 25–50 digits (O(N²) work in O(N) passes) |
| nested expressions (C4) | frozen organs under a LEARNED dispatch controller over a generic value stack | single-op, ≤3 digits | 200/200 @ 3–5 ops, 20–40-digit operands |
| sorting (C5) | learned bubble transducer; SELECT value routing; tape fixpoint halt | len ≤ 8 | 200/200 @ 100–250 elements, natural encoding |
| division (C6) | streaming transducer, registers (divisor set-once, remainder), MSB natural | ≤7 digits, d∈2..12 | 200/200 @ 80–150 digits |
| **big/big division (M5a)** | learned per-pass transducers (SUB borrow / ADD carry / SHIFT lag-write) under oracle driver | ≤8-digit rows | 300/300 across 8/4 … 40/35, 40/1, 1/1 |

Bug forensics as instrument: C6 diagnosed a trace bug from its failure RATE alone
(15/200 = the ÷2 prior); M5b required token-exact (vs int-exact) certification to
expose legitimate leading zeros (100÷7 → q-digits "014").

## 3. The hybrid (cycle 7 — problem P11 closed)

**Architecture.** Host: 2-layer RoPE transformer (d=64, ~72k params) over one token
stream mixing Markov text and in-band arithmetic spans (`/`, `+`, `×`, `//` syntax).
Router: table machine (states: text / {div,mul,add,dbig}-question / -answer; classes
text + 4 organs), trained by direct supervision — crystallizes deterministically
(L-DIRECT-GRADIENT), certified by held-out decision-sequence exactness incl. 150-digit
streams. Organs: the frozen certified machines of §2, self-terminating (they emit
their own end-of-span token; the router never counts).

**Results (greedy span completion, operands ≤8 digits at training):**

| span type | width tested | plain host | neural-probe router | KR-routed hybrid |
|---|---|---|---|---|
| addition | 8 → 150 (19×) | 0–10% | 100% in-dist → 0% | **100%** |
| division (d≤12) | 8 → 150 (19×) | 0–2.5% | 100% → 0% | **100%** |
| multiplication (iterated) | 8 → 25 | 0% | 100% → 0% | **100%** |
| **big/big division (learned)** | 8×4 → 40×35 | 0% | ≤70% → 0% | **100%** |

Text fluency is preserved (text CE at parity or better vs the plain host at every
width). The 40×35 item routes ~75 rounds × 3 pass types × 41-cell tapes mid-stream —
the heaviest exact computation dispatched inside a token stream in this program.

**Transcript elision (M4a).** Replacing organ-owned answer regions with one `[ANS]`
marker in the host's context view: never costs fluency (3.17 vs 3.45 plain / 3.36
masked at 19× width), saves ~48% context, exactness untouched — lossless by
construction because correctness is organ-owned (contrast: heuristic KV-compaction).

**Controls.** (A) L-HEAD-DECOUPLE: an auxiliary gate head trained jointly through the
host trunk destroys length-extrapolated text quality (CE 4.08 vs 2.65 at 19×; all
arms equal in-distribution); a gradient-detached head is bit-identical to no head.
Fix adopted. (B) Host seed sweep 3/3 exact on the heaviest spans.

## 4. Law index additions (ARC-2)

- **L-ENCODING** (C2): task hardness class is representation-relative; choosing the
  encoding IS architecture (carry is permutation-reset-class LSB-first). Re-applied
  at mount time twice (remainder ≥10 token collision; big-div leading zeros).
- **L-DETERMINISM (a)(b)** (C3): heads index OLD state where roles differ;
  don't-care outputs masked from loss. Diagnosed from failure rates; applied at
  design time thereafter.
- **L-COMPOSE-EXACT** (C4): exactness composes across certified components.
- **L-DIRECT-GRADIENT** (C4): direct per-cell supervision makes crystallization
  deterministic — the ssr_lab lottery was a property of indirect supervision.
  Program-wide evidence: zero restarts.
- **L-STRUCTURAL-ROUTING** (C5): routing values by selection removes the value
  dimension from learning entirely.
- **L-GATE-EXTENT** (M1/M2): span routing must be self-terminating; per-position
  neural gates cannot hold answer extent (nor boundary decisions) under length
  extrapolation — 75.9% accuracy at 19×, 0% exact spans — while boundary-local
  routing with organ-owned termination is exact by construction.
- **L-HEAD-DECOUPLE** (control A): auxiliary heads sharing an LM trunk's gradients
  degrade length-extrapolated text quality; detached auxiliary heads are free.

**Practice laws** (process, not architecture): the index-collision audit (cycle-3
protocol) applied prospectively caught the SUB/ADD table collision before training;
token-exact certification is strictly stronger than int-exact; oracle-first gates
caught every defect pre-results (10+ lifetime catches, zero contaminated results).

## 5. Honest ledger

Negatives logged in full: T3 required two failed attempts (0/5, 4/5) before
certification; T7 one (0/3); the same-budget micro-transformer baseline of C5 was
undertrained (not the fair bar — external bar cited instead); M3's host text-CE
drift was real until control A attributed and fixed it; the neural-probe router is
reported collapsing, not hidden; big/binary division by iterated transducers was
attempted only after its encoding was oracle-proven (511/511) — the learnable-form
design needed the aprev-lag decomposition (documented in M4b/M5a notes).
Out of scope (charter honesty clause): world knowledge, alignment, multimodality.

## 6. Cost accounting

Suite: ~1.7M params across seven machines, <5 min total training, peak RAM ~310MB.
Hybrid stack: host ~72k + routers/tables ~50k params; longest single experiment 566s;
peak RAM 937MB (charter ceiling 2048MB). Frontier comparison point: the judge items
that frontier models failed were answered exactly by machines whose *combined*
training cost is under ten CPU-minutes.

## 7. Status and succession

ARC-2 closes at cycle 7 with the charter's victory condition met on both sides
(machine 29/29; frontier measured by operator) and P11 closed (four-operation
certified hybrid). Successor candidates named by the record: (i) real-text trial of
the hybrid on natural corpora (mirroring ssr_lab cycle 41); (ii) nesting the hybrid
inside an LM proper at scale (operator-executed GPU protocol); (iii) multi-pass
organs beyond arithmetic (P4 open-ended: learned pass-programs for new algorithm
classes).

*Everything in this artifact reproduces from the repository: each cycle is a single
seeded self-contained script writing its own RESULT line to log.jsonl.*
