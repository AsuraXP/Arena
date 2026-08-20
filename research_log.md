# SSR — Sinkhorn State Router: Autonomous Research Log

**Agent:** autonomous AI research scientist · **Hardware:** 1–2 CPU threads, 2GB RAM, no GPU
**Date:** 2026-08-16 · **Code:** `ssr_lab/` (`models.py`, `tasks.py`, `run.py`) · **Raw results:** `ssr_lab/results.jsonl`

---

## 1. Hypothesis (Phase 1)

Diagonal SSMs (S4/Mamba) and fixed-depth transformers are confined to **TC⁰** and provably
cannot solve NC¹-complete state tracking such as the **S5 permutation-composition word
problem** (Merrill et al., arXiv 2404.08819 "The Illusion of State in State-Space Models").
Known escapes: complex eigenvalues (Mamba-3, ICLR'26), Householder transitions (DeltaNet line).

**Prior-art check (mandatory):** Sinkhorn networks exist only for *static* permutation
learning (DeepPermNet CVPR'17; Gumbel-Sinkhorn, Mena'18) and attention sparsification
(Sinkhorn Transformer). **No prior work uses input-dependent doubly-stochastic matrices as
the recurrent transition operator of a slotted state.** → mechanism registered as novel.

**SSR mechanism (final form, v2):**

```
State      S_t ∈ R^{k×d_slot}           (k slots; S_0 is a LEARNED parameter)
Codebook   {P_m}, m=1..M   free k×k matrices, projected by Sinkhorn(P_m/τ)
           onto the Birkhoff polytope (vertices = permutation matrices)
Routing    α_t = softmax(W_a h_t)   →   R_t = Σ_m α_m P̂_m
Update     S_t = R_t · S_{t-1}  [+ g_t · (w_t ⊗ v_t)  optional write path]
Readout    r_t = W_f · vec(S_t)  → residual into token stream → LM head
Inference  vertex snapping: P̂_m → exact permutation (greedy assignment),
           α_t → argmax  (straight-through during hard fine-tune)
```

Complexity **O(L·(k² + k·d_slot))** time, **O(k·d_slot)** memory — no attention, no position
embeddings, no context window. Key theoretical properties:
1. Hard permutations generate S_k ⇒ expressivity strictly above TC⁰ recurrences.
2. Permutation matrices are orthogonal ⇒ T-step gradient products neither vanish nor
   explode (unitary-RNN property for free).
3. Vertex-snapped inference composes **exact** group elements ⇒ zero error compounding
   at any sequence length.

## 2. Tasks (Phase 2)

Per-token state tracking, train L=16, eval L up to 512 (length generalization = reasoning proxy):
- `parity` — Z₂ (running XOR)
- `mod5` — Z₅ (running sum mod 5)
- `s5` — **S5 word problem**, 5 generators, 120 states, NC¹-complete

Baseline: micro-Transformer (2 layers, d=32, 2 heads, sinusoidal PE, causal mask, ~25–30k params).

## 3. Experiment ledger (Phase 3/4)

| Exp | Model / mutation | Task | Acc @16 | @128 | @256/512 | Verdict |
|---|---|---|---|---|---|---|
| 001 | TF baseline | parity | 99.9 | 56.8 | — | fits in-dist, no length gen |
| 002 | TF baseline | mod5 | 98.1 | 30.7 | — | same failure |
| 001 | SSR v1 (per-token Sinkhorn logits) | parity | 72.2 | 56.1 | — | UNDERFIT — mixing collapse |
| 003 | M1: identity-init + sharp τ | parity | 64.4 | — | — | FAIL — saturation trap |
| 005 | M2: permutation codebook | parity | 61.9 | — | — | FAIL — S₀=0 (nothing to permute!) |
| 007 | M3: learned S₀ + crisp τ + gate −3 | parity | **100** | 60.7 | — | in-dist SOLVED, soft drift |
| 008 | M3 | mod5 | **100** | 39.9 | — | same |
| 009 | M4: straight-through vertex snap | parity | **100** | 91.2 | 72.5 | big gain |
| 010 | M4 | mod5 | **100** | **100** | **100**@256 | PERFECT extrapolation |
| 011 | M5: full-state readout | parity | **100** | 99.8 | 99.2@256 | near-perfect |
| 013 | M5 hard-from-start | s5 | 32.7 | 5.0 | — | optimization failure |
| 014 | TF baseline | s5 | 48.5 | 7.1 | 3.9@256 | TC⁰ ceiling confirmed |
| 016 | M7: soft τ-anneal, hard eval | s5 | (loss 8e-4 soft) 20.8 hard | — | — | soft solves; vertex gap |
| 017 | M8: hard switch @60%, writes ON | s5 | 19.1 | — | — | write-path shortcut collapse |
| 018 | **M9: no-write + anneal + hard-ST** | s5 | **100** | **100** | **100**@256 | ★ BREAKTHROUGH |
| 023 | M9 stress | s5 | **100** | — | **100 @ L=512** | **32× extrapolation, zero decay** |
| 019 | strip n_proto 12→6, d_slot 16→8 | s5 | 26.9 | — | — | codebook slack is required |
| 021/024 | unified/crisp-reg recipes | mod5 | 26–76 | — | — | recipe bifurcation persists |

Resources (worst case): **361 MB RAM**, ≤ 75 s per training run, 1 CPU thread. No OOM ever.

## 4. Headline result

**SSR (11k params) solves the NC¹-complete S5 word problem with 100% per-token accuracy at
32× its training length (L=512 vs trained L=16), where a 2.7×-larger transformer degrades
from 48.5% → 3.9% (chance 0.83%).** Zn tasks are solved perfectly to L≥256 as well
(parity 99.2%@256, mod5 100%@256). This validates the core hypothesis: input-selected,
Sinkhorn-learned, vertex-snapped permutation transitions give a recurrent model *exact*,
length-invariant algebraic state tracking that attention fundamentally cannot express.

## 5. Empirical laws discovered

- **L-S₀:** a permutation recurrence with zero initial state is untrainable — the group
  action needs a learned non-degenerate S₀ to act on.
- **L-CRISP:** crisp (near-vertex) prototypes simultaneously fix expressivity AND gradient
  flow (orthogonality ⇒ non-vanishing products).
- **L-SHORTCUT:** with the additive write path enabled, SGD prefers TC⁰ soft-counting
  shortcuts that have no nearby vertex solution; hard-switching then collapses training.
- **L-SLACK:** discrete codebook search needs ≥2× overparameterization (n_proto ≥ 2·|generators|).
- **L-PROTOCOL:** algebraic tasks need soft-explore → hard-ST fine-tune → hard eval;
  memory-flavored tasks tolerate hard-from-start.

## 6. Training recipes (reproducible)

```
# Recipe A (algebraic / group tasks):
python3 run.py --model ssr --task s5 --steps 8000 --anneal 0.7 0.12 \
  --hard_after 0.6 --hard_eval --curriculum --n_proto 12 --no_write
# Recipe B (with memory/write path):
python3 run.py --model ssr --task mod5 --steps 1500 --tau 0.15 --hard
```

## 7. Next-cycle queue (loop continues)

1. **M11 — write/route reconciliation:** token-conditioned binary write-freeze gate with
   L0-style penalty so a single recipe covers algebraic + memory tasks.
2. **Recall benchmark:** "swap-and-query" task (interleaved transpositions + positional
   queries) — exercises group action AND write path jointly; add associative-recall probe.
3. **Composition scaling:** two-layer SSR on nested tasks (S5 × Z₅ product groups).
4. **Chunk-parallel training:** the update is affine ⇒ associative scan for O(log L) depth.
5. **Hybridization:** SSR layer as a state-tracking coprocessor inside a micro-transformer
   (SSR provides what attention provably lacks; attention provides recall SSR's k slots lack).

---

# CYCLE 11 — PRAM: Permutation-Routed Associative Machine (unification)

**Goal:** one recurrence with exact NC¹ state logic + O(1)-per-step associative recall + parallel trainability.

## Prior art (Phase 1)
- D-NTM (OpenReview BkSmc8qll): discrete addressing > soft for exact retrieval, but REINFORCE training killed it → our ST-through-Sinkhorn supplies the missing low-variance gradient.
- Grazzi et al. ICLR'25: state-tracking vs recall trade-off posed as an open hypothesis. DeltaNet: delta rule affine recurrence, parallelizable, weak length-gen (soft drift).

## Architecture
```
S_t = A_t S_{t-1} + b_t
A_t = R_t (I − g_t w_t w_tᵀ)          b_t = g_t R_t w_t v_tᵀ
R_t = (1−ρ_t)·I + ρ_t·Σ_m α_m(x_t) Sinkhorn(P_m)   (permutation codebook, opt-in)
w_t (write addr), q_t (read addr): ST one-hot;  g_t (write), ρ_t (route): ST binary
Reads: q_tᵀS_t  +  W_f·vec(S_t).  Inference fully discrete = exact register machine.
Cost: O(k²+k·d) per token, O(k·d) state, zero KV-cache.
```
**Parallel trainability PROVEN:** update is affine ⇒ associative under (A₂A₁, A₂b₁+b₂);
Hillis-Steele scan == sequential loop to 3.6e-7 (fwd) / 4.6e-5 (grads), hard mode included;
depth L → ⌈log₂L⌉ matmul stages (`scan_proof.py`).

## Benchmark: track5 (fused, cup-shuffle semantics)
5 registers × 8 values; STO_jv overwrites, G_i permutes CONTENTS (S5 generators), Q_j queried
positions only. `far` variant: stores only in first 12 tokens → up to ~1000-token pure-shuffle
desert → queries at end. Solving requires simultaneously exact group composition AND recall.

## Experiment ledger (cycle 11)
| Exp | Config | std 64/1024 | far 64/1024 | Verdict |
|---|---|---|---|---|
| 025 | PRAM hard-from-start | 27.9 / 25.8 | 22.9 / 16.9 | discrete-search stall |
| 026 | Transformer | 58.2 / 16.2 | 25.5 / 15.5 | no extrapolation, 964MB |
| 027 | SSR legacy | 29.4 / 16.9 | 25.3 / 21.6 | soft writes can't recall |
| 028 | + S5 anneal recipe | 27.6 | — | soft stall too |
| D1 | pure recall | 98.7 / 97.6 (@1024) | — | write path SOUND |
| D2 | pure shuffle (g bias 0) | 75.9 / 22.6 | — | routing poisoned by garbage writes |
| D2b | M14: write-gate bias −3 | **100 / 100 (@1024)** | — | routing SOUND |
| 030-033 | fused: ramp / mixture / dense-aux + anneal | ≤36 | — | L-COMPOSE deadlock + anneal harm |
| 034 | P3: soft+aux then hard-ST tune | 36.7 | 14.6 | hard fine-tune DIVERGES (0.11→1.79) |
| **035** | **P4: dense-aux soft → snap → NO tune** | **98.6 / 97.7** | **99.0 / 98.2** | ★ UNIFIED RESULT |

Peak RAM 385MB @ L=1024 (TF: 964MB). Params 10.9k (TF: 27.3k). All 1-CPU, ≤11 min/run.

## New empirical laws
- **L-COMPOSE:** two individually learnable circuits deadlock when jointly required from
  sparse supervision — neither can bootstrap while the other corrupts its gradient signal.
  Broken by dense state-probe supervision (deep supervision at every position).
- **L-GATE-INIT (regression of cycle-10 law):** every discrete capability must default to
  no-op at init (R→I via ρ bias −2, writes via g bias −3); live random discrete ops poison
  all other circuits' gradients.
- **L-ANNEAL-HARM:** τ-annealing helps only under sparse supervision (S5); under dense
  supervision it freezes exploration prematurely — keep τ moderate and constant.
- **L-NOTUNE:** dense-supervised soft optima are already vertex-consistent; snap directly.
  Hard-ST *fine-tuning* is unstable (reproduced 2×: initial hard loss 0.11 → divergence).

## Headline (cycle 11)
**PRAM answers the Grazzi et al. open trade-off in the negative at micro-scale: one
O(1)-per-step, zero-KV-cache, log-depth-trainable recurrence achieves 98%+ on BOTH exact
NC¹ contents-tracking and long-range associative recall simultaneously at 16× training
length (L=1024), where a 2.5×-larger transformer collapses to 16% and uses 2.5× the RAM.**

## Next-cycle queue
1. Close the last ~1.5% (static decode error): larger d_slot / longer soft phase / discrete
   value codebook with tied input-output embeddings.
2. Scale stress: k=8 registers, 16 values, L=4096 far-recall.
3. Aux-supervision ablation: anneal aux weight →0 late in training; test if binding survives.
4. Two-layer PRAM: hierarchical programs (routing-of-routers) for context-free tasks.
5. Chunked scan for O(L) work / O(log chunk) depth (block-parallel training at scale).

---

# CYCLE 12 / PHASE 6 — Tied value codebook, L=4096 stress, certification

## Context restore
Sandbox rebuilt (torch 2.13.0+cpu reinstalled). Cycle-11 checkpoint `pram_A.pt` re-verified
under hard inference: std1024 97.3% / far1024 97.4% — baseline preserved.

## M18 — Tied discrete value codebook
Write: v_t = β_tᵀC, β = ST one-hot over learned C ∈ R^{8×d_slot}.
Decode: logits += r_addr·Cᵀ (same codebook closes the loop).
Under hard inference the stored vector is EXACTLY a code row and permutations are exact,
so the value round-trip is discrete end-to-end — the continuous MLP is bypassed for values.

## EXP036 (seed 0) — headline
| eval | L=64 | L=256 | L=1024 | L=4096 |
|---|---|---|---|---|
| standard mix | **100.0** | **100.0** | **100.0** | **100.0** |
| far stress   | **100.0** | **100.0** | **100.0** | **100.0** |

**64× train-length extrapolation, zero errors** (far@4096 = recall through a ~4000-token
shuffle desert ≈ 2800 composed S5 elements). 10,722 params. Train RSS 337MB, peak 381MB
(19% of budget), eval 1.2s per 4×4096 batch, 1 CPU. Checkpoint: `pram_m18.pt`.
Bonus: tied codebook sped convergence ~10× (q-loss 0.013 @1.5k steps vs 0.14 in EXP035);
soft loss reached exactly 0.0000 → vertex-consistent optimum.

## Reliability audit (EXP037–039, seeds 0–4, same protocol)
certified-exact (hard 100% @64 ⇒ 100% @4096): **1/5** · ≥97.5%: 2/5 · all ≥86%.
- **L-CERT:** cheap certification (hard-snap eval @ L=64) has empirically implied exactness
  at all tested lengths in every run — discrete programs make finite testing meaningful.
- **L-RESTART:** an uncertified run is a *discrete local basin*: seed2 trained 20k steps
  never certified (plateau ~91%). Repair mechanism = certification-gated random restart,
  NOT longer training. Current hit rate ~20%/seed ⇒ ~5 restarts expected per exact program.

## Phase-6 verdict
Mission targets met: decode error eliminated (100.0% all cells on certified run), L=4096
stress passed with zero degradation, peak RAM 381MB ≪ 2GB. Open weakness: per-seed
reliability (20%) — promoted to next-cycle priority.

## Next-cycle queue (updated)
1. Reliability: orthogonal codebook init, EMA weights, cosine LR, prototype dropout,
   larger n_proto slack; target ≥80% certification rate; automate restart loop.
2. Aux-supervision annealing ablation (does binding survive λ_aux→0 late?).
3. Two-layer PRAM: routing-of-routers → context-free/hierarchical tasks (Dyck, tree eval).
4. Chunked scan: O(L) work / O(log chunk) depth block-parallel training.
5. Scale stress: k=8 registers, 16 values, 10-generator vocabulary.

---

# CYCLE 13 — Beyond regular languages: emergent stacks (Dyck-2)

## Roadmap decision (autonomous)
A) Reliability recipe v2 (orthogonal init + cosine LR): REGRESSED (87.3/79.9 vs ~91 v1) —
   REJECTED; constant LR is load-bearing; restarts remain the mechanism (L-RESTART holds).
B) GRAND: hierarchical structure. Prior art (DeepMind Chomsky ICLR'23; Bhattamishra'20):
   transformers & RNNs fail CF length-gen; only HAND-BUILT stack/tape memories succeed.
   H3: a stack is a permutation program — push = write(slot7)∘shift(+1), pop = shift(−1),
   top = read(slot0). PRAM's generic codebook should LEARN the stack (none built in).

## Benchmark: dyck2 (bounded depth ≤6, dense stack-top supervision, vocab {(,),[,]})

## Ledger (cycle 13)
| Exp | Config | hard@64 | @4096 | Verdict |
|---|---|---|---|---|
| 042 | Transformer | 99.9 (soft) | 11.2 | in-dist memorization, collapse, 989MB |
| 041 | PRAM, no-op priors | 62.8 | 58.5 | stack not found |
| 043 | + depth-1→6 curriculum | 62.0 | 58.6 | REJECTED — depth-1 stage builds a
      no-routing basin layout-incompatible with the stack program |
| 044 | M21: polarity-inverted priors (ρ+2, g 0) | 79.4/89.1 | 77.9/87.7 | ★ SOFT LOSS
      0.0000 both seeds — Dyck-2 solved in soft mode; vertex gap 11–21pt |
| 045 | M22: + late crispness reg λ→0.3 | 80.4/91.3 | 77.7/91.3 | marginal; seed1 length-
      INVARIANT 91.3% (exact-but-incomplete discrete program) |

## New laws
- **L-POLARITY:** the no-op-default priors (L-GATE-INIT) are TASK-POLARITY-dependent:
  correct sign = match the firing density of the target program (track5: routing sparse →
  bias −; Dyck: routing fires every token → bias +). Priors are a program-shape hypothesis.
- **L-CURRICULUM-TRAP:** a curriculum stage whose easiest solution is layout-incompatible
  with the target program creates an untraversable discrete basin (depth-1 Dyck ⇒
  write0/read0/no-shift traps against write7/shift/read0).
- **L-VERTEX-GAP:** dense-supervised soft optima crispify spontaneously only for
  routing-SPARSE programs (track5). Routing-dense programs (Dyck) retain soft prototype
  blends; snapping costs 10–20pt; late entropy pressure recovers only 1–2pt.

## Standing result
PRAM hard @4096 = 91.3% (length-invariant) vs Transformer 11.2% on bounded Dyck-2 —
first evidence the codebook can host stack semantics; exactness gap open.

## Cycle-14 attack plan (vertex gap, priority order)
1. Forensic decision dump: per token×depth, compare learned hard ops vs ideal stack
   program — identify WHICH branch is wrong before further recipe work.
2. Staged hardening: freeze+snap prototypes P first (continue training α,g,ρ,w soft),
   then α, then gates — one mechanism at a time, guarded.
3. Gumbel noise on α/ρ/g during soft phase (vertex-seeking exploration).
4. k=10 slots + n_proto 16 slack (shift program may want spare lanes).
5. Then: Dyck-3, mixed CF+recall streams (stack ∘ registers), two-layer PRAM.

---

# CYCLE 14 — Forensics, program surgery, and the Dyck existence certificate

## Method innovation: PROGRAM FORENSICS
1-layer PRAM discrete decisions depend only on token embeddings ⇒ the entire learned
program is a |V|-row op table (m,w,q,β,g,ρ per token) + hard prototypes. Dumping this
table turns interpretability into an exact debugging tool.

## Ledger (cycle 14)
| Exp | Mutation | hard@64 | @4096 | Finding |
|---|---|---|---|---|
| 046 | staged hardening (P̂ first) | 91.6 | 89.3 | dump FALSIFIED assumed mechanism: g≈0.03–0.2 (write path abandoned), '['=']' share proto → soft optimum = pure permutation automaton + ANALOG write perturbations. L-VERTEX-GAP is not noise around a vertex — it is a genuinely analog program with no nearby vertex. |
| 047 | M23 decode bottleneck (no flat read) + g bias +2 | 78.3 | 74.3 | program SHAPE fixed (4 distinct protos, push writes slot7 conf 0.99, emergent erase-on-pop) but gates stay analog 0.3–0.5 (load-bearing) |
| 048 | M24 hard gates from step 0 | 78.9 | 70.9 | REJECTED — ST boundary flicker, addressing conf drops |
| 049 | M25a program surgery (greedy 1-op hill-climb + continuous-only tune) | 92.0 | 87.0 | +4pt then LOCAL OPTIMUM — single edits can't coordinate multi-op rewiring |
| 050 | M25b EXECUTABLE EXISTENCE CERTIFICATE (hand table + continuous calibration) | **100.0** | **100.0** | ★ bounded-Dyck EXACT at 64× length; calib loss 0.000000 |

## New laws
- **L-ANALOG:** on routing-dense tasks with any analog decode channel available, SGD
  prefers high-precision analog programs (partial gates + soft blends) that have NO
  vertex equivalent. Removing the channel (decode bottleneck) fixes program SHAPE but
  gates remain the last analog refuge.
- **L-SURGERY-LOCAL:** greedy 1-op program repair recovers small deficits (+4pt) but
  stalls; coordinated multi-op rewiring needs population/beam search over programs.
- **L-CERTIFICATE:** representability claims must be EXECUTED — the hand-constructed
  stack table runs at exactly 100%/L=4096 inside PRAM semantics, formally separating
  expressivity (SOLVED: regular ∪ bounded-CF) from learnability (OPEN on Dyck: best
  learned 92%).

## Standing scoreboard (all @ L=4096, 64× train length, ≤420MB RAM, 1 CPU)
| Task class | PRAM learned | PRAM certified-program | Transformer |
|---|---|---|---|
| S5 word problem (NC¹) | **100.0** | — | 3.9 |
| fused track5 (NC¹×recall) | **100.0** | — | ~15 |
| bounded Dyck-2 (CF) | 91.3–92.0 | **100.0** | 11.2 |

## Cycle-15 attack plan
1. LEARNABILITY of Dyck: population-based program search (beam over op tables seeded by
   K gradient runs), and/or "stack-probe" token curriculum (random probe tokens that
   reveal deep stack contents — dense supervision that analog automata cannot satisfy).
2. Escalate certified machine: Dyck-3, mixed stack∘register streams, two-layer PRAM
   (hierarchical routing) — expressivity now trusted, focus on program discovery.
3. Natural-language-ish stress: character-level LM with synthetic morphology (agreement
   at distance) as first non-formal benchmark.

---

# CYCLE 15 — OpBook architecture + neuro-algebraic compilation: Dyck-2 SOLVED

## Prior art (H4): Stack Attention ICLR'24 = built-in differentiable stack primitive, soft,
no exactness. Probing literature = post-hoc only. Multi-depth PEEK supervision as a
TRAINING-signal design + gate-free op-selection recurrence: no prior art found.

## M26 — OpBook (PRAM-v3, gate-free)
Law **L-SOFTMAX-CRISP** (from 5 forensic dumps): softmax selections over structurally
distinct alternatives crispify spontaneously (α always 1.00); independent sigmoid gates
never commit. Therefore: eliminate ALL sigmoids — each token selects ONE op from a
16-entry codebook; op = (learned perm P_o, HARDWIRED write-bit g_o, per-op write slot w_o).
A_t = Σ_o α_o P_o(I−g_o w_o w_oᵀ). 4,174 params (2.6× leaner than v2).

## Ledger (cycle 15)
| Exp | Step | peek@4096 | plain@4096 | Finding |
|---|---|---|---|---|
| 051 | PRAM-v2 + PEEK task | — | 54.1 | peeks adopt pure-read ops but sigmoid gates stay analog |
| 052 | OpPRAM (M26) | 96.8 | — | ★ near-ideal emergent stack; shared push-op for (/[, erase-on-pop, per-depth read lanes |
| 053 | + greedy surgery | 96.8→98.1@64 | 97.3 | local optimum (L-SURGERY-LOCAL) |
| 054 | error census + algebra check | — | — | errors are structural: pop∘push ≠ id on lanes {3,4,7} |
| 055 | frozen-table calibration | no change | no change | loss floor 0.037 ⇒ program information-deficient |
| 056 | SYMBOLIC ORBIT ANALYSIS + repair | 92.2 | **100.0** | ★ defect identified EXACTLY: learned staging orbit length 5 = a perfect DEPTH-5 stack for a depth-6 task; orbit-swap + pop:=push⁻¹ ⇒ plain Dyck EXACT |
| 057/058 | audit: surgery had poisoned proto#12 (4↔6 swap, formerly-spare lanes); peeks → true identity op#14; recalibrate | **100.0** | **100.0** | ★★ CERTIFIED all cells, calib loss 5e-6 |

## THE PIPELINE (new methodology): NEURO-ALGEBRAIC COMPILATION
1. SGD on gate-free OpBook + probe-rich dense supervision → ~97% discrete program
2. Forensic table dump (program = |V| rows + op codebook)
3. SYMBOLIC verification against algebraic invariants (orbit lengths, inverse pairs,
   identity ops, staging-lane conflicts)
4. Minimal algebra-constrained repair (2–3 edits) + continuous-only recalibration
5. Certification at short length ⇒ empirically exact at 64× length
Laws: L-SOFTMAX-CRISP, L-ORBIT (stack capacity = staging-lane return time − 1),
L-SPARE-LANE (surgery edits on currently-spare lanes are time bombs under later repairs).

## Standing scoreboard (@ L=4096, 64× train length, 1 CPU, ≤420MB)
| Task class | best learned+repaired | Transformer |
|---|---|---|
| S5 word problem (NC¹, regular) | 100.0 (pure SGD) | 3.9 |
| track5 (NC¹ × recall) | 100.0 (pure SGD) | ~15 |
| bounded Dyck-2 ± PEEK (context-free) | **100.0 (SGD + 3 algebraic edits)** | 11.2 |

## Cycle-16 queue
1. AUTOMATE the verify-repair loop (invariant checker + repair proposer as code).
2. Dyck-3 / deeper stacks (k=12) — test repair pipeline scaling.
3. Fused stack∘register streams (CF + recall in one stream).
4. Two-layer OpPRAM — hierarchical op composition (routing-of-routers).
5. First NL-adjacent benchmark: synthetic long-distance agreement LM.

---

# CYCLE 16 — The auto-compiler: automated neuro-algebraic compilation at scale

## Prior art: automata-extraction lines (L* DFA extraction; register-automata extraction,
arXiv 2511.19100; symbolic circuit distillation) all build SURROGATES outside the model.
Novel position here: OpPRAM's architecture IS a native op-table, so verification and
repair happen IN-REPRESENTATION — the certified program remains the neural model itself.

## Deliverable: auto_compile.py / auto_compile2.py
Automated passes over any OpPRAM checkpoint:
  v1 (local): identity-snap read-ops, inverse-pairing, per-op staging-orbit extension,
     S0 normalization, greedy table edits — all scored on hard validation.
  v2 (composite canonicalization): merge all same-role tokens onto the best learned op;
  pop := σ_push⁻¹ with erase-∅ at top; peeks → identity-snapped op; orbit auto-extension;
  greedy q/β re-polish; S0 := C_∅; recalibrate; certify. Nothing hand-picked — every
  candidate scored, every choice automated.

## Ledger (cycle 16) — Dyck-3, depth ≤8, k=12, PEEK_0..2 (harder than cycle-15 task)
| Stage | cert@64 | @4096 peek/plain | wall |
|---|---|---|---|
| SGD (14k steps, 5.9k params) | 70.7 raw snap | — | 10 min |
| auto-compile v1 (5 repairs) | 98.96 | 95.8 / 96.4 | 2 min |
| auto-compile v2 (composite) | **100.0** | **100.0 / 100.0** | **67 s** |

Law **L-JOINT-MONOID:** per-op invariants (orbit length) are insufficient when SGD
splits one role across ops — verification must consider the joint action; composite
(multi-edit) repair candidates break the greedy barrier that defeats single-edit search.

## Standing scoreboard (@ L=4096, 64× train length, 1 CPU, ≤450MB peak)
| Task class | result | method | Transformer |
|---|---|---|---|
| S5 (NC¹ regular) | 100.0 | pure SGD | 3.9 |
| track5 (NC¹ × recall) | 100.0 | pure SGD | ~15 |
| Dyck-2 ±PEEK d≤6 (CF) | 100.0 | SGD + 3 manual algebraic edits | 11.2 |
| Dyck-3 ±PEEK d≤8 (CF, harder) | **100.0** | **SGD + AUTOMATED compiler** | — |

## Cycle-17 queue
1. Multi-seed / multi-task automation study: run train→auto-compile on 3 seeds × {track5,
   dyck2p, dyck3p} — measure certification rate of the full pipeline (target 100%).
2. Fused stack∘register streams (CF + associative recall in one stream, one machine).
3. Two-layer OpPRAM: op-composition hierarchies (target: A^nB^nC^n, context-sensitive).
4. NL-adjacent: synthetic long-distance agreement LM, perplexity + agreement accuracy
   vs micro-transformer.
5. Publish-grade artifact: consolidated PDF/manuscript draft from research_log.md.

---

# CYCLE 17 — Multi-counter languages + pipeline reliability

## H5: permutations as synchronized multi-counters
One permutation = product of disjoint cycles ⇒ ONE op can advance several exact counters
simultaneously ('a' = +1⊕+1 on two 6-cycles; 'b' = −1⊕id; 'c' = id⊕−1; zero-detection =
origin-marker reads). Prior art (Chomsky-hierarchy ICLR'23): LSTMs generalize aⁿbⁿcⁿ only
near training lengths; transformer counting = softmax frequencies (1/N decay); Stack-RNN
needs probabilistic tricks. Exact discrete synchronized counting: no prior architecture.

## Ledger (cycle 17)
| Exp | What | Result |
|---|---|---|
| 062 | abc SGD (soft 0.00000, raw snap) | 29.1% — max vertex gap on record |
| 063 | abc auto-compile (counter-canonicalization + recalib) | **100.0% @ 64/256/1024/4096** ★ |
| 065 | abc Transformer baseline | 100%@64 → 53.6%@4096, 987MB |
| 052-s1/s2 + 064 | dyck2p pipeline cert-rate study | seed0 ✓, seed1 ✓ (auto-compiled), seed2 ✓ RAW — **3/3 certified** |

Provenance note (honest): abc counters were TEMPLATE-INSTANTIATED by the compiler
(SGD's learned counters unusable; SGD contributed embeddings + decode). Dyck-3 remains
the strongest learned-then-repaired result. Counter LEARNABILITY → open ledger.

## Standing scoreboard (@ L=4096, 64× train length, 1 CPU, ≤450MB vs TF ~990MB)
| Task class | OpPRAM/PRAM | method | Transformer |
|---|---|---|---|
| S5 word problem (regular/NC¹) | 100.0 | pure SGD | 3.9 |
| track5 (NC¹ × recall) | 100.0 | pure SGD | ~15 |
| Dyck-2 ±PEEK d≤6 (CF) | 100.0 | SGD + manual repairs (3/3 seeds now auto) | 11.2 |
| Dyck-3 ±PEEK d≤8 (CF) | 100.0 | SGD + AUTO-compiler | — |
| aⁿbⁿcⁿ streams (counter/CS-adjacent) | 100.0 | compiler-templated counters + learned decode | 53.6 |

## Updated laws
- **L-PIPELINE (new):** train→auto-compile certification rate 3/3 on dyck2p; raw-SGD
  cert rate 1/3 — the compiler is the reliability mechanism, SGD the proposal mechanism.
- **L-TEMPLATE-BOUNDARY (new, honest):** compiler repairs so far divide into
  (a) repairs of learned structure (Dyck) and (b) template instantiation (abc counters).
  The scientific frontier is shrinking (b) into (a).

## Cycle-18 queue
1. Counter LEARNABILITY: shape abc supervision (count-probe tokens, à la PEEK) to make
   SGD discover cycle-counters organically; measure how much of the program stays learned.
2. Two-layer OpPRAM: hierarchical op composition (stack-of-counters: wᵢ#wᵢᴿ with counts).
3. NL-adjacent long-distance agreement LM vs TF (perplexity + agreement accuracy).
4. Manuscript artifact: consolidate 17 cycles, 15 laws, 5 task classes into a paper draft.

---

# CYCLE 18 — Counter-learnability (negative), nested agreement (certified), manuscript

## Ledger
| Exp | Track | Result |
|---|---|---|
| 066/068 | A: abcp count-probes + GENERIC compile (templates forbidden) | 73.9% length-invariant; dump shows NO cyclic geometry emerged. NEGATIVE: probes shape ADDRESSING, not GROUP STRUCTURE (sharpens L-TEMPLATE-BOUNDARY). Counter learnability remains open. |
| 067 | B: agree, Transformer | 100%@64 → verb 48.2%@4096 (chance; Lakretz-consistent collapse), 991MB |
| 069 | B: agree, OpPRAM + composite compile | **100.0% @ 64/256/1024/4096** ★ after two compiler generalizations: (i) ∅-class as searched parameter; (ii) POP-AS-PURE-ROTATION — when the target is the REMOVED item, pop must be g=0 rotation and the popped value is read at the STAGING lane (σ⁻¹ maps top→staging). |

## New law
- **L-POP-READ:** read-after-update machines retrieve a popped value iff pop is a pure
  rotation and the read lane is the staging lane (top ↦ staging under σ⁻¹). Erase-on-pop
  is required only when later reads visit vacated lanes (Dyck tops) — the compiler now
  distinguishes the two close-semantics automatically via scoring.

## Standing scoreboard (@ L=4096, 64× train length, 1 CPU, ≤450MB vs TF ~990MB)
| Task class | OpPRAM | TF |
|---|---|---|
| S5 (regular/NC¹) | 100.0 | 3.9 |
| track5 (NC¹×recall) | 100.0 | ~15 |
| Dyck-2 ±PEEK (CF) | 100.0 (3/3 seeds) | 11.2 |
| Dyck-3 ±PEEK (CF) | 100.0 (auto) | — |
| aⁿbⁿcⁿ (counters) | 100.0 (templated) | 53.6 |
| nested agreement (NL-adjacent) | **100.0** | 48.2 (verb) |

## Cycle-19 queue
1. Counter learnability (still #1 open problem): architectural prior — cycle-structured
   prototype parameterization (learn rotation OFFSETS per lane-block, not free perms).
2. Two-layer OpPRAM hierarchies (w#wᴿ, stack-of-counters).
3. Mixed-skill single machine: one model trained on {agree ∪ dyck2p ∪ track5} streams.
4. Manuscript upkeep (paper.md created this cycle).

---

# CYCLE 19 — CycleOp parameterization (negative), mixed-machine capacity (partial)

## H6: block-cyclic offset parameterization
Ops as ⊕_b Σ_d soft(off)_d·R_b^d — program space collapsed to 16 ops × 6×6 offsets;
counter geometry native. Prior art: cyclic-invariant CNNs = data-space symmetry;
Mamba-3 = continuous phases; discrete learned offsets with vertex inference: none.

## Ledger (cycle 19)
| Exp | What | Result |
|---|---|---|
| 070 | CycleOp on abcp, end-to-end SGD | soft 4e-5, hard 27.4% — analog persists AS OFFSET MIXTURES; 'b','c' collapse onto one op. H6's learnability claim FALSIFIED. |
| 072 | + generic offset-surgery (no templates) | 27.3→74.5%, greedy stall — coordinated-rewiring barrier reproduced in a 46k-point space |
| 071 | ONE OpPRAM on {dyck2p ∪ agree} mixed streams (k=8, 16 ops) | dyck 77.2%@4096, agree 16.7% — two-grammar interference; capacity slack untested |

## Sharpened open problem (COUNTER LEARNABILITY) — 3 failed angles now documented:
(1) dense count-probe supervision → shapes addressing only;
(2) free Sinkhorn perms + compiler → needs full template instantiation;
(3) native cyclic parameterization + generic greedy → analog offset mixtures + local optima.
Mechanism conjecture: counting programs have ZERO partial-credit gradient structure —
every near-miss offset assignment scores ≈ uniformly badly beyond n=1, unlike stacks
where partial layouts earn partial accuracy (Dyck learned to 97% organically).
Next angles: 2-edit beam search; offset-entropy annealing; curriculum on nmax with
FROZEN-hard offsets between stages.

## Cycle-20 queue
1. Counter beam-search (width 4, 2-edit neighborhoods, ~10^3 scored candidates).
2. Mixed-machine slack study: k=12, n_ops=32, per-family aux heads.
3. Two-layer OpPRAM hierarchies (w#wᴿ) — still pending.
4. Manuscript upkeep (limitations updated this cycle).

---

# CYCLE 20 — Landscape verdict on counters; capacity law for multi-grammar machines

## Ledger
| Exp | What | Result |
|---|---|---|
| 073 | Beam search (width 4, 2-edit offset pairs, 8,186 candidates) on CycleOp abcp | search-best 68.8, post-recalib 81.9 — better than greedy (74.5) but NOT certified |
| 074 | Mixed machine {dyck2p ∪ agree}, slack k=8→12, ops 16→32 | dyck 80.7 / agree 83.9 @4096 (was 77.2/16.7) — interference was op-table CONTENTION |

## Closing verdict on counter learnability (4 attack angles now complete)
Greedy (1-edit), beam (2-edit, 8k candidates), SGD (3 parameterizations), and dense probe
supervision ALL terminate on a plateau ≤0.82 while the exact program scores 1.0 with no
intermediate ridge. **L-NEEDLE (new law):** counter programs occupy needle-in-haystack
landscapes — ≥~6 coordinated edits from every attractor, zero partial-credit structure.
Consequence: for counter-class programs, gradient+search discovery is the wrong paradigm;
the compiler-template mechanism is RATIFIED as the correct one. Open problem reframed:
STRUCTURE SELECTION (differentiably choose among canonical op-role families) rather than
structure discovery. Contrast: stack programs have dense partial-credit landscapes and are
SGD-discoverable to ~97% (Dyck), needing only 2–3 algebraic repairs.

## L-CAPACITY (new law)
Multi-grammar interference in a shared op-table is a resource contention effect: doubling
op slots (16→32) + lanes (8→12) took the weaker family from 16.7% → 83.9% @4096 with both
families at parity. Program capacity scales with op-table slack, not d_model.

## Cycle-21 queue
1. Mixed-machine certification: audit op-set disjointness across families; per-family
   composite compilation on private rows; target dual-certified single machine.
2. Structure selection: op-ROLE codebook (each op draws a role from {shift-A, shift-B,
   shift-both, identity, write-shift} with learned assignment) — template CHOICE learned,
   template CONTENTS fixed — the ratified paradigm's next refinement.
3. Two-layer OpPRAM hierarchies (w#wᴿ) — pending two cycles; promote.
4. Manuscript v2 with landscape analysis section.

---

# CYCLE 21 — ISA-PRAM: the counter problem falls to structure selection

## Ledger
| Exp | Mutation | hard@4096 | Finding |
|---|---|---|---|
| 075 | RoleOp (learned 6-way role softmax per op) | 3.2 | role dists stay analog (0.41–0.76) — free op-internal params NEVER crispify |
| 076 | M31 ISA-PRAM (fixed instruction set = 6 role perms × write-bit; zero op-internal params) | 33.0 raw | dispatch 100% crisp (p=1.00) — L-SOFTMAX-CRISP at full strength; SGD dispatches a wrong-but-analog-viable program |
| 077 | + generic ISA-space beam surgery (6 edits) + recalib | **100.0** ★ | recovered the EXACT theoretical program (a→AB+1, b→A−1, c→B−1, P→id) with NO task template |

## Law synthesis (the crispness hierarchy, final form)
**L-CRISP-HIERARCHY:** per-token dispatch over ENUMERATED, structurally-distinct
instructions crispifies always; free op-internal parameters (sigmoid gates, Sinkhorn
logits, offset dists, role dists) crispify never. Architecture design rule: put ALL
discrete structure in an enumerated instruction basis; let SGD learn only dispatch and
operands; close residual gaps with generic search (tractable because the ISA space is
tiny and edit-local).

## Paradigm statement (supersedes L-TEMPLATE-BOUNDARY)
Neuro-algebraic compilation v2 = ISA enumeration (design-time, task-agnostic group
generators) + SGD dispatch learning + validation-scored beam repair + decode
recalibration + short-length certification ⇒ empirically exact programs at 64× length.
Counter-class programs — unlearnable by 4 discovery methods — certify in 2 minutes.

## Standing scoreboard (@ L=4096, 64× train length, 1 CPU, ≤480MB)
7 certified task classes: S5 · track5 · Dyck-2±PEEK · Dyck-3±PEEK · aⁿbⁿcⁿ (templated,
cycle 17) · nested agreement · **aⁿbⁿcⁿ+probes (ISA, learned dispatch — cycle 21)**.

## Cycle-22 queue
1. Port ALL prior task families to ISA-PRAM (one architecture, one pipeline): dyck2p,
   dyck3p, agree, track5 — measure cert-rate and edit-counts under the unified ISA.
2. Dual-certified mixed machine on ISA (carried from cycle 20).
3. Two-layer ISA hierarchies (w#wᴿ; promoted 3rd time).
4. Manuscript v2: crispness-hierarchy law + ISA paradigm section.

---

# CYCLE 22 — Universal micro-ISA: one instruction set across families

## H7: union basis {id, A±1, B±1, AB+1, FULL±1} × write-bit = 16 instructions.
Prior art: Differentiable Fixer (2006.10924) repairs DSL programs externally via a
learned fixer; here program = the model's own dispatch table, repair = in-representation
validation-scored search, correctness = length-invariance certification.

## Ledger (EXP078, one architecture, one pipeline, three families)
| Family | raw snap | + polish/beam | @4096 | verdict |
|---|---|---|---|---|
| agree | 85.1 | polish only, 0 beam edits | **100.0 CERTIFIED** | agreement is operand-polish-free under ISA |
| dyck2p | **99.91** (best raw stack ever) | 1.0 on search-val | 99.90 (not certified) | FULL±1 instructions are SGD-native for stacks; 0.1% branch escaped search-val — gate on certification-grade val |
| abcp | 63.6 (33.0 under 6-role ISA: union basis +30pts) | beam stalls 81.1 | 80.9 | unified surgery missing the S0 origin-marker pass that certified abcp in cycle 21 — pipeline gap, not interference |

## Findings
- NO cross-family instruction interference: each family recruits its natural subset
  (stacks→FULL±1, agreement→FULL±1 + pure-rotate pop, counters→block cycles).
- Union basis actively HELPS raw learnability (abcp raw +30.6pts) — richer crisp
  dispatch competition (consistent with L-CRISP-HIERARCHY).
- Pipeline completeness is now the binding constraint, not architecture or landscape:
  two exact, queued fixes (cert-grade polish val; S0-marker pass in unified surgery).

## Cycle-23 queue
1. Pipeline v3: port S0-marker pass + cert-grade val gating into unified.py; rerun
   dyck2p + abcp from saved checkpoints (surgery-only, minutes) — target 3/3 certified
   under ONE architecture + ONE pipeline.
2. Dual-certified mixed machine on union ISA.
3. Two-layer ISA hierarchies (w#wᴿ) — promoted 4th time, hard-scheduled next cycle.
4. Manuscript v2 (crispness hierarchy + ISA + H7 table).

---

# CYCLE 23 — Pipeline v3: unified certification achieved (3/3)

## Fixes shipped
(a) certification-grade validation for polish gating (6×24×64 + 2×8×160 batches);
(b) S0 origin-marker candidate pass in unified surgery; (c) checkpointed trainers;
(d) restart policy applied at pipeline level (L-RESTART).

## Ledger (EXP079, union ISA = 16 fixed instructions, one pipeline)
| Family | raw | surgery3 | @4096 | edits path |
|---|---|---|---|---|
| dyck2p | 99.97 | **100.0 CERTIFIED** | 100.0 | cert-grade operand polish only, 59s |
| agree | 85.1 | **100.0 CERTIFIED** (cycle 22) | 100.0 | operand polish only |
| abcp s0 | 62.6 | 81.1 stuck (sticky basin) | — | beam frozen ⇒ restart |
| abcp s1 | 58.8 | **100.0 CERTIFIED** | 100.0 | S0-pass + beam, 178s |
| abcp s2 | 51.9 | 90.3 | — | restart statistics: 1/3 certify per attempt |

**HEADLINE:** one 16-instruction ISA, one trainer, one surgery, one certification gate →
stacks (CF), nested agreement (NL-adjacent), and synchronized multi-counters ALL at
100.0% @ L=4096 (64× train length), ≤3.9k params each, ≤478MB, 1 CPU.

## Law refinement
- **L-RESTART-PIPELINE:** basin stickiness varies by seed at the DISPATCH level too;
  the restart loop belongs inside the pipeline (train→surgery→cert, repeat on failure).
  Observed abcp certify rate ~1/3 per attempt; expected ≤3 attempts per family.

## Cycle-24 queue
1. w#wᴿ as 4th family under unchanged pipeline (task-config addition only).
2. Dual-certified mixed machine on union ISA (carried).
3. Two-layer ISA hierarchies (carried; candidate task: Dyck-within-counters).
4. Manuscript v2 finalization with the completed H7 table.

---

# CYCLE 24 — The context-dispatch frontier (two instructive negatives)

## Ledger
| Exp | What | Result |
|---|---|---|
| 078-WWR | w#wR under unified pipeline (config-only) | raw 42.9 → surgery 55.7, NOT certified. Post-mortem: wwr requires phase-dependent dispatch ('a'=push pre-#, pop post-#) — outside 1-layer token-indexed op-tables. Unified pipeline's generality boundary mapped: context-free dispatch only. |
| 081-1L | modal-dyck, 1 layer | soft 0.00000 (analog mode-smuggling), hard 56.9 — expressivity wall confirmed at the DISCRETE level |
| 081-2L | modal-dyck, 2 layers | soft 0.050, hard 55.6 — naive stacking pays an ST-optimization tax exceeding the expressivity gain |

## New law
- **L-CONTEXT:** context-dependent dispatch (same token → different instruction under a
  latent mode) is the binding frontier. 1-layer: impossible discretely (dispatch is
  token-indexed). 2-layer: expressible but not learnable at micro-scale (ST through two
  recurrences). Both negatives reproduce the analog-smuggling signature in soft mode.

## M32 registered (cycle-25 headline design) — CASCADED DISPATCH
stage-1: tiny pure-permutation MODE AUTOMATON (solved SSR technology; group state m_t,
associative-scan-parallel, certifiable). stage-2: ISA machine with dispatch α(token, m_t)
— an enumerated (token × mode) table, still the always-crisp selection channel. Predicted
to cover modal-dyck AND wwr with the existing surgery/certification pipeline extended to
contextual tables ((VIN × |M|) rows).

## Cycle-25 queue
1. M32 implementation + modal-dyck & wwr certification runs.
2. Dual-certified mixed ISA machine (carried).
3. Manuscript v3: context-dispatch chapter.

---

# CYCLE 25 — KR-ISA: the neural Krohn-Rhodes cascade

## Theory anchor (prior art)
Krohn-Rhodes: every finite automaton = feed-forward cascade of permutation-reset
components. Stage-1 basis {id, const_j, shift} IS a permutation-reset component; stage-2
is the permutation-register ISA machine. KR theory recently used to characterize
transformer expressivity (star-free <-> reset cascades, STACS'25); no prior trainable,
vertex-certifiable neural KR cascade. Architecture: models6.py, 3.1k params; both stages
associative (matrix-monoid scan) => parallel-trainable by construction.

## Ledger (cycle 25)
| Exp | What | Result |
|---|---|---|
| 082 | KR-ISA raw SGD on modal-dyck | soft 85/60/55, hard 38/31 — mode channel misprogrammed (shift instead of const; analog mixtures). MODE-GRADIENT STARVATION: mode affects loss only through downstream table mixing. |
| 083 | contextual surgery (mdisp + 24-context tables, pair beam) | stalls 68.5% — joint mode×table space re-creates the needle (repairing modes invalidates all context tables simultaneously) |
| 084 | EXECUTABLE EXISTENCE CERTIFICATE | **100.0% @ 64/256/1024/4096**, calib loss 0.000000 ★ — context-dependent dispatch exactly expressible; 1L=56.9, naive 2L=55.6, TF-class analog: all fail |

## Laws
- **L-KR:** the ISA paradigm extends one Chomsky-relevant axis further: enumerated
  permutation-reset mode components + contextual op-tables = exact context-dependent
  dispatch, scan-parallel, zero KV-cache.
- **L-MODE-STARVE:** indirectly-supervised discrete channels (mode) receive gradient
  only through downstream mixtures ⇒ SGD misprograms them; joint post-hoc repair is
  needle-class. Predicted fix (M33): direct probe supervision on the mode state +
  STAGED COMPILATION (certify mode automaton on probes first, freeze, then the 24
  context-local table problems decouple).

## Cycle-26 queue
1. M33: mode-probe supervision + staged compilation → target LEARNED modal-dyck cert.
2. wwr under KR-ISA + feedback frontier (state-conditioned mode flips) — design study.
3. Dual-certified mixed machine (carried).
4. Manuscript v3: KR chapter + full law index (17 laws).

---

# CYCLE 26 — Staged compilation for KR cascades: three refinements, frontier narrowed

## Ledger
| Exp | Stage | Result |
|---|---|---|
| 085 | M33 mode-probe supervision, raw SGD | DEFEATED by new analog bypass: mode stored as REGISTER VALUE (write at M-tokens, read at MP), mode automaton never programmed. **L-MODE-STARVE-II:** direct supervision on a control state is insufficient when the universal register sink shadows it. |
| 086 | staged enumeration (36 stage-1 hypotheses × 1-round stage-B) | 28.2→75.9; selection biased by unequal stage-B budget (near-miss md=[c0,shift] won) |
| 087 | deep equal-budget selection over const-family | TRUE program md=[c0,c1] correctly selected (77.5); context-greedy cannot REBUILD stage-2 (greedy can repair, not construct — consistent with all prior laws) |
| 088 | + stage-2 SGD retrain under frozen-correct modes (with probes) | soft 0.105 / hard 38.5 — probe scratch-circuitry + 5-class decode interfere with the stack program |

## Pipeline theory consolidated (division of labor, now empirically forced):
  ENUMERATION selects control structure (stage-1 hypothesis space is tiny) ·
  SGD constructs content programs (stacks: 99.9 raw when contexts are clean) ·
  SEARCH repairs local defects (2-6 edits) · CALIBRATION fixes decode ·
  CERTIFICATION gates everything. Each mechanism provably cannot do the others' jobs.

## Cycle-27 queue
1. Corrected staged pipeline: stage-1 select on probes → freeze → stage-2 retrain on
   PLAIN modal (probes removed post-selection) → snap/polish → certify. Prediction:
   replicates dyck-class raw (≥99%) ⇒ first fully-learned+compiled context dispatch.
2. wwr feedback-dispatch design study (state-conditioned mode flips).
3. Manuscript v3 (KR chapter + 19-law index + division-of-labor theory).

---

# CYCLE 27 — Control experiment isolates the true defect (implementation, not concept)

## Ledger
| Exp | What | hard raw | Finding |
|---|---|---|---|
| 089 | stage-2 SGD, frozen-correct modes, plain modal | 55.9 | still failing |
| 090 | + no-op-biased table init (L-GATE-INIT applied) | 67.8 | +12pts, law transfers to contextual tables |
| 091 | + contextual surgery | 70–75 stall | cross-context geometric-consistency defect suspected |
| 092 | M34 minimal contextualization (β-only context) | 29.1 | regressed — suspicion shifts off the modal axis |
| 093 | **CONTROL: KRISA on plain dyck2p (modes inert)** | **49.6** | ★ DECISIVE: identical task where RoleOpPRAM = 99.91. The KRISA implementation degrades learning; modal dispatch was never the stage-2 blocker. |

## New law candidate (cycle-28 test)
**L-EMBED-COUPLING:** dispatch parameterized as linear maps on SHARED token embeddings
(α=W·emb) trains far better than direct per-token parameter tables (softmax(T[x])) —
embeddings couple dispatch, residual, and readout gradients into one geometry; isolated
table rows learn each context from scratch. Fix: KR-ISA v2 with linear heads over
[emb(token); mode-onehot] — contextual capability + embedding coupling.

## Method note
The control experiment (modal machinery inert on a solved task) should have been run at
cycle 26 — two cycles of modal-specific hypotheses (mode starvation, cross-context
consistency, minimal contextualization) were partially confounded by an orthogonal
implementation regression. Control-first is now standing protocol for new model classes.

## Cycle-28 queue
1. KR-ISA v2 (linear contextual dispatch) — control on dyck2p first (target ≥99 raw),
   then frozen-mode modal stage-2, then full staged pipeline → learned context dispatch.
2. wwr feedback-dispatch design study (carried).
3. Manuscript v3 (carried; add control-first protocol note).

---

# CYCLE 28 — KR-ISA v2: L-EMBED-COUPLING confirmed; modal attractor is method-invariant

## Ledger
| Exp | What | raw / final | Finding |
|---|---|---|---|
| 094-ctrl | KRISA2 (linear ctx dispatch) on plain dyck2p | 78.1 | L-EMBED-COUPLING confirmed directionally: +28.5pts over table dispatch (49.6); residual gap to RoleOpPRAM (99.9) unexplained — single-seed comparisons; multi-seed control queued |
| 094-modal | KRISA2, frozen-selected modes | 55.8 | soft 0.168 — not solved |
| 095 | + bounded contextual surgery (greedy + top-30 pair beam + recalib) | 68.5 | SAME ~68–70% attractor as all six prior attacks |

## Cross-cycle synthesis: THE MODAL ATTRACTOR (open problem, formally logged)
Seven attacks (cycles 24–28): 1L tables · probes · hypothesis enumeration · staged SGD
under frozen-correct modes · β-only contextualization · linear contextual dispatch ·
greedy/pair-beam surgeries — ALL terminate at 56–75%, length-invariant. Certificate
program exists and executes at 100.0%@4096 (EXP084). Characterization: the task couples
(i) a mode-consistent shared stack geometry with (ii) mode-dependent value coding;
SGD's soft optima satisfy the loss analogically without either; every discrete repair
path crosses a wide fitness valley. This is the strongest needle instance on record
(counters fell to ISA enumeration; this resists even enumeration because stage-2 content
must be CONSTRUCTED, and SGD constructs only in clean-context settings).

## Operational pipeline fault logged
Unbounded pair-beam (64k candidates) caused a wall-clock timeout — bounded to top-30
single-edit preselection (435 pairs). Standing rule: all search loops budget-bounded.

## Cycle-29 queue (redirect per plateau rules)
1. Multi-seed control study: RoleOpPRAM vs KRISA2 on dyck2p, 3 seeds each — settle the
   residual L-EMBED-COUPLING gap (variance vs mechanism).
2. Dual-certified mixed machine (deferred 4×; promote to mandatory).
3. Manuscript v3 (deferred; promote to mandatory): 21 laws, modal-attractor dossier,
   division-of-labor theory, control-first protocol.
4. Modal attractor: parked with full dossier; revisit tooling = population SGD (8
   restarts × cert gate) once cheaper per-run training exists.

---

# CYCLE 29 — Multi-seed adjudication + manuscript v3

## Ledger
| Exp | Arch | dyck2p raw across seeds | Verdict |
|---|---|---|---|
| 078-S1/S2 + prior | RoleOpPRAM | {99.9, 61.9, 100.0} (s1: surgery 61.9→95.8; s2 raw-CERTIFIED) | |
| 096-S1/S2 + prior | KRISA2 | {78.1, 76.9, 100.0} (s2 raw-CERTIFIED) | |

**ADJUDICATION:** the cycle-28 "L-EMBED-COUPLING residual gap" (99.9 vs 78.1) was a
seed-lottery artifact — population means ≈87 vs ≈85 (n=3, overlapping ranges).
L-EMBED-COUPLING downgraded to n=1-supported. **L-SEED-LOTTERY finalized:** raw-SGD
certified rate ≈1/2 pooled (3/6), individual range 62–100; architecture choice moves the
mean far less than seed draws move individuals; reliability is a PIPELINE property
(certify → restart → surgery), not an architecture property. KRISA2 reaches 100.0 raw
(s2) — linear contextual dispatch is structurally sound.

## Manuscript v3 shipped (paper.md)
21-law index · neuro-algebraic compilation with division-of-labor theory · 7-row
certified scoreboard · modal-attractor + feedback-dispatch open-problem dossiers ·
protocol register (control-first, bounded search, multi-seed evidence, cert-grade
gating). 29 cycles, 131 experiments, 6 architecture generations, 9 benchmark families.

## Cycle-30 queue
1. Dual-certified mixed machine (deferred 5×; hard-mandatory next).
2. Population-SGD × cert-gate tooling (8 cheap restarts/run) — then re-attack the modal
   attractor with statistics instead of single draws.
3. Feedback-dispatch design study (wwr): state-conditioned mode transitions — the first
   architecture question beyond feed-forward KR cascades.

---

# CYCLE 30 — Dual-certified single machine; modal attractor withstands the lottery

## Ledger
| Exp | What | Result |
|---|---|---|
| 097 | dual-family canonical compile, v1 | agree CERTIFIED in shared machine; dyck 90.4 — two defects found: peek off-by-one (PEEK_0 = top), polish-before-recalib ORDER BUG (argmax flat under uncalibrated decode) |
| 098 | v2 (recalib → polish → recalib) | ★★ **DUAL-CERTIFIED: dyck2p+PEEK AND nested agreement BOTH 100.0% @ 64/256/1024/4096 in ONE 9.1k-param machine** (shared push op across grammars; family-split pop per L-POP-READ; shared S0/decode; recalib loss 0.000000; 38s compile) |
| 096-S3/S4 | modal lottery arm | 34.3 / 22.0 — attractor stands at n=5 seeds {56,29,51,34,22}; landscape-limited, not seed-limited |

## New protocol rule
**Polish-after-calibration:** discrete polish is blind under an uncalibrated decode
(argmax plateaus); pipeline order is recalibrate → polish → recalibrate. (Second
order-dependency bug found by differential debugging; added to the standing register.)

## Milestone summary
Certified classes now include a MULTI-DOMAIN machine: two grammars, one op-table, both
exact at 64× training length. Deferred-milestone queue is empty for the first time
since cycle 22.

## Cycle-31 queue
1. Triple-domain machine: + abcp counters into the shared machine (op-table has 30 spare
   slots; blocks A/B unused by the FULL-shift stack ops — capacity analysis favorable).
2. Feedback-dispatch design study (wwr; state-conditioned mode transitions) — the
   standing architecture frontier.
3. Modal attractor: parked (n=5 dossier); revisit only with a new mechanism class.

---

# CYCLE 31 — Triple-domain machine + feedback dispatch: two certifications

## Ledger
| Exp | What | Result |
|---|---|---|
| 099 | TRIPLE MACHINE: dyck2p+PEEK ∪ agreement ∪ abcp+probes in ONE op-table (12 lanes, 32 instr) | ★ **100.0% on all 9 cells @ 64/1024/4096**; recalib 1e-6; 46s. Lane time-sharing + counter origin-markers at S0[3]/S0[9] proven safe by the surfaced-index analysis ((j−depth) mod 12 ∈ {0,1,2}); count decode via S0-as-position-code through the shared head. |
| 100 | Feedback certificate v1 (wwr) | 82.5, loss floor 0.36 — decode ambiguity found by inspection: control state m not exposed to readout |
| 101 | + mode-embedding in readout (control state is architecture state) | ★ **100.0% @ 64/256/1024/4096** — feedback-dispatch machine class certified; wwr (cycle-24 failure) now exact |

## New capability class
**FB-KR machine:** permutation-reset cascade + ONE bit of data→control feedback
(hard-thresholded register read). Costs the associative scan (sequential control), keeps
O(1)/step and zero KV-cache. Covers state-conditioned phase structure (w#wR class).

## Standing certified scoreboard (@ 64× train length, 1 CPU, ≤480MB)
Single-domain: S5 · track5 · Dyck-2±PEEK · Dyck-3±PEEK · aⁿbⁿcⁿ(+probes) · agreement ·
modal-dyck (certificate) · wwr (FB certificate). Multi-domain: DUAL machine (2 grammars)
· **TRIPLE machine (3 grammars, 9/9 cells)**. Open: modal-dyck LEARNABILITY (n=5 dossier).

## Cycle-32 queue
1. FB-KR learnability probe: can SGD+pipeline learn the feedback program on wwr?
   (control: the certificate exists; apply control-first + staged protocols.)
2. QUAD machine: + wwr via FB into the triple machine (feedback bit is family-gated).
3. Manuscript v4: multi-domain + feedback chapters.

---

# CYCLE 32 — The quad machine; feedback learnability adjudicated

## Ledger
| Exp | What | Result |
|---|---|---|
| 102 | FBISA (trainable feedback machine) on wwr, 2 seeds | 16.0 / 41.9 raw — feedback programs NOT SGD-learnable (three stacked indirectly-supervised channels: ê-probe × mode table × dispatch). Expected-negative; feedback joins counters and modal dispatch in the ENUMERATE/COMPILE column. |
| 103 | QUAD machine v1 | 3 families 100.0, wwr 83 — EXP100 signature (control state not exposed to readout); executor patched per EXP101 |
| 104 | QUAD machine v2 | ★★ **100.0% on ALL 12 cells @ 64/1024/4096** — 4 grammars, one machine, loss 2.6e-4, 100s compile |

## Instruction economy (new observation)
The quad machine REUSES instructions across grammars: push op0 serves stack, agreement,
and copy-reverse; pure-rotation pop op16 serves agreement and copy-reverse; block cycles
18-20 serve counters; identity op17 serves five roles. 7 active instructions of 32 host
four task families — shared-primitive compression, the ISA thesis at machine scale.

## Division-of-labor table (final adjudication, all columns now populated)
| program class | SGD-learnable | pipeline route |
|---|---|---|
| group tracking / recall (S5, track5) | YES (pure SGD, 100%) | none needed |
| stacks / agreement | MOSTLY (62-100% raw, seed lottery) | polish + restart |
| counters | NO (L-NEEDLE) | ISA dispatch learned + beam edits |
| context dispatch (modal) | NO (n=5 attractor) | certificate (learnability open) |
| feedback dispatch (wwr) | NO (16-42% raw) | enumeration + certificate |

## Cycle-33 queue
1. Manuscript v4 (quad machine, feedback chapter, final division-of-labor table).
2. Quint escalation candidate: add S5/track5 (pure-SGD families) into the machine.
3. New frontier scan: probabilistic/weighted outputs (LM-style soft targets) on the
   certified machine substrate — first step toward token-prediction proper.

---

# CYCLE 33 — Token prediction proper: oracle-level LM from certified state

## Benchmark: stochastic-Dyck LM (analytic oracle)
Bounded-depth random bracket streams with per-state ground-truth next-token
distributions; metric ΔCE = model CE − oracle CE (nats/token). Prior art: perplexity
methodology standard; no oracle-referenced extrapolation benchmark for exact-state
recurrences found.

## Ledger
| Exp | Model | ΔCE @64 | @1024 | @4096 | RAM |
|---|---|---|---|---|---|
| 105 | Machine-LM (compiled dyck program + [top, boundary-lane] reads + trained head) | 0.0031 | 0.0027 | **0.0031** | 328MB |
| 106 | Transformer LM (2L, trained to CE 0.93 in-dist) | 0.0049 | 2.0045 | 1.9746 | 984MB |

**650× lower excess CE at 64× context, length-INVARIANT.** The 0.003 floor is head
capacity (shared across lengths), not state error. Law **L-SUFFICIENT:** an exact
discrete state is an exact sufficient statistic — certified reasoning machines transfer
to the probabilistic LM objective with zero length decay; soft-state models drift to the
guessing regime (~2 nats) precisely where their state estimate degrades.

## Cycle-34 queue
1. Manuscript v4 finalization (LM chapter added this cycle; quad + feedback chapters).
2. Richer LM grammars: stochastic modal/agreement mixtures with oracle CE; multi-family
   LM on the quad substrate.
3. Learned-LM variant: can the unified SGD+compiler pipeline reach the 0.003 floor from
   scratch on the LM objective (targets soft, no per-token state labels)? — the last
   supervision regime not yet tested.

---

# CYCLE 34 — The machine emerges from the raw LM objective

## Ledger (EXP107, LM-ISA: discrete transitions + continuous decode, raw LM loss only)
| seed | soft dCE @64/1024/4096 | HARD dCE @64/1024/4096 | verdict |
|---|---|---|---|
| 0 | 0.019 / 0.315 / 0.554 | 1.122 / 1.154 / 1.148 | lottery miss |
| 1 | −0.001 / 0.002 / 0.042 | **0.00078 / 0.0047 / 0.0056** | ★ ORACLE-LEVEL, FLAT, DISCRETE |

Firsts: (a) full discrete program learned from SOFT probabilistic targets — no state
labels, no probes, no compiler; (b) hard mode BEATS its own soft mode at 4096 (snapping
removes soft drift); (c) learned machine beats the hand-compiled one at 64
(0.0008 vs 0.0031 — head co-training).

## Program dump (seed 1) — a solution class not in any of my designs
'(' → A+1, ')' → A−1, '[' → FULL+1, ']' → FULL−1; ALL pure-route (zero writes).
SGD invented a WRITELESS stack: bracket history as a group word in ⟨A±1,F±1⟩ ≤ S12,
LIFO cancellation exact by construction, top-type decoded from the S0-orbit position.
Dispatch confidence 1.00 across the table.

## New law
**L-SOFT-TARGETS:** probabilistic LM targets supply calibrated partial credit at every
position — a smooth score on state quality that 0/1 classification lacks. The
supervision regime presumed hardest (no labels at all beyond next-token) is where
discrete programs emerge organically. Retrospective: five cycles of probe engineering
addressed a deficiency of the CLASSIFICATION objective, not of the architecture.

## Cycle-35 queue
1. LM-objective retraining across the full task family suite (does L-SOFT-TARGETS
   generalize beyond Dyck? modal-dyck-as-LM is the big one — the parked attractor may
   fall to soft targets).
2. Seed-population statistics for LM-ISA (n=6) + pipeline integration (cert gate on dCE).
3. Manuscript v5: L-SOFT-TARGETS chapter; the writeless-stack solution as a case study
   in machine-invented programs.

---

# CYCLE 35 — Soft targets vs the modal attractor: control falls, content holds

## Ledger
| Exp | Setup | mode table | hard dCE @4096 | Finding |
|---|---|---|---|---|
| 108-s0/s1 | LM-KR, everything learnable, raw LM loss | s1: M0→c1, M1→c2, brackets→id @ p≥0.97 — **CORRECT CONTROL STRUCTURE, learned organically** | 2.9 / 9.7 | ★ the control program that 7 supervised attacks never produced crystallizes from soft targets; stage-2 stays analog |
| 109-s2/s3 | learned modes frozen hard, stage-2 on LM loss | (frozen) | 1.24 / 1.29 | modal stage-2: 0/4 LM seeds crystallize |

## Law refinements
- **L-SOFT-TARGETS (scope sharpened):** soft targets crystallize CONTROL automata
  (mode tables: 2/2 structurally correct) and non-modal stacks (EXP107: 1/2), but NOT
  yet mode-conditioned content programs (0/4). Mechanism note: the writeless-stack
  solution class likely breaks under mid-word mode swaps (generator semantics change
  destroys exact LIFO cancellation), removing the easiest crystallization path.
- Modal dossier updated: attractor HALF-dissolved — control half solved organically;
  remaining open problem is narrower and precisely characterized.

## Cycle-36 queue
1. Modal stage-2: seed population n=8 at reduced steps (lottery statistics), and/or
   write-based curriculum (pretrain on plain-dyck LM, then transfer to modal streams —
   transfer should preserve the crystallized stack while modes attach).
2. Manuscript v5 (soft-target chapter + modal half-resolution).
3. Escalation scan: two-mode-token grammars with 3+ modes; natural-ish morphology LM.

---

# CYCLE 36 — Transfer curriculum vs modal content: three clean negatives

## Ledger
| Exp | Setup | hard dCE @4096 | Finding |
|---|---|---|---|
| 110-A s0/s1 | write-forced ISA (route ops 9-15 masked), plain-dyck LM | 0.38 / 1.19 (+0.105 oracle-bias corr.) | write-family stacks do NOT crystallize under LM loss — masking removed the one family (writeless) that LM loss likes |
| 110-B | modal fine-tune of best stage-A | 12.7 | transfer cannot bridge the structural break; fine-tune destroys partial hard structure |
| — | oracle p_mode-reservation bias | — | caught via impossible negative dCE; tasks11 fixed (mass rescaled by 1−p_mode) |

## Modal-content dossier (final form, 3 supervision regimes)
supervised classification: 7 attacks, 56–75% attractor · organic LM: 0/4 seeds (control
automaton DOES crystallize — cycle 35) · masked/transfer LM: 0/3. Mechanistic core: the
LM-crystallizable solution (writeless group-word stack) is structurally incompatible
with mid-word generator-semantics swaps; the compatible family (write-based) has no
LM-loss crystallization path found. PARKED with the program's deepest dossier.

## Law refinement
**L-SOFT-TARGETS (final scope):** soft targets crystallize control automata and
UNCONSTRAINED-basin content programs; constraining the basin (masking) or breaking the
preferred solution class (semantics swaps) removes the effect. Crystallization follows
the loss's preferred solution family — it cannot be steered by structural masks.

## Cycle-37 queue
1. Manuscript v5 (overdue): soft-target chapter, modal tri-regime dossier, 25-law index.
2. Breadth escalation: 3-mode grammars; morphology-flavored oracle-LM on quad substrate.
3. Modal content: PARKED (revisit trigger: any new crystallization mechanism).

---

# CYCLE 37 — Morphology-LM replication + manuscript v5

## Ledger
| Exp | Model | dCE @64 / @1024 / @4096 | RAM |
|---|---|---|---|
| 111 | Machine-LM (compiled agree program + [top,boundary] reads + head) | **0.0015 / 0.0021 / 0.0015 (FLAT)** | 328MB |
| 112 | Transformer LM | 0.0022 / 1.416 / 1.508 | 983MB |

L-SUFFICIENT replicated on the NL-closest benchmark (nested-agreement morphology with
verb-form prediction): 1000x lower excess CE at 64x context, zero decay.

## Manuscript v5 shipped (paper.md): 14-row certified scoreboard, 25-law index,
machine-invented-program case study, tri-regime modal dossier, standing protocols.

## Cycle-38 queue
1. LM-crystallization statistics: LMISA n=6 population on stochastic-Dyck (cert-gate on
   hard dCE) — quantify L-SOFT-TARGETS lottery precisely.
2. Escalation: mixed-grammar oracle-LM on the quad substrate (one LM head, 4 languages).
3. Modal content: parked (revisit trigger unchanged).

---

# CYCLE 38 — Polyglot oracle-LM + LM-crystallization lottery quantified

## Ledger
| Exp | What | Result |
|---|---|---|
| 113 | POLYGLOT-LM: one machine, one head, three languages (dyck ∪ morph ∪ abc-LM, union vocab 13) | ★★ ΔCE 0.0009–0.0039 on ALL 9 cells, FLAT to L=4096 (dyck .003 / morph .003 / abc .0004) — instruction economy transfers to the probabilistic objective |
| 107-s2/s3 | LMISA lottery (population n=4) | {1.12 ✗, 0.0056 ✓, 0.0050 ✓, 1.77 ✗} → **crystallization rate 2/4 (50%)**, matching the classification lottery. |

## New tool: label-free LM certification gate
Crystallized runs satisfy: hard ≈ soft ΔCE @64 AND hard < soft @4096 (snapping removes
soft drift; e.g. seed2: 0.005 hard vs 0.113 soft @4096). Non-crystallized runs show
hard >> soft everywhere. Gate needs only the oracle-free CE trend across lengths →
restart policy applies to LM training with zero labels.

## Cycle-39 queue
1. Wire the label-free gate + restart loop into an autonomous LM-training daemon
   (train → gate → restart until crystallized) — full pipeline for the LM regime.
2. Polyglot escalation: add wwr-LM (feedback) as 4th language.
3. Manuscript v6: polyglot chapter + LM lottery statistics + certification gate.

---

# CYCLE 39 — The zero-supervision daemon: train → gate → restart → certify, no labels

## Ledger (EXP114, fresh seeds 4-7; characterized seeds 0-3 excluded for honesty)
| seed | gate (label-free) | post-gate oracle dCE @64/1024/4096 |
|---|---|---|
| 4 | REJECT (near-miss: passes @64, +0.13 hard-soft drift @1024) | — |
| 5 | REJECT (clear, +1.1) | — |
| 6 | REJECT | — |
| 7 | **PASS** | **0.0050 / 0.0050 / 0.0041 — FLAT** ★ |

Gate definition (zero oracle, zero labels): PASS ⟺ CE_hard−CE_soft < 0.05 @64 AND
CE_hard ≤ CE_soft + 0.02 @1024 (snapping-removes-drift signature). Validation over all
8 characterized seeds: 0 false positives, 0 false negatives.

## Significance
The complete discovery loop now runs unsupervised end-to-end: raw next-token loss
trains; the hard/soft CE divergence detects crystallization without any ground truth;
restarts handle the ~40% lottery; certification-by-length-invariance follows. Combined
population statistics (n=8): crystallization 3/8; expected attempts per certified
machine ≈ 2.7, ~22 min on 1 CPU.

## Cycle-40 queue
1. Manuscript v6: daemon chapter + gate validation table (final artifact refresh).
2. wwr-LM 4th polyglot language (carried).
3. Frontier scan: apply the daemon to the morphology-LM and polyglot regimes
   (end-to-end learned polyglot machine — currently compiled).

---

# CYCLE 40 — Polyglot crystallization: interference law + curriculum partial remedy

## Ledger
| Exp | Setup | gates (dyck/morph/abc) | Finding |
|---|---|---|---|
| 115 s0/s1 | joint polyglot LM training | ✗✗✗ / ✗✗✗ | joint dispatch finds soft blends satisfying all losses — crystallization fully suppressed |
| 116 | sequential curriculum (dyck→+morph→+abc) | ✗ / half / **✓** | ★ abc CERTIFIED inside the polyglot machine (oracle dCE 0.0035/0.0032 @64/4096 FLAT); morph hard<soft @1024 (partial); dyck lost |

## New law
**L-POLY-INTERFERENCE:** multi-language dispatch sharing suppresses LM crystallization
(0/6 family gates joint vs 3/8 single-language); sequential curriculum partially
restores it — and the crystallization ORDER inverts the classification-regime hardness
ranking (counters crystallize first under LM loss; stacks last). Fix hypothesis for
cycle 41: family-private dispatch heads over the shared instruction set (the LM
analogue of L-CAPACITY's op-slot slack).

## Cycle-41 queue
1. Family-private dispatch polyglot (shared ISA, per-family alpha) + daemon gates.
2. Manuscript v6 finalization (daemon + polyglot chapters).
3. wwr-LM 4th language (carried).

---

# CYCLE 41 — GPU-SCALE VALIDATION ON REAL TEXT (user-executed, Colab T4)

## Protocol: scripts built+smoke-tested in sandbox, executed by operator on GPU.
Corpus: 60MB TinyStories (real English), word-level vocab 8192, matched-scale arms,
trained ctx=256, evaluated 256/1024/4096. Three rounds.

## Round 1 (synthetic oracle mix, 2 seeds): mixed-language training suppressed
crystallization (L-POLY-INTERFERENCE confirmed on GPU; label-free gate 0 false
positives). SURPRISE: soft recurrent machine length-invariant on BOTH tasks,
beating TF 20x on fuzzy @4096 (positional-OOD artifact identified).

## Round 3 — CONFIRMATION RUN (3 seeds x 3 archs, real text, 101.5 GPU-min):
| arch | params | ce@256 | ce@1024 | ce@4096 |
|---|---|---|---|---|
| tf_sin | 5.26M | 3.30 | 6.31 | 6.84 |
| tf_rope (fair baseline) | 5.26M | **2.18** | **3.04** | 4.49 |
| machine (ours) | 2.41M | 3.28 | 3.27 | **3.26** |
Seed ranges tight, non-overlapping. Samples: tf_rope clearly most fluent; machine
coherent, attention-free.

## FINAL ADJUDICATION (the program's headline pair)
1. FALSIFIED: "machine beats transformer at short-context fluency" — held only vs
   the weak sinusoidal baseline; RoPE wins in-distribution by 1.1 nats. Attention is
   genuinely superior at local statistical prediction.
2. CONFIRMED AT SCALE, REPLICATED: the machine is length-INVARIANT on real English
   (3.28 -> 3.26 from 256 to 4096, n=3) — the certified toy-grammar property
   transfers to natural language. Crossover vs RoPE at ~1.5-2k tokens; +1.2 nats
   better at 4096. Caveat: claim is precisely "train short, generalize long."
3. Architecture conclusion of the 41-cycle program: HYBRID (attention for local
   fluency + register machine for unbounded state) — both halves now proven with
   GPU-scale data. Round-1 hybrid fusion design failed; hybrid-v2 (RoPE + gated
   coprocessor) is the open engineering item.

---

# PROGRAM CLOSED — FINAL STATUS (cycle 41)
41 cycles, 166+ experiments, 26 laws, 12 benchmark families, 8 architecture
generations. Headline pair: (1) certified exact reasoning at 64x length where
transformers are provably incapable; (2) GPU-replicated length-invariance on real
text (flat 256->4096 vs RoPE degradation), with the honest falsification of
short-range superiority. Contribution class: neuro-algebraic compilation +
certified register machines. Successor program: see ../arc2/CHARTER.md
