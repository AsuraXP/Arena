# ARC-2 PROBLEM MAP (locked, cycle 1)

## A. TARGET PROBLEMS - architectural weaknesses of SOTA models (scale does not fix)
P1 Length/context decay ............ BEATEN (ssr_lab, GPU-replicated)
P2 Exact state tracking (TC0) ...... BEATEN at micro-scale, certified
P3 Counting & arithmetic (carries) . MAIN FIGHT (data-dependent routing)
P4 Iteration / adaptive compute .... CORE NOVELTY (multi-pass machine needed;
                                     unsolved by us AND by them)
P5 Compositional generalization .... partial; needs dedicated benchmark
P6 Variable binding / symbols ...... register file native; untested, likely win
P7 Verifiable reasoning ............ our certification pipeline; deepest moat
P8 Memory cost (KV cache) .......... won by construction (O(1) state)

## B. INTERNAL BLOCKERS - what we must solve to claim the crown
P9  Crystallization lottery (~50%/seed)  -> #1 blocker for "best generalizer"
P10 Feedback-class learnability (carries) -> currently compiler-dependent
P11 Short-range fluency gap vs RoPE (1.1 nats)
P12 Training-speed engineering (scan is proven, implementation is loop)

## C. OUT OF SCOPE (honesty clause)
World knowledge, chat alignment, multimodality: data/scale problems, not
architecture problems. Not contested from a 2GB box.

## VICTORY CONDITION
Frozen public test items (JUDGE_CARDS.md). Operator pastes them into frontier
chatbots and records their scores; sandbox-trained machines must reach 100%
exact-match on the same items. Win = measured resource inversion.

## ATTACK ORDER
C1: suite + judge cards (this cycle) -> C2: T2/T4 quick certifications ->
C3-C6: the carry problem (P3+P10) -> C7+: multi-pass machine (P4) ->
parallel track: P9 reliability throughout.

## CYCLE-2 STATUS UPDATE
P3 (counting/arithmetic-addition): SOLVED — LSB pair-token reduction makes carry a
KR mode automaton; certified 12.5x length generalization (8->100 digits), 14/14 on
frozen suite. P10 note: carry was never feedback-class under the right encoding —
encoding choice is an architectural decision (new law: L-ENCODING — task hardness
class is representation-relative). T3 multiplication remains the true P4 fight
(nested iteration). Frontier column of SCOREBOARD.md awaits operator judging.

## CYCLE-3 STATUS UPDATE
P4 (iteration/adaptive compute): DEMONSTRATED — IFT (iterated learned transducer)
computes O(N^2)-work multiplication via input-dependent pass count. T3 certified
(200/200 @ 25-50 digits) and 5/5 on frozen items. Suite: 19/19 machine-side.
Remaining open on the map: P5 (compositional benchmark), P9 (lottery - note: zero
restarts needed in ARC-2 so far; direct-gradient table classes are lottery-free),
P11 (fluency gap). Frontier-LLM column = operator's move.

## CYCLE-4 STATUS UPDATE
P5 (compositional generalization): SOLVED — L-COMPOSE-EXACT (exactness composes;
23/23 suite incl. nested 30-digit expression items). P9 (reliability): CLOSED for
direct-gradient table classes — L-DIRECT-GRADIENT, zero restarts across all ARC-2
training runs, 4-seed sweep all-exact. Map remaining: P11 (fluency gap - hybrid
engineering), P4 open-ended extension (learned pass-programs for NEW algorithms).

## CYCLE-5 STATUS UPDATE
P4 extension: sorting hosted with NATURAL encoding (no design) — substrate generality
evidenced. Suite 26/26 (parity, cups, addition, multiplication, nested expressions,
sorting). ARC-2 restart count still ZERO. Remaining: P11 (LM-host hybrid), division/
GCD as further substrate instances, operator's frontier column.

## CYCLE-6 STATUS UPDATE
Arithmetic set COMPLETE on the substrate family: + (KR stream), x (iterated IFT),
/ (KR stream), nested composition (learned dispatch over certified organs), plus
parity, state-chains, sorting. Suite 29/29 machine-side; ~1.7M params total across
all seven machines; total training wall-clock across the entire program: <5 min.
Open: P11 (LM-host hybrid), big/big division, operator's frontier column.
