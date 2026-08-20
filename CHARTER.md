# ARC-2: THE RESOURCE INVERSION PROGRAM

## Mission
Train machines in a 2GB-RAM / 1-CPU sandbox that OUTPERFORM frontier LLMs
(100s-of-GPU training runs) on a public suite of exact algorithmic-generalization
tasks — with victory independently verifiable by the operator pasting identical
test cases into any frontier chat model.

## Why this is winnable from a sandbox
Frontier models sit under the TC0 architectural ceiling: documented failures on
long-digit arithmetic, extreme-length parity/counting, deep nesting, long exact
state chains — failures that scale does NOT cure. Predecessor program (ssr_lab)
proved permutation-register machines walk through this ceiling with certified
exactness, and shipped the train->extract->verify->repair->certify pipeline.

## Target suite (public, operator-judgeable)
T1 ADDITION   : train <=8-digit, test 50-100 digit exact addition
T2 PARITY/COUNT: train <=64 items, test 1000+ exact
T3 MULTIPLICATION: train small, test large (stretch goal - nested carries)
T4 STATE-CHAINS: long transformation sequences (swap/move/set), exact final state
Success = 100% exact-match where frontier models measurably fail on the SAME items.

## Rules of engagement (inherited from ssr_lab)
No task-specific hardcoding: generic instruction sets only; programs learned by
SGD and/or repaired by the generic compiler; certification by length-invariance;
control-first for new model classes; multi-seed evidence; every negative logged.

## Inherited assets
Architecture family (ISA-PRAM/KR-ISA), compiler (surgery3/auto_compile*),
label-free gates, 26 laws — especially L-NEEDLE, L-SOFT-TARGETS, L-CRISP-HIERARCHY.

## Known hard parts (honest)
Carry propagation is data-dependent routing at every digit (feedback-class, our
weakest learnability zone); multiplication needs nested iteration (beyond current
1-pass machines - likely needs a multi-tape/multi-pass design = the new
architecture work). This is where the novelty budget goes.
