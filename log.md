# ARC-2 LOG
## Cycle 1: charter, problem map (12 problems), frozen suite (14 items) + judge cards.
## Cycle 2: KR-automaton solver (generic basis: id/const/shift/transpositions,
learned dispatch, contextual output, hard inference).
- All 3 tasks certified seed-0 first try: parity@2000, cups@350, addition@120 digits.
- JUDGE SUITE: 14/14 exact-match. ~3.6k params total, 174s, 1 CPU.
- New law L-ENCODING: hardness class is representation-relative — carry is
  feedback-class MSB-first but permutation-reset-class LSB-first. Choosing the
  encoding IS part of the architecture.
- Next: T3 multiplication (true nested-iteration frontier, P4); frontier-LLM column.
## Cycle 3: T3 multiplication SOLVED — Iterated Factored Transducer (IFT).
- Mechanism: one learned FST pass (factored registers: mult/carry/aprev/first-pair;
  factored output heads; generic FST primitives) iterated on its own tape to fixed
  point. Escapes the regular class by ITERATION — P4 (adaptive compute) demonstrated.
- Oracle-first protocol: encoding verified 500/500 BEFORE any learning.
- Two determinism bugs found+fixed (index-collision audits): L-DETERMINISM laws:
  (a) heads must index OLD state where position roles differ; (b) don't-care outputs
  must be masked from loss, not supervised to arbitrary values.
- CERTIFIED: 200/200 exact 25-50 digit products (trained <=6 digits, ~8x length,
  O(N^2) work via O(N) passes). Judge items T3-1..5 (35x35..40x40): 5/5.
- SUITE TOTAL: 19/19. Frontier column awaits operator.
## Cycle 4: P5 compositional generalization SOLVED + P9 formalized.
- T5 "certified compositional calculator": frozen certified adder+multiplier under a
  LEARNED dispatch controller over a generic value stack. Trained on single-op,
  1-3-digit expressions only -> 200/200 EXACT on novel 3-5-op expressions with
  20-40-digit operands. Judge T5-1..4: 4/4. SUITE: 23/23.
  Law L-COMPOSE-EXACT: exactness composes — certified components have zero error to
  multiply across depth; compositional generalization is free once parts are exact.
- P9 seed sweep: t1 across 4 total seeds, t2/t3/t4/t5 across all attempts: EVERY seed
  certified on first training run. Law L-DIRECT-GRADIENT (formalized): when every
  learned discrete decision receives direct per-cell supervision, crystallization is
  DETERMINISTIC — the ssr_lab lottery was a property of indirectly-supervised
  channels, not of discrete learning itself. ARC-2 restart count to date: ZERO.
## Cycle 5: T6 sorting SOLVED — paradigm-generality test passed.
- Learned bubble transducer: state=held value, output=SELECT{token,held} (structural
  value routing — values never enter continuous space), iterate to tape FIXPOINT
  (generic halt). Trained len<=8 -> 200/200 exact at 100-250 elems. Judge 3/3.
- SUITE: 26/26. Encoding was NATURAL (raw list) — first algorithm hosted with zero
  encoding design, weakening the "designer did the work" objection.
- Baseline note (honest): same-budget micro-TF failed in-dist (undertrained; not the
  fair bar). External bar: NeurIPS'20 NEE — vanilla TF <10% @100 elems; specialized
  fixes ~100 max. L-DIRECT-GRADIENT held again: seed 0, first attempt, zero restarts.
- New law L-STRUCTURAL-ROUTING: routing values by SELECTION (copy-token/copy-state)
  rather than embedding them removes the value dimension from the learning problem
  entirely — generalization over the value domain is free by construction.
## Cycle 6: T7 division SOLVED — arithmetic set complete (+,x,/, plus parity/sort/compose).
- Streaming long-division transducer (registers: divisor set-once, remainder; MSB
  natural encoding). Trained <=7 digits in 5.9s -> 200/200 exact at 80-150 digits.
- Judge T7 3/3. SUITE: 29/29 across seven families. Restarts to date: ZERO.
- Bug forensics: failure rate 15/200 == d=2 prior (1/11) -> instant diagnosis of an
  old-state snapshot violation in the trace generator. The law index now functions
  as a numerical diagnostic table.
- Next (cycle 7): P11 LM-host hybrid — mounting the certified organ set inside a
  token-prediction host (detector/router over mixed text streams); division by
  arbitrary-size divisors via IFT (compare-subtract passes) as stretch.
