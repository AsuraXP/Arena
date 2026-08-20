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
## Cycle 7 (P11 M1): division organ MOUNTED inside a token-prediction host — CERTIFIED.
- Host: tiny RoPE-TF (d=64, 2L, ~72k params). Organ: frozen div_t7.pt tables (mounted
  inference, remount re-certified 300/300 incl. 80-150-digit). Mixed streams: order-2
  Markov text + marked division spans. Arms: TF-FULL (plain LM), TF-MASK (fluency
  control), HYBRID (host-text + learned gate + organ answers).
- Span exact-match (greedy rollout), train operands <=8 digits:
    nd=8:   TF 10%   HYBRID 100%
    nd=40:  TF 2.5%  HYBRID 100%
    nd=100: TF 0%    HYBRID 100%
    nd=150: TF 2.5%  HYBRID 100%   (19x training length)
- Fluency preserved: hybrid text CE <= both TF arms at every length.
- NEGATIVE (informative): per-token gate arbitration degrades with answer length
  (100/62.5/37.5/7.5% exact; gate acc 75.9%, FN 830 @150) — the ROUTER hits TC0-counting,
  not the organ. Fix = protocol: organ self-terminates (emits DEND), gate decides ONCE
  at the first answer slot -> 0 first-slot misses at all lengths.
- New law L-GATE-EXTENT: span routing must be self-terminating; per-position neural
  gates cannot hold answer extent under length extrapolation, boundary-local routing
  with organ-owned termination is exact by construction.
- Forensics: initial mount violated L-DETERMINISM(a) (divisor register updated before
  remainder read from OLD state) — caught instantly by the P0 oracle gate at 22/300,
  diagnosed from the law index, fixed. Law index remains a working diagnostic table.
- Wall 263s, peak RAM 937MB (<2GB charter), 2 CPUs, restarts: ZERO (seed 0 first try).
- Next (M2): marker-free routing — detect unlabeled spans in raw mixed text via a
  certified KR-class gate machine (the gate problem is now precisely a table-learnable
  transducer); mount adder/multiplier organs; big/big division via IFT (stretch).
## Cycle 7 M2: certified KR router + multi-organ mount — CERTIFIED.
- Router: 5-state table machine (TEXT/div-q/div-ans/add-q/add-ans), learned by direct
  supervision, crystallized deterministically (L-DIRECT-GRADIENT; seeds 0,1,2 all
  zero-error). CERTIFIED 65230/65230 routing decisions incl. 150-digit streams.
- Organs mounted: division (frozen div_t7.pt, remount 300/300) + NEW streaming adder
  (LSB pair tables, trained <=4 digits in seconds, certified 200/200 at 40-100-digit
  carry chains — C2's L-ENCODING result reproduced in streaming mount form).
- End-to-end (marker-free in-band syntax "/d=" and "+pairs=", distractor digit runs in
  text): HY-KR = 100% exact on BOTH organs at operand widths 8/40/100/150 (19x train);
  plain host (TF-FULL) = 0% at every width; text CE parity held (2.85-3.52 band both arms).
- ABLATION (the point of M2): neural-probe gate = 100% in-dist but 0% at >=40 (first
  boundary decision collapses under length extrapolation). Certified table router holds
  by construction. L-GATE-EXTENT extended: boundary decisions, not just extent, must be
  machine-owned under extrapolation.
- Defects found & fixed this milestone (all caught by oracle/audit gates, zero training
  restarts): (1) mount violated L-DETERMINISM(a) old-state indexing (22/300 instantly);
  (2) remainder>=10 encoded as single token = collision with span-openers (L-ENCODING at
  mount time; fixed: %02d two-token remainder); (3) data-gen cls length bug (distractor
  run shorter than drawn length -> global +1 class shift; found by index-collision audit
  showing 6 phantom 'ambiguous' cells); (4) certification harness off-by-one (gate was
  correct; harness compared decision-after-k+1 vs class-k+1) + Ts-table audit added +
  cert condition wrongly folded the control arm's misses. Audit protocol now covers BOTH
  tables (Th outputs + Ts transitions) permanently.
- Wall 360s, peak RAM 782MB (<2GB charter), 2 CPUs. Artifacts: cycle7b_krgate.py,
  krgate_s0.pt, addorgan_s0.pt, hybrid_host_s0.pt (M1).
- Next: M3 — mount multiplier organ (ift_t3.pt, iterated passes) for O(N^2)-work spans;
  big/big division via IFT compare-subtract (stretch); 4-op calculator = full T5-style
  dispatch inside a token host. Then P11 closes.
## Cycle 7 M3: ITERATED organ mounted — three-organ dispatch CERTIFIED.
- IFT multiplication organ (ift_t3.pt, trained <=6 digit operands in C3) remounted from
  frozen tables: 320/320 exact across 1-50-digit operands incl. 20x 40x40 judge-scale
  products. The organ's internal O(N)-pass fixed-point computation runs inside a
  single-pass span surface behind the router.
- KR gate extended to 7 states / 4 classes (text + {div,mul,add}-q/-ans): certified
  83695/83695 routing decisions, seeds 0,1,2 zero-error, dual-table audit clean.
- End-to-end: HY-KR = 100% exact on ALL THREE organs at every width —
  div 8..150 (19x), add 8..150 (19x), mul 8..40 (5x, incl 40x40 products = judge
  scale T3) — while plain host TF-FULL = 0% at every width including in-distribution.
- Neural-probe control: 100% in-dist, 0% at >=2x operand width — three-way dispatch
  breaks the probe even earlier than M2's two-way. Certified table router unaffected.
- Honest negative: HOST-M text CE drifts up at long contexts (2.8 -> 4.0-4.5 vs
  TF-FULL 2.5-3.0) when operand widths exceed training — the masked host's TEXT
  quality suffers extrapolating over long span contexts even though hybrid ANSWER
  exactness is unaffected (structural, organ-owned). Logged for M4: host context
  curriculum or span-summarized context.
- Defects found & fixed (caught by gates, zero training restarts): double-argmax on
  already-hard M2 tables; sync()-after-answers cls overlength (caught by hardened
  oracle instantly); seed-sweep variable shadowed an organ table (H2); make_eval cls
  built from formula instead of segment lengths (assert caught).
- Wall 564s, peak RAM 806MB (<2GB charter). Artifacts: cycle7c_multorgan.py,
  krgate3_s0.pt. Suite implication: T1+T3+T5+T7-style spans all routable in one host.
- Remaining for P11 closure: big/big division via IFT compare-subtract (stretch),
  HOST-M text-CE drift fix (M4), multi-seed host, real-text corpus trial.
