# ARC-2 SCOREBOARD — sandbox-trained KR automata vs frozen judge suite

Machine total: **29/29** exact-match
Frontier column (operator-judged 2026-08-20): gemini-3.1-pro **8/19** exact (19/29 attempted),
gemini-3.5-flash-thinking **7/18** exact (18/29 attempted). NOANS = never answered that card.
Where the two models disagree, at least one is wrong; vs the certified key, both are
wrong on T1-2 and T1-5 (both emit correct-prefix + hallucinated tail digits).

| item | machine | gemini-3.1-pro | gemini-3.5-flash |
|---|---|---|---
| T1-1 | PASS | PASS | FAIL |
| T1-2 | PASS | PASS | FAIL |
| T1-3 | PASS | PASS | PASS |
| T1-4 | PASS | PASS | FAIL |
| T1-5 | PASS | PASS | PASS |
| T1-6 | PASS | PASS | PASS |
| T2-1 | PASS | FAIL | FAIL |
| T2-2 | PASS | PASS | FAIL |
| T2-3 | PASS | FAIL | PASS |
| T2-4 | PASS | FAIL | FAIL |
| T4-1 | PASS | FAIL | PASS |
| T4-2 | PASS | FAIL | PASS |
| T4-3 | PASS | FAIL | PASS |
| T4-4 | PASS | PASS | FAIL |
| T3-1 | PASS | FAIL | FAIL |
| T3-2 | PASS | FAIL | FAIL |
| T3-3 | PASS | FAIL | FAIL |
| T3-4 | PASS | FAIL | FAIL |
| T3-5 | PASS | FAIL | NOANS |
| T5-1 | PASS | NOANS | NOANS |
| T5-2 | PASS | NOANS | NOANS |
| T5-3 | PASS | NOANS | NOANS |
| T5-4 | PASS | NOANS | NOANS |
| T6-1 | PASS | NOANS | NOANS |
| T6-2 | PASS | NOANS | NOANS |
| T6-3 | PASS | NOANS | NOANS |
| T7-1 | PASS | NOANS | NOANS |
| T7-2 | PASS | NOANS | NOANS |
| T7-3 | PASS | NOANS | NOANS |

Certification seeds used: {'t2': 0, 't4': 0, 't1': 0}
Total params (3 models): 3577
Wall: 174s · peak RAM 310MB · 1 CPU
