# ARC-2 — The Resource Inversion Program

Autonomous research program (successor to ssr_lab): certified register machines
trained in a 2GB-RAM/1-CPU sandbox that outperform frontier LLMs on exact
algorithmic tasks, and a certified hybrid mounting them inside a token-prediction
host. **Status: CLOSED at cycle 7 — machine 29/29 vs frontier 8/19, 7/18.**

| artifact | content |
|---|---|
| `PAPER_ARC2.md` | final manuscript (results, laws, honest ledger) |
| `CHARTER.md` | mission and rules of engagement |
| `log.md` / `log.jsonl` | cycle log / machine-written RESULT records |
| `SCOREBOARD.md`, `JUDGE_CARDS.md`, `FRONTIER_RESULTS.md` | frozen suite + frontier judging |
| `PROBLEM_MAP.md` | 12-problem map with per-cycle status |
| `paper.md`, `research_log.md` | predecessor program (ssr_lab, 41 cycles) |
| `t*.py`, `phase*.py`, `cycle7*.py` | one seeded self-contained script per cycle |

Reproduce any cycle: `python3 <script>.py` (torch CPU + numpy only).
