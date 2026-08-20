"""ARC-2 cycle 1: frozen test suite + operator judge cards + answer key."""
import json, random
rng = random.Random(20260818)
cards, key = [], {}

def add(tid, prompt, answer):
    cards.append((tid, prompt)); key[tid] = str(answer)

# T1 addition: 50-100 digit
for i in range(6):
    nd = rng.choice([50, 70, 100])
    a = rng.randrange(10**(nd-1), 10**nd); b = rng.randrange(10**(nd-1), 10**nd)
    add(f"T1-{i+1}", f"Compute exactly, digits only: {a} + {b} = ?", a + b)
# T2 parity / counting at length 1000+
for i in range(4):
    n = rng.choice([1000, 1500])
    bits = [rng.randrange(2) for _ in range(n)]
    add(f"T2-{i+1}", "Below is a bit sequence. Reply with ONE word, 'even' or "
        f"'odd': the count of 1s is even or odd?\n{''.join(map(str, bits))}",
        "even" if sum(bits) % 2 == 0 else "odd")
# T4 state chains: cup shuffle, 300 ops
for i in range(4):
    pos = list(range(5)); ops = []
    for _ in range(300):
        a, b = rng.sample(range(5), 2); ops.append(f"swap {a} {b}")
        pos[a], pos[b] = pos[b], pos[a]
    ball = rng.randrange(5)
    add(f"T4-{i+1}", "5 cups at positions 0-4. A ball starts under the cup at "
        f"position {ball}. Apply these {len(ops)} swaps of positions in order, "
        "then answer with a single digit: where is the ball?\n" + "; ".join(ops),
        pos.index(ball) if False else [p for p in range(5)][pos.index(ball)] if False else pos.index(ball))
# note: ball tracking = find where initial position 'ball' content moved:
# recompute properly below (positions permuted)
# --- fix T4 answers by explicit simulation ---
cards2, key2 = [], {}
rng = random.Random(20260818)
for tid, p in cards:
    if not tid.startswith("T4"): cards2.append((tid, p)); key2[tid] = key[tid]
for i in range(4):
    slots = list(range(5))            # slots[pos] = which original cup is here
    ops = []
    r2 = random.Random(999 + i)
    ball_cup_pos = r2.randrange(5)    # ball under cup currently at this position
    ball_cup = ball_cup_pos           # identify cup by its start position
    for _ in range(300):
        a, b = r2.sample(range(5), 2); ops.append(f"swap {a} {b}")
        slots[a], slots[b] = slots[b], slots[a]
    ans = slots.index(ball_cup)
    prompt = ("5 cups sit at positions 0-4. A ball is under the cup at position "
              f"{ball_cup_pos}. The swaps below exchange the CUPS at the two "
              "positions (ball moves with its cup). After all 300 swaps, at "
              "which position is the ball? Answer with one digit.\n"
              + "; ".join(ops))
    cards2.append((f"T4-{i+1}", prompt)); key2[f"T4-{i+1}"] = str(ans)
cards, key = cards2, key2

with open("JUDGE_CARDS.md", "w") as f:
    f.write("# ARC-2 JUDGE CARDS - paste each prompt into any frontier chatbot;"
            " record exact answers.\n\n")
    for tid, p in cards:
        f.write(f"## {tid}\n```\n{p}\n```\n\n")
json.dump(key, open("answer_key.json", "w"), indent=1)
print(f"suite frozen: {len(cards)} items -> JUDGE_CARDS.md + answer_key.json")
