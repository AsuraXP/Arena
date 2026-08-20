"""Synthetic reasoning task generators. All tasks are per-token state-tracking:
model sees token stream, must predict the running 'state' at every position.
Parity and S5 are NC^1-hard-ish state tracking; TC^0 models fail to length-generalize.
"""
import itertools, torch

def gen_parity(batch, length, g=None):
    # tokens in {0,1}; target_t = XOR of tokens[0..t]
    x = torch.randint(0, 2, (batch, length), generator=g)
    y = torch.cumsum(x, dim=1) % 2
    return x, y, 2, 2  # vocab_in, vocab_out

def gen_mod5(batch, length, g=None):
    # tokens in {0..4}; target_t = sum mod 5 (abelian group Z5)
    x = torch.randint(0, 5, (batch, length), generator=g)
    y = torch.cumsum(x, dim=1) % 5
    return x, y, 5, 5

# ---- S5 permutation composition (NC^1-complete word problem) ----
_PERMS = list(itertools.permutations(range(5)))          # 120 elements
_P2I = {p: i for i, p in enumerate(_PERMS)}
def _compose(a, b):  # apply a after b
    return tuple(a[b[i]] for i in range(5))
# generators: a transposition and a 5-cycle generate S5
_GENS = [(1,0,2,3,4), (1,2,3,4,0), (0,1,2,3,4), (0,2,1,3,4), (4,3,2,1,0)]

_COMP_TABLE = None
def _table():
    global _COMP_TABLE
    if _COMP_TABLE is None:
        t = torch.empty(len(_GENS), 120, dtype=torch.long)
        for gi, gp in enumerate(_GENS):
            for si, sp in enumerate(_PERMS):
                t[gi, si] = _P2I[_compose(gp, sp)]
        _COMP_TABLE = t
    return _COMP_TABLE

def gen_s5(batch, length, g=None):
    # tokens are generator indices; target_t = index of composed permutation in S5
    tab = _table()
    x = torch.randint(0, len(_GENS), (batch, length), generator=g)
    y = torch.empty(batch, length, dtype=torch.long)
    state = torch.full((batch,), _P2I[(0,1,2,3,4)], dtype=torch.long)
    for t in range(length):
        state = tab[x[:, t], state]
        y[:, t] = state
    return x, y, len(_GENS), 120

TASKS = {"parity": gen_parity, "mod5": gen_mod5, "s5": gen_s5}
