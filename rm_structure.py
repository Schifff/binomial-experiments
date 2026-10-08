# Reed-Muller structure of the extraction maps: for random blocks f in F_2^N, compute
# (a) the algebraic degree, as a Boolean function of the spectral index u, of z^(j)_u = bit_j(w_u) and of the sign bit,
# (b) the GF(2) rank of a matrix whose rows are many output blocks (dimension of the span).
import numpy as np, json, sys
from math import comb, log2
rng = np.random.default_rng(7)
def hadamard(N):
    H = np.array([[1.0]])
    while H.shape[0] < N: H = np.block([[H, H], [H, -H]])
    return H
def anf_degree(vec, b):
    # Moebius transform over u in F_2^b; vec indexed by integer u
    a = vec.copy().astype(np.uint8)
    for i in range(b):
        step = 1 << i
        for u in range(1 << b):
            if u & step: a[u] ^= a[u ^ step]
    nz = np.flatnonzero(a)
    return max((bin(int(u)).count("1") for u in nz), default=-1)
def gf2_rank(rows_bits):
    # rows_bits: list of python ints
    rank = 0; rows = list(rows_bits); pivots = {}
    for r in rows:
        x = r
        while x:
            h = x.bit_length() - 1
            if h in pivots: x ^= pivots[h]
            else: pivots[h] = x; rank += 1; break
    return rank
out = {}
for N in [16, 32, 64, 256]:
    b = int(log2(N)); H = hadamard(N)
    M = 4000 if N <= 64 else 3000
    f = rng.integers(0, 2, size=(M, N)).astype(np.float64) * 2 - 1
    F = (f @ H).astype(np.int64); w = (N - F) // 2
    res = {}
    maps = {f"bit{j}": ((w >> j) & 1) for j in range(b + 1)}
    maps["sign"] = (F < 0).astype(np.int64)
    for name, Z in maps.items():
        degs = [anf_degree(Z[i], b) for i in range(min(M, 300))]
        rows = [int("".join(map(str, Z[i].tolist()))[::-1], 2) for i in range(M)]
        # rank of span including the all-ones vector (affine span)
        rk = gf2_rank(rows)
        res[name] = {"max_deg_in_u": int(max(degs)), "rank_of_span": rk}
    # RM dimensions for reference
    res["RM_dims"] = {str(r): int(sum(comb(b, i) for i in range(r + 1))) for r in range(b + 1)}
    out[str(N)] = res
    print(N, json.dumps(res), flush=True)
json.dump(out, open("rm_structure.json", "w"), indent=1)
