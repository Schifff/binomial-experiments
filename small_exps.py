"""
Self-contained verification experiments (no external data):
  E1  linear complexity / period of binomial sequences and their sums (Berlekamp-Massey)
  E2  exact filter bias for the four filters vs. Monte Carlo
  E3  Monte-Carlo check of the exact sign-extraction bias and single-bit sensitivity
  E4  output distinctness of the extraction maps (N=16 exhaustive, N=32 sampled)
  E5  hardware cost model (full-adder cells) for sign vs. weight-bit extraction
Writes small_results.json.
"""
import json, math, os
from fractions import Fraction
from math import comb, log2
import numpy as np
from exact import sign_bias, sign_sensitivity, weightbit_bias, weightbit_sensitivity

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(2026)

def bm_int(bits):
    """Berlekamp-Massey over GF(2) for a Python list of bits; returns L."""
    n = len(bits); C = 1; B = 1; L = 0; m = 1
    for t in range(n):
        d = bits[t]
        c = C >> 1; i = 1
        while c and i <= L:
            if c & 1: d ^= bits[t - i]
            c >>= 1; i += 1
        if d:
            T = C; C ^= B << m
            if 2 * L <= t: L = t + 1 - L; B = T; m = 1
            else: m += 1
        else: m += 1
    return L

def binom_seq(k, n): return [comb(t, k) & 1 for t in range(n)]

def period(seq):
    n = len(seq)
    for p in range(1, n):
        if all(seq[i] == seq[i + p] for i in range(n - p)): return p
    return n

def E1():
    rows = []
    for k in [1, 2, 3, 5, 7, 8, 12, 31, 32, 33, 100, 127, 128, 200, 255]:
        s = binom_seq(k, 4 * (k + 2) + 512)
        L = bm_int(s); per = period(s[: 2 ** (int(log2(k)) + 2)])
        rows.append({"k": k, "wt": bin(k).count("1"), "LC": L, "pred_LC": k + 1, "period": per, "pred_period": 2 ** (int(log2(k)) + 1)})
    sums = []
    for K in [[3, 5], [7, 32], [1, 2, 4, 8, 16, 64], [255, 3], [12, 9, 100], [7, 11, 13, 200, 201]]:
        n = 4 * (max(K) + 2) + 512
        s = [0] * n
        for k in K:
            bk = binom_seq(k, n); s = [a ^ b for a, b in zip(s, bk)]
        sums.append({"K": K, "LC": bm_int(s), "pred_LC": max(K) + 1, "deg": max(bin(k).count("1") for k in K)})
    return {"single": rows, "sums": sums}

def E2():
    # exact P[Q=1] for monomial sums on disjoint supports; Monte Carlo on uniform inputs
    filt = {"Q1": [8] * 8, "Q2": [1] + [8] * 8, "Q3": [1] + [2] * 16, "Q4": [1] + list(range(2, 9))}
    out = {}
    for name, degs in filt.items():
        prod = Fraction(1)
        for d in degs: prod *= (1 - Fraction(2, 2 ** d))
        p1 = (1 - prod) / 2                          # P[Q = 1]
        nonlin_degs = [d for d in degs if d >= 2]
        prodn = Fraction(1)
        for d in nonlin_degs: prodn *= (1 - Fraction(2, 2 ** d))
        p_ne_lin = (1 - prodn) / 2                   # P[Q != x0] when a linear term is present
        # Monte Carlo
        M = 1 << 20; acc = np.zeros(M, dtype=np.uint8)
        for d in degs:
            acc ^= (rng.integers(0, 2, size=(M, d), dtype=np.uint8).all(axis=1)).astype(np.uint8)
        out[name] = {"degrees": degs, "P1_exact": float(p1), "P1_mc": float(acc.mean()),
                     "corr_to_x0_exact": float(1 - 2 * p_ne_lin) if 1 in degs else None,
                     "union_bound_P_ne_lin": float(sum(Fraction(1, 2 ** d) for d in nonlin_degs))}
    return out

def E3():
    out = {}
    for N in [16, 64, 256, 1024]:
        M = 1 << 16
        f = rng.integers(0, 2, size=(M, N)).astype(np.float64) * 2 - 1
        # Walsh spectrum via matrix product with Hadamard matrix (exact in float64 for |values| <= N)
        H = np.array([[1.0]])
        while H.shape[0] < N: H = np.block([[H, H], [H, -H]])
        F = (f @ H).astype(np.int64)
        z = (F < 0)
        bias_mc = float(z.mean() - 0.5)
        # sensitivity: flip a random coordinate, recompute sign of all coefficients
        idx = rng.integers(0, N, size=M)
        F2 = F - (2 * f[np.arange(M), idx][:, None] * H[idx]).astype(np.int64)
        sens_mc = float(((F2 < 0) != z).mean())
        w = (N - F) // 2
        wb = {str(j): float(((w >> j) & 1).mean() - 0.5) for j in range(int(log2(N)) + 1)}
        w2 = (N - F2) // 2
        wbs = {str(j): float((((w2 >> j) & 1) != ((w >> j) & 1)).mean()) for j in range(int(log2(N)) + 1)}
        out[str(N)] = {"sign_bias_exact": float(sign_bias(N)), "sign_bias_mc": bias_mc,
                       "sign_sens_exact": float(sign_sensitivity(N)), "sign_sens_mc": sens_mc,
                       "wbit_bias_exact": {str(j): float(weightbit_bias(N, j)) for j in range(int(log2(N)) + 1)},
                       "wbit_bias_mc": wb,
                       "wbit_sens_exact": {str(j): float(weightbit_sensitivity(N, j)) for j in range(int(log2(N)) + 1)},
                       "wbit_sens_mc": wbs, "samples": M}
    return out

def E4():
    out = {}
    # N = 16 exhaustive
    N = 16; H = np.array([[1.0]])
    while H.shape[0] < N: H = np.block([[H, H], [H, -H]])
    allf = ((np.arange(1 << N)[:, None] >> np.arange(N)) & 1).astype(np.float64) * 2 - 1
    F = (allf @ H).astype(np.int64); w = (N - F) // 2
    def distinct(rows):
        packed = np.packbits(rows.astype(np.uint8), axis=1)
        return len({bytes(r) for r in packed})
    out["N16"] = {"inputs": 1 << N, "sign": distinct(F < 0)}
    for j in range(int(log2(N)) + 1): out["N16"][f"bit{j}"] = distinct((w >> j) & 1)
    # N = 32 sampled: collisions among 2^20 random inputs
    N = 32; H = np.array([[1.0]])
    while H.shape[0] < N: H = np.block([[H, H], [H, -H]])
    M = 1 << 20
    f = rng.integers(0, 2, size=(M, N)).astype(np.float64) * 2 - 1
    F = (f @ H).astype(np.int64); w = (N - F) // 2
    out["N32_sampled"] = {"samples": M, "expected_collisions_if_uniform_32bit": M * (M - 1) / 2 / 2 ** 32}
    out["N32_sampled"]["sign_distinct"] = distinct(F < 0)
    for j in range(int(log2(N)) + 1): out["N32_sampled"][f"bit{j}_distinct"] = distinct((w >> j) & 1)
    return out

def E5():
    # full-adder cells for an unrolled N-point FWHT: stage s (1..b) has N/1 add/sub of width (s+1) bits (full sign) or (j+2) bits (mod 2^{j+2})
    out = {}
    for N in [64, 256, 1024]:
        b = int(log2(N))
        full = sum(N * (s + 1) for s in range(1, b + 1))
        row = {"N": N, "stages": b, "addsub_ops": N * b, "fa_cells_full_width": full}
        for j in [1, 2, 3]:
            row[f"fa_cells_mod2^{j+2}"] = sum(N * min(s + 1, j + 2) for s in range(1, b + 1))
        out[str(N)] = row
    return out

if __name__ == "__main__":
    res = {"E1": E1(), "E2": E2(), "E3": E3(), "E4": E4(), "E5": E5()}
    with open(os.path.join(HERE, "small_results.json"), "w") as f: json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
