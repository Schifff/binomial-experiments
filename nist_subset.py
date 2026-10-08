"""
Subset of NIST SP 800-22 Rev. 1a statistical tests (Bassham et al., 2010), implemented
with numpy/scipy.  Ten of the fifteen tests are implemented: Frequency, Block Frequency,
Cumulative Sums (forward/backward), Runs, Longest Run of Ones, Binary Matrix Rank,
Discrete Fourier Transform, Serial, Approximate Entropy, Linear Complexity.
Not implemented: Non-overlapping/Overlapping Template, Maurer's Universal, Random
Excursions (+Variant).  Test parameters follow the SP 800-22 defaults for n >= 10^6.
"""
import math, subprocess, os
import numpy as np
from scipy.special import erfc, gammaincc
from scipy.stats import norm

GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen.exe")

def load_bits(path, nbits):
    raw = np.fromfile(path, dtype=np.uint8, count=(nbits + 7) // 8)
    return np.unpackbits(raw, bitorder="little")[:nbits]

def frequency(b):
    n = b.size; s = 2.0 * b.sum() - n
    return erfc(abs(s) / math.sqrt(2 * n))

def block_frequency(b, M=128):
    N = b.size // M
    pi = b[:N * M].reshape(N, M).mean(axis=1)
    chi = 4.0 * M * np.sum((pi - 0.5) ** 2)
    return gammaincc(N / 2.0, chi / 2.0)

def runs(b):
    n = b.size; pi = b.mean()
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n): return 0.0
    v = 1 + int(np.count_nonzero(b[1:] != b[:-1]))
    return erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi)))

def longest_run(b):
    n = b.size
    if n >= 750000: M, K, cats, pi = 10000, 6, [10, 11, 12, 13, 14, 15], [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    elif n >= 6272: M, K, cats, pi = 128, 5, [4, 5, 6, 7, 8], [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else: M, K, cats, pi = 8, 3, [1, 2, 3], [0.2148, 0.3672, 0.2305, 0.1875]
    N = n // M; blocks = b[:N * M].reshape(N, M)
    nu = np.zeros(K + 1)
    for blk in blocks:
        z = np.flatnonzero(np.concatenate(([0], blk, [0])) == 0)
        L = int((np.diff(z) - 1).max())
        nu[min(max(L - cats[0], 0), K)] += 1   # categories are consecutive: <=cats[0], ..., >=cats[-1]+1
    chi = np.sum((nu - N * np.array(pi)) ** 2 / (N * np.array(pi)))
    return gammaincc(K / 2.0, chi / 2.0)

def cusum(b, forward=True):
    n = b.size; x = 2 * b.astype(np.int64) - 1
    if not forward: x = x[::-1]
    S = np.cumsum(x); z = int(np.abs(S).max())
    k1 = np.arange(int((-n / z + 1) // 4), int((n / z - 1) // 4) + 1)
    t1 = np.sum(norm.cdf((4 * k1 + 1) * z / math.sqrt(n)) - norm.cdf((4 * k1 - 1) * z / math.sqrt(n)))
    k2 = np.arange(int((-n / z - 3) // 4), int((n / z - 1) // 4) + 1)
    t2 = np.sum(norm.cdf((4 * k2 + 3) * z / math.sqrt(n)) - norm.cdf((4 * k2 + 1) * z / math.sqrt(n)))
    return 1.0 - t1 + t2

def _rank_gf2(rows):
    rank = 0; rows = list(rows)
    for bit in range(31, -1, -1):
        piv = next((i for i in range(rank, len(rows)) if (rows[i] >> bit) & 1), None)
        if piv is None: continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and (rows[i] >> bit) & 1: rows[i] ^= rows[rank]
        rank += 1
    return rank

def rank_test(b, M=32, Q=32):
    N = b.size // (M * Q)
    mats = b[:N * M * Q].reshape(N, M, Q)
    weights = (1 << np.arange(Q, dtype=np.int64))
    rows_all = (mats.astype(np.int64) * weights).sum(axis=2)
    F = np.zeros(3)
    for rws in rows_all:
        r = _rank_gf2(rws.tolist())
        F[0 if r == M else 1 if r == M - 1 else 2] += 1
    p = np.array([0.2888, 0.5776, 0.1336])
    chi = np.sum((F - N * p) ** 2 / (N * p))
    return math.exp(-chi / 2.0)

def dft(b):
    n = b.size; x = 2.0 * b - 1.0
    mod = np.abs(np.fft.rfft(x))[: n // 2]
    T = math.sqrt(math.log(1 / 0.05) * n); N0 = 0.95 * n / 2; N1 = np.count_nonzero(mod < T)
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    return erfc(abs(d) / math.sqrt(2))

def _psi2(b, m):
    if m <= 0: return 0.0
    n = b.size; v = np.zeros(n, dtype=np.int64)
    for j in range(m): v = (v << 1) | np.roll(b, -j).astype(np.int64)
    c = np.bincount(v, minlength=1 << m).astype(np.float64)
    return (2 ** m / n) * np.sum(c * c) - n

def serial(b, m=16):
    p0, p1, p2 = _psi2(b, m), _psi2(b, m - 1), _psi2(b, m - 2)
    d1 = p0 - p1; d2 = p0 - 2 * p1 + p2
    return gammaincc(2 ** (m - 2), d1 / 2.0), gammaincc(2 ** (m - 3), d2 / 2.0)

def _phi(b, m):
    n = b.size; v = np.zeros(n, dtype=np.int64)
    for j in range(m): v = (v << 1) | np.roll(b, -j).astype(np.int64)
    c = np.bincount(v, minlength=1 << m).astype(np.float64); c = c[c > 0] / n
    return float(np.sum(c * np.log(c)))

def approximate_entropy(b, m=10):
    n = b.size
    chi = 2.0 * n * (math.log(2) - (_phi(b, m) - _phi(b, m + 1)))
    return gammaincc(2 ** (m - 1), chi / 2.0)

def linear_complexity(path, nbits, M=500):
    out = subprocess.run([GEN, "bmblocks", "--in", path, "--nbits", str(nbits), "--M", str(M)], capture_output=True, text=True, check=True).stdout
    L = np.array([int(t) for t in out.split()], dtype=np.float64); N = L.size
    mu = M / 2.0 + (9 + (-1) ** (M + 1)) / 36.0 - (M / 3.0 + 2 / 9.0) / 2 ** M
    T = (-1) ** M * (L - mu) + 2 / 9.0
    edges = [-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf]
    nu = np.histogram(T, bins=edges)[0].astype(np.float64)
    pi = np.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
    chi = np.sum((nu - N * pi) ** 2 / (N * pi))
    return gammaincc(3.0, chi / 2.0)

def battery(path, nbits):
    b = load_bits(path, nbits)
    res = {}
    res["frequency"] = frequency(b)
    res["block_frequency"] = block_frequency(b)
    res["cusum_fwd"] = cusum(b, True)
    res["cusum_bwd"] = cusum(b, False)
    res["runs"] = runs(b)
    res["longest_run"] = longest_run(b)
    res["rank"] = rank_test(b)
    res["dft"] = dft(b)
    s1, s2 = serial(b); res["serial_1"] = s1; res["serial_2"] = s2
    res["apen"] = approximate_entropy(b)
    res["linear_complexity"] = linear_complexity(path, nbits)
    return {k: float(v) for k, v in res.items()}

TEST_NAMES = ["frequency", "block_frequency", "cusum_fwd", "cusum_bwd", "runs", "longest_run", "rank", "dft", "serial_1", "serial_2", "apen", "linear_complexity"]
