# Exact combinatorial quantities for the Walsh-Hadamard post-processing stage.
from fractions import Fraction
from math import comb, sqrt, pi, log2
def sign_bias(N):            # P[w > N/2] - 1/2 = -1/2 * C(N,N/2) 2^-N  (output 1 iff hat f_u < 0 iff w > N/2)
    return Fraction(-1, 2) * Fraction(comb(N, N // 2), 2 ** N)
def sign_sensitivity(N):     # P[single-bit flip changes sign bit] = C(N,N/2) 2^-N
    return Fraction(comb(N, N // 2), 2 ** N)
def weightbit_bias(N, j):    # P[bit_j(w)=1] - 1/2, w ~ Bin(N,1/2)
    tot = sum(comb(N, w) for w in range(N + 1) if (w >> j) & 1)
    return Fraction(tot, 2 ** N) - Fraction(1, 2)
def weightbit_sensitivity(N, j):
    # flip changes w by +-1; bit j changes iff w mod 2^{j+1} crosses 2^j-1 <-> 2^j or 2^{j+1}-1 <-> 0
    M = 1 << (j + 1)
    p = Fraction(0)
    for w in range(N + 1):
        pw = Fraction(comb(N, w), 2 ** N)
        # flipped coordinate is 1 with prob w/N (then w-1), else 0 (then w+1)
        up = Fraction(N - w, N); dn = Fraction(w, N)
        if w < N and (((w + 1) >> j) & 1) != ((w >> j) & 1): p += pw * up
        if w > 0 and (((w - 1) >> j) & 1) != ((w >> j) & 1): p += pw * dn
    return p
print("N, sign bias eps_N, |eps| approx (2*pi*N)^-1/2, sensitivity, samples for |z|=3 monobit (9/(4 eps^2))")
for N in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    e = sign_bias(N); s = sign_sensitivity(N)
    print(f"{N:5d} {float(e):+.6e} {1/sqrt(2*pi*N):.6e} {float(s):.6e} {9/(4*float(e)**2):.3e}")
print()
print("weight-bit biases: rows N, cols j")
for N in [64, 256, 1024, 4096]:
    b = int(log2(N))
    row = []
    for j in range(b + 1):
        x = float(weightbit_bias(N, j))
        row.append(f"j={j}:{x:+.3e}")
    print(N, " ".join(row))
print()
print("weight-bit sensitivities: rows N, cols j")
for N in [64, 256, 1024]:
    b = int(log2(N))
    print(N, " ".join(f"j={j}:{float(weightbit_sensitivity(N,j)):.4f}" for j in range(b + 1)))
