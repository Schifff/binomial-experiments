# Verify primitivity of the binary polynomials used for the LFSR experiments.
# Polynomials are given as (degree, taps) meaning x^r + sum_{i in taps} x^i.
from math import gcd
def polymulmod(a, b, mod, r):
    res = 0
    while b:
        if b & 1: res ^= a
        b >>= 1; a <<= 1
        if a >> r & 1: a ^= mod
    return res
def polypowmod(base, e, mod, r):
    res = 1
    while e:
        if e & 1: res = polymulmod(res, base, mod, r)
        base = polymulmod(base, base, mod, r); e >>= 1
    return res
def factor(n):
    f = []; d = 2
    while d * d <= n:
        while n % d == 0: f.append(d); n //= d
        d += 1
    if n > 1: f.append(n)
    return sorted(set(f))
FERMAT = [3, 5, 17, 257, 65537, 641, 6700417, 274177, 67280421310721, 59649589127497217, 5704689200685129054721]
def is_primitive(r, taps):
    mod = (1 << r) | sum(1 << t for t in taps)
    order = (1 << r) - 1
    if r == 256:
        primes = FERMAT
        assert all(order % p == 0 for p in primes)
        prod = 1
        for p in primes: prod *= p
        assert prod == order, "factorization of 2^256-1 incomplete"
    elif r == 64:
        primes = [3, 5, 17, 257, 641, 65537, 6700417]
        prod = 1
        for p in primes: prod *= p
        assert prod == order
    else:
        primes = factor(order)
    if polypowmod(2, order, mod, r) != 1: return False
    return all(polypowmod(2, order // p, mod, r) != 1 for p in primes)
cands = {16: [0, 2, 3, 5], 20: [0, 3], 24: [0, 1, 2, 7], 32: [0, 1, 2, 22], 64: [0, 1, 3, 4], 256: [0, 2, 5, 10]}
for r, taps in cands.items():
    print(f"r={r:3d} p(x)=x^{r}+" + "+".join(f"x^{t}" if t else "1" for t in sorted(taps, reverse=True)), "primitive:", is_primitive(r, taps))
