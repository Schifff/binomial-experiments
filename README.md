# Binomial filters and Walsh–Hadamard bit extraction: experiments

Reference implementation and test scripts for the paper *Binomial Filters and
Walsh–Hadamard Bit Extraction for Keystream Generation: Exact Linear Complexity,
Reed–Muller Structure, and Sign Bias*. Every number in the paper's tables is
produced by the scripts in this repository.

## Contents

| File | Purpose |
|---|---|
| `gen.c` | Keystream generator (binomial filter over a 256-bit LFSR or counter, FWHT post-processing with sign or bit-*j* extraction), ChaCha20 baseline (RFC 8439), bit-packed Berlekamp–Massey |
| `nist_subset.py` | Ten of the fifteen NIST SP 800-22 tests (frequency, block frequency, cumulative sums, runs, longest run, rank, DFT, serial, approximate entropy, linear complexity) |
| `exact.py` | Exact bias and sensitivity of sign and bit-*j* extraction under uniform blocks |
| `primitivity.py` | Primitivity check of the LFSR feedback polynomials, including *x*^256 + *x*^10 + *x*^5 + *x*^2 + 1 |
| `small_exps.py` | Linear complexity of binomial sequences (Berlekamp–Massey), filter statistics, Monte Carlo checks, output distinctness, hardware cost model → `small_results.json` |
| `rm_structure.py` | Algebraic degree in the spectral index and GF(2) rank of extracted blocks → `rm_structure.json` |
| `run_all.py` | Full grid: keystream generation, statistical battery, block ranks, counter mode, throughput → `results.json` |
| `counter_straddle.py` | Counter mode with every monomial straddling the block boundary, where the bound 2^\|K\| is attained → `counter_straddle.json` |
| `*.json` | Raw results used in the paper |

## Requirements

- GCC (tested with 13.2, MinGW-W64) or any C99 compiler
- Python 3.12+ with `numpy` and `scipy`

## Build and run

```
gcc -O2 -std=gnu11 -o gen.exe gen.c
python primitivity.py
python small_exps.py
python rm_structure.py
python run_all.py
```

`run_all.py` writes about 80 MB of keystream files to `out/` and takes roughly
25 minutes on one core of an Intel i7-12700H. The other scripts finish in a few
minutes.

## Generator usage

```
gen keystream --drive {counter|lfsr} --r 256 --taps 0,2,5,10 \
              --K "0;1-2-3-4-5-6-7-8" --post {none|sign|bit:J} --N 256 \
              --nbits 16777216 --seed 1 --out file.bin
gen chacha20  --nbits NB --seed S --out file.bin
gen lfsr      --r R --taps ... --nbits NB --seed S --out file.bin
gen bm        --in file.bin --nbits NB
gen bmblocks  --in file.bin --nbits NB --M 500
```

Monomials in `--K` are given as bit positions joined by `-`, separated by `;`.
Bits are packed little-endian, least significant bit first.

## Licence

MIT, see `LICENSE`.
