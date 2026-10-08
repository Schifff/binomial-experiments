"""
Orchestrates keystream generation, the statistical battery and throughput runs.
Writes results.json.  Run from the experiments directory:  python run_all.py
"""
import json, os, subprocess, sys, time, statistics
import numpy as np
import nist_subset as nist

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "gen.exe")
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
NB = 1 << 24          # bits per statistical sample
NB_TP = 1 << 27       # bits per throughput run
TAPS256 = "0,2,5,10"  # x^256 + x^10 + x^5 + x^2 + 1 (verified primitive)

def mono(*bits): return "-".join(str(b) for b in bits)
Q1 = ";".join(mono(*range(s, s + 8)) for s in [1, 33, 65, 97, 129, 161, 193, 225])           # eight disjoint degree-8 monomials
Q2 = "0;" + Q1                                                                                 # x0 + Q1 (balanced)
Q3 = "0;" + ";".join(mono(16 + 2 * i, 17 + 2 * i) for i in range(16))                          # x0 + bent quadratic on 32 vars
Q4 = "0;" + ";".join(mono(*range(s, s + d)) for d, s in zip(range(2, 9), [1, 3, 6, 10, 15, 21, 28]))  # degrees 2..8, disjoint
FILTERS = {"Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4}
POSTS = ["none", "sign", "bit:1", "bit:2", "bit:3", "bit:7"]

def run(args, **kw):
    r = subprocess.run([GEN] + [str(a) for a in args], capture_output=True, text=True, check=True, **kw)
    return r.stdout

def keystream_file(name, extra, nbits=NB, seed=1):
    path = os.path.join(OUT, name + ".bin")
    if not os.path.exists(path) or os.path.getsize(path) < nbits // 8:
        run(["keystream", "--r", 256, "--taps", TAPS256] + extra + ["--nbits", nbits, "--seed", seed, "--out", path])
    return path

def throughput(args, reps=3):
    vals = []
    for i in range(reps):
        out = run(args + ["--nbits", NB_TP, "--seed", 11 + i])
        vals.append(float(out.split("throughput_Mbit_s=")[1].split()[0]))
    return statistics.mean(vals), statistics.pstdev(vals)

def gf2_rank(rows):
    rank = 0; pivots = {}
    for x in rows:
        while x:
            h = x.bit_length() - 1
            if h in pivots: x ^= pivots[h]
            else: pivots[h] = x; rank += 1; break
    return rank

def block_rank(path, N, nblocks=4096):
    b = nist.load_bits(path, N * nblocks).reshape(nblocks, N)
    rows = [int.from_bytes(np.packbits(r, bitorder="little").tobytes(), "little") for r in b]
    return gf2_rank(rows)

def main():
    results = {"battery": {}, "throughput": {}, "onset": {}, "nblock": {}, "counter": {}, "block_rank": {}}
    t0 = time.time()
    # ---- baselines ----
    cc = os.path.join(OUT, "chacha20.bin")
    if not os.path.exists(cc): run(["chacha20", "--nbits", NB, "--seed", 1, "--out", cc])
    lf = os.path.join(OUT, "lfsr256.bin")
    if not os.path.exists(lf): run(["lfsr", "--r", 256, "--taps", TAPS256, "--nbits", NB, "--seed", 1, "--out", lf])
    results["battery"]["chacha20"] = nist.battery(cc, NB); print("chacha20", results["battery"]["chacha20"], flush=True)
    results["battery"]["lfsr256"] = nist.battery(lf, NB); print("lfsr256", results["battery"]["lfsr256"], flush=True)
    # ---- main grid: 4 filters x 6 post-processings, N = 256 ----
    for fn, K in FILTERS.items():
        for post in POSTS:
            name = f"{fn}_N256_{post.replace(':', '')}"
            path = keystream_file(name, ["--drive", "lfsr", "--K", K, "--post", post, "--N", 256])
            results["battery"][name] = nist.battery(path, NB)
            results["block_rank"][name] = block_rank(path, 256)
            print(name, "block_rank", results["block_rank"][name], {k: round(v, 4) for k, v in results["battery"][name].items()}, flush=True)
    # ---- block-size ablation on Q3: N in {16,64,256,1024,4096} for sign and bit:2 ----
    for N in [16, 64, 256, 1024, 4096]:
        for post in ["sign", "bit:2"]:
            name = f"Q3_N{N}_{post.replace(':', '')}"
            path = keystream_file(name, ["--drive", "lfsr", "--K", Q3, "--post", post, "--N", N])
            results["nblock"][name] = nist.battery(path, NB)
            results["block_rank"][name] = block_rank(path, N)
            print(name, "block_rank", results["block_rank"][name], {k: round(v, 4) for k, v in results["nblock"][name].items()}, flush=True)
    # ---- failure onset: monobit p-value vs prefix length for sign extraction ----
    for N in [64, 256, 1024]:
        name = f"Q3_N{N}_sign"
        path = os.path.join(OUT, name + ".bin")
        if not os.path.exists(path): path = keystream_file(name, ["--drive", "lfsr", "--K", Q3, "--post", "sign", "--N", N])
        b = nist.load_bits(path, NB)
        results["onset"][name] = {str(n): float(nist.frequency(b[:n])) for n in [256, 1024, 4096, 16384, 65536, 1 << 20, NB]}
        results["onset"][name]["ones_fraction"] = float(b.mean())
        print("onset", name, results["onset"][name], flush=True)
    # ---- counter mode: number of distinct N-bit output blocks ----
    for fn in ["Q2", "Q4"]:
        for post in ["none", "sign", "bit:2"]:
            name = f"ctr_{fn}_N256_{post.replace(':', '')}"
            path = keystream_file(name, ["--drive", "counter", "--K", FILTERS[fn], "--post", post, "--N", 256], nbits=1 << 22)
            b = nist.load_bits(path, 1 << 22).reshape(-1, 256)
            packed = np.packbits(b, axis=1)
            distinct = len({bytes(r) for r in packed})
            results["counter"][name] = {"blocks": int(b.shape[0]), "distinct": distinct, "monomials": FILTERS[fn].count(";") + 1}
            print("counter", name, results["counter"][name], flush=True)
    # ---- throughput (in-memory, 2^27 bits, 3 repetitions) ----
    tp = {}
    tp["chacha20"] = throughput(["chacha20"])
    tp["lfsr256"] = throughput(["lfsr", "--r", 256, "--taps", TAPS256])
    for fn in ["Q2", "Q3"]:
        for post in ["none", "sign", "bit:2"]:
            for N in ([256] if post == "none" else [64, 256, 1024]):
                tp[f"{fn}_N{N}_{post.replace(':', '')}"] = throughput(["keystream", "--r", 256, "--taps", TAPS256, "--drive", "lfsr", "--K", FILTERS[fn], "--post", post, "--N", N])
    results["throughput"] = tp
    print("throughput", tp, flush=True)
    results["elapsed_s"] = time.time() - t0
    with open(os.path.join(HERE, "results.json"), "w") as f: json.dump(results, f, indent=1)
    print("DONE", results["elapsed_s"])

if __name__ == "__main__":
    main()
