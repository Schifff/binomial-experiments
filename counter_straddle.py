"""
Counter mode with every monomial straddling the block boundary (Proposition "block repetition").
Each of the eight monomials has one bit above position b = 8 (a distinct one) and a distinct part
below position 8, so as the counter's bits 8..15 run through all 256 values every subset of K
appears and the bound 2^|K| = 256 distinct raw blocks must be attained.  Writes counter_straddle.json.
"""
import json, os, subprocess
import numpy as np
import nist_subset as nist

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "gen.exe")
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
K = "0-8;1-9;0-1-10;2-11;0-2-12;1-2-13;0-1-2-14;3-15"
res = {}
for post in ["none", "sign", "bit:2"]:
    path = os.path.join(OUT, f"ctr_straddle_{post.replace(':', '')}.bin")
    subprocess.run([GEN, "keystream", "--r", "256", "--taps", "0,2,5,10", "--drive", "counter", "--K", K,
                    "--post", post, "--N", "256", "--nbits", str(1 << 22), "--seed", "1", "--out", path],
                   capture_output=True, check=True)
    b = nist.load_bits(path, 1 << 22).reshape(-1, 256)
    distinct = len({bytes(r) for r in np.packbits(b, axis=1)})
    res[post] = {"blocks": int(b.shape[0]), "distinct": distinct, "monomials": 8, "bound": 256}
    print(post, res[post])
json.dump({"K": K, "results": res}, open(os.path.join(HERE, "counter_straddle.json"), "w"), indent=1)
