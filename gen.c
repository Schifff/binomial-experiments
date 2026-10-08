#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXR 256
#define WORDS (MAXR / 64)
#define MAXK 64
#define MAXN 4096

typedef struct { uint64_t w[WORDS]; } st_t;

static st_t tapmask;
static st_t kmask[MAXK]; static int nK = 0;
static int R = 256;

static uint64_t sd = 0x9E3779B97F4A7C15ull;
static uint64_t rnd64(void) { sd ^= sd << 13; sd ^= sd >> 7; sd ^= sd << 17; return sd * 0x2545F4914F6CDD1Dull; }

static inline int parity64(uint64_t x) { return __builtin_parityll(x); }

static inline void lfsr_step(st_t *x) {
    uint64_t acc = 0;
    for (int i = 0; i < WORDS; i++) acc ^= x->w[i] & tapmask.w[i];
    uint64_t nb = (uint64_t)parity64(acc);
    for (int i = 0; i < WORDS - 1; i++) x->w[i] = (x->w[i] >> 1) | (x->w[i + 1] << 63);
    x->w[WORDS - 1] >>= 1;
    x->w[(R - 1) / 64] |= nb << ((R - 1) % 64);
}
static inline void mask_state(st_t *x) {
    if (R < MAXR) { int hw = R / 64, hb = R % 64;
        if (hb) x->w[hw] &= (1ull << hb) - 1;
        for (int i = hw + (hb ? 1 : 0); i < WORDS; i++) x->w[i] = 0; }
}

static inline void counter_step(st_t *x) {
    for (int i = 0; i < WORDS; i++) { if (++x->w[i] != 0) break; }
    mask_state(x);
}

static inline int filterQ(const st_t *x) {
    int q = 0;
    for (int j = 0; j < nK; j++) {
        int all = 1;
        for (int i = 0; i < WORDS; i++) if ((x->w[i] & kmask[j].w[i]) != kmask[j].w[i]) { all = 0; break; }
        q ^= all;
    }
    return q;
}

static void fwht(int32_t *a, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) { int32_t u = a[j], v = a[j + len]; a[j] = u + v; a[j + len] = u - v; }
}

#define ROTL(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
#define QR(a,b,c,d) (a += b, d ^= a, d = ROTL(d,16), c += d, b ^= c, b = ROTL(b,12), a += b, d ^= a, d = ROTL(d,8), c += d, b ^= c, b = ROTL(b,7))
static void chacha20_block(const uint32_t in[16], uint32_t out[16]) {
    uint32_t x[16]; memcpy(x, in, 64);
    for (int i = 0; i < 10; i++) {
        QR(x[0], x[4], x[8], x[12]); QR(x[1], x[5], x[9], x[13]); QR(x[2], x[6], x[10], x[14]); QR(x[3], x[7], x[11], x[15]);
        QR(x[0], x[5], x[10], x[15]); QR(x[1], x[6], x[11], x[12]); QR(x[2], x[7], x[8], x[13]); QR(x[3], x[4], x[9], x[14]);
    }
    for (int i = 0; i < 16; i++) out[i] = x[i] + in[i];
}

static inline uint64_t fetch64(const uint64_t *s, int64_t pos, int64_t nbits) {
    if (pos + 64 <= 0 || pos >= nbits) return 0;
    if (pos < 0) { int shift = (int)(-pos); return shift >= 64 ? 0 : (s[0] << shift); }
    int64_t w = pos >> 6; int sh = (int)(pos & 63);
    uint64_t lo = s[w], hi = ((w + 1) * 64 < nbits) ? s[w + 1] : 0;
    return sh ? ((lo >> sh) | (hi << (64 - sh))) : lo;
}
static int64_t berlekamp_massey(const uint64_t *s, int64_t n) {
    int64_t P = n;
    int64_t nw = (P + 64) / 64 + 2;
    uint64_t *C = calloc((size_t)nw, 8), *B = calloc((size_t)nw, 8), *T = calloc((size_t)nw, 8);
    C[P >> 6] |= 1ull << (P & 63); B[P >> 6] |= 1ull << (P & 63);
    int64_t L = 0, nB = -1;
    for (int64_t t = 0; t < n; t++) {
        int d = 0;
        int64_t lo = P - L;
        for (int64_t q = (lo >> 6) << 6; q <= P; q += 64) {
            uint64_t cw = C[q >> 6]; if (!cw) continue;
            d ^= parity64(cw & fetch64(s, q + (t - P), n));
        }
        if (!d) continue;
        int64_t m = t - nB;
        int64_t q0 = P - L - m - 64; if (q0 < 0) q0 = 0; q0 = (q0 >> 6) << 6;
        if (2 * L <= t) {
            memcpy(T, C, (size_t)nw * 8);
            for (int64_t q = q0; q <= P; q += 64) { uint64_t bw = fetch64(B, q + m, P + 1); if (bw) C[q >> 6] ^= bw; }
            L = t + 1 - L; memcpy(B, T, (size_t)nw * 8); nB = t;
        } else {
            for (int64_t q = q0; q <= P; q += 64) { uint64_t bw = fetch64(B, q + m, P + 1); if (bw) C[q >> 6] ^= bw; }
        }
    }
    free(C); free(B); free(T);
    return L;
}

static void set_bit(st_t *x, int b) { x->w[b / 64] |= 1ull << (b % 64); }
static void parse_taps(const char *s) {
    memset(&tapmask, 0, sizeof tapmask);
    char *buf = strdup(s), *tok = strtok(buf, ",");
    while (tok) { set_bit(&tapmask, atoi(tok)); tok = strtok(NULL, ","); }
    free(buf);
}
static void parse_K(const char *s) {
    char *buf = strdup(s), *save1, *mono = strtok_r(buf, ";", &save1);
    nK = 0;
    while (mono && nK < MAXK) {
        memset(&kmask[nK], 0, sizeof(st_t));
        char *save2, *b = strtok_r(mono, "-", &save2);
        while (b) { set_bit(&kmask[nK], atoi(b)); b = strtok_r(NULL, "-", &save2); }
        nK++; mono = strtok_r(NULL, ";", &save1);
    }
    free(buf);
}
static void random_state(st_t *x) {
    for (int i = 0; i < WORDS; i++) x->w[i] = rnd64();
    mask_state(x);
    int any = 0; for (int i = 0; i < WORDS; i++) any |= (x->w[i] != 0); if (!any) x->w[0] = 1;
}
static const char *argval(int argc, char **argv, const char *key, const char *def) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
    return def;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gen {keystream|chacha20|lfsr|bm|bmblocks} [options]\n"); return 1; }
    const char *cmd = argv[1];
    {
        uint64_t z = (uint64_t)atoll(argval(argc, argv, "--seed", "1")) + 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
        sd = z ? z : 0x2545F4914F6CDD1Dull; for (int i = 0; i < 16; i++) rnd64();
    }
    int64_t nbits = atoll(argval(argc, argv, "--nbits", "1048576"));
    const char *out = argval(argc, argv, "--out", NULL);

    if (!strcmp(cmd, "bmblocks")) {
        const char *in = argval(argc, argv, "--in", NULL); int M = atoi(argval(argc, argv, "--M", "500"));
        FILE *f = fopen(in, "rb"); if (!f) { perror(in); return 1; }
        int64_t nb = (nbits + 7) / 8; uint8_t *raw = calloc((size_t)nb + 8, 1);
        size_t got = fread(raw, 1, (size_t)nb, f); fclose(f);
        if ((int64_t)got * 8 < nbits) { fprintf(stderr, "short read\n"); return 1; }
        int64_t nblk = nbits / M; uint64_t *blk = calloc((size_t)(M / 64 + 4), 8);
        for (int64_t i = 0; i < nblk; i++) {
            memset(blk, 0, (size_t)(M / 64 + 4) * 8);
            for (int j = 0; j < M; j++) { int64_t p = i * (int64_t)M + j; if ((raw[p >> 3] >> (p & 7)) & 1) blk[j >> 6] |= 1ull << (j & 63); }
            printf("%lld%c", (long long)berlekamp_massey(blk, M), (i + 1 == nblk) ? '\n' : ' ');
        }
        free(raw); free(blk); return 0;
    }
    if (!strcmp(cmd, "bm")) {
        const char *in = argval(argc, argv, "--in", NULL); FILE *f = fopen(in, "rb"); if (!f) { perror(in); return 1; }
        int64_t nw = (nbits + 63) / 64 + 2; uint64_t *s = calloc((size_t)nw, 8);
        size_t got = fread(s, 1, (size_t)((nbits + 7) / 8), f); fclose(f);
        if ((int64_t)got * 8 < nbits) { fprintf(stderr, "short read\n"); return 1; }
        if (nbits % 64) s[nbits / 64] &= (1ull << (nbits % 64)) - 1;
        clock_t c0 = clock(); int64_t L = berlekamp_massey(s, nbits);
        printf("nbits=%lld LC=%lld time_s=%.2f\n", (long long)nbits, (long long)L, (double)(clock() - c0) / CLOCKS_PER_SEC);
        free(s); return 0;
    }

    uint8_t *buf = calloc((size_t)((nbits + 7) / 8) + 8, 1);
    clock_t c0 = clock();

    if (!strcmp(cmd, "chacha20")) {
        uint32_t st[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};
        for (int i = 4; i < 16; i++) st[i] = (uint32_t)rnd64();
        st[12] = 0;
        uint32_t blk[16]; int64_t pos = 0;
        while (pos < nbits) {
            chacha20_block(st, blk); st[12]++;
            int64_t take = nbits - pos < 512 ? nbits - pos : 512;
            memcpy(buf + pos / 8, blk, (size_t)((take + 7) / 8)); pos += take;
        }
    } else if (!strcmp(cmd, "lfsr") || !strcmp(cmd, "keystream")) {
        R = atoi(argval(argc, argv, "--r", "256"));
        parse_taps(argval(argc, argv, "--taps", "0,2,5,10"));
        st_t x; random_state(&x);
        if (!strcmp(cmd, "lfsr")) {
            for (int64_t t = 0; t < nbits; t++) { buf[t / 8] |= (uint8_t)((x.w[0] & 1) << (t % 8)); lfsr_step(&x); }
        } else {
            int counter = !strcmp(argval(argc, argv, "--drive", "lfsr"), "counter");
            parse_K(argval(argc, argv, "--K", "0-1-2"));
            const char *post = argval(argc, argv, "--post", "none");
            int N = atoi(argval(argc, argv, "--N", "256"));
            int mode = 0, J = 0;
            if (!strcmp(post, "sign")) mode = 1; else if (!strncmp(post, "bit:", 4)) { mode = 2; J = atoi(post + 4); }
            if (counter) x.w[0] &= ~((uint64_t)N - 1);
            if (mode == 0) {
                for (int64_t t = 0; t < nbits; t++) { buf[t / 8] |= (uint8_t)(filterQ(&x) << (t % 8)); if (counter) counter_step(&x); else lfsr_step(&x); }
            } else {
                int32_t a[MAXN]; int64_t pos = 0;
                while (pos < nbits) {
                    for (int n = 0; n < N; n++) { a[n] = filterQ(&x) ? -1 : 1; if (counter) counter_step(&x); else lfsr_step(&x); }
                    fwht(a, N);
                    for (int u = 0; u < N && pos < nbits; u++, pos++) {
                        int bit;
                        if (mode == 1) bit = a[u] < 0;
                        else { int32_t w = (N - a[u]) / 2; bit = (w >> J) & 1; }
                        buf[pos / 8] |= (uint8_t)(bit << (pos % 8));
                    }
                }
            }
        }
    } else { fprintf(stderr, "unknown command\n"); return 1; }
    double secs = (double)(clock() - c0) / CLOCKS_PER_SEC;
    printf("throughput_Mbit_s=%.2f nbits=%lld time_s=%.3f\n", secs > 0 ? (double)nbits / secs / 1e6 : 0.0, (long long)nbits, secs);
    if (out) { FILE *f = fopen(out, "wb"); fwrite(buf, 1, (size_t)((nbits + 7) / 8), f); fclose(f); }
    free(buf); return 0;
}
