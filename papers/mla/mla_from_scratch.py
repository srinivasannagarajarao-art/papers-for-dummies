"""
Multi-head Latent Attention (DeepSeek-V2, arXiv 2405.04434) -- for programmers.

Run it:      python3 mla_from_scratch.py
Debug it:    breakpoint in scores_absorbed() and diff it against
             scores_expanded() on the same inputs.

No torch. No training. NumPy only. This is a linear-algebra paper: a forward
pass and some byte counting are enough to see the whole idea.

The claim, in one line: cache ONE small latent vector per token instead of
all the keys and values, and fold the up-projection into the query matrix so
the keys never have to be expanded at all.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- the cache formulas. The first one is straight from the KV-cache
# page; MLA only changes what sits inside the per-layer per-token term.
#
#   MHA/GQA:  2 (K and V)  x  kv_heads x head_dim   floats per layer per token
#   MLA:      d_c (the latent)  +  d_rope (the decoupled RoPE key)
#
# Note the missing 2 in the MLA line. There is no separate V to cache: one
# latent serves as both, because K and V are compressed jointly.
# ---------------------------------------------------------------------------
def kv_cache_bytes(layers, kv_heads, head_dim, seq_len, dtype_bytes=2,
                   batch=1):
    """Standard KV cache. Same formula as the KV-cache page, unchanged."""
    return 2 * layers * kv_heads * head_dim * seq_len * dtype_bytes * batch


def mla_cache_bytes(layers, d_c, d_rope, seq_len, dtype_bytes=2, batch=1):
    """MLA cache: one latent per token per layer, plus the decoupled RoPE key.
    Both are shared across ALL heads, which is where the saving comes from."""
    return layers * (d_c + d_rope) * seq_len * dtype_bytes * batch


# ---------------------------------------------------------------------------
# STAGE 1 -- softmax, and a low-rank factorisation helper.
#
# To show correctness honestly I need a down/up pair that CAN be exact at full
# rank. An SVD gives me that: factor the joint [W_k | W_v] into
# W_dkv @ W_ukv, truncated to rank d_c. At full rank the product is the
# original matrix to float precision; below it, it is the best rank-d_c
# approximation there is. A trained model learns its own factors -- this is
# just the cleanest way to sweep the rank and see the trade.
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def low_rank_factors(W, rank):
    """W (m, n) -> (W_down (m, rank), W_up (rank, n)) with W ~ W_down @ W_up."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return U[:, :rank] * S[:rank], Vt[:rank]


# ---------------------------------------------------------------------------
# STAGE 2 -- the model. Ordinary multi-head attention on the left, MLA on the
# right, built from the SAME W_k / W_v so the two are directly comparable.
#
#   standard:   k = x @ W_k                       cache k and v, per head
#   MLA:        c = x @ W_dkv                     cache c only  (d_c floats)
#               k = c @ W_uk                      re-expand on demand
#
# W_dkv is (d_model, d_c); W_ukv is (d_c, 2 * n_heads * head_dim) and gets
# split back into the key half and the value half.
# ---------------------------------------------------------------------------
class ToyMLA:
    def __init__(self, d_model, n_heads, head_dim, d_c, seed=0):
        rng = np.random.default_rng(seed)
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, head_dim
        self.d_c = d_c
        d_all = n_heads * head_dim

        self.W_q = rng.normal(0, 0.4, (d_model, d_all))

        # The joint K/V projection. I shape its singular values to DECAY,
        # because a trained weight matrix has a decaying spectrum and a
        # freshly randomised one does not. Without this the rank sweep below
        # is meaningless -- every direction matters equally, so truncation is
        # uniformly catastrophic and you learn nothing from it.
        raw = rng.normal(0, 0.4, (d_model, 2 * d_all))
        U, _, Vt = np.linalg.svd(raw, full_matrices=False)
        spectrum = 6.0 * (0.82 ** np.arange(min(U.shape[1], Vt.shape[0])))
        W_kv = (U * spectrum) @ Vt                            # (d, 2*d_all)
        self.W_k, self.W_v = W_kv[:, :d_all], W_kv[:, d_all:]

        # Compress K and V JOINTLY -- one latent serves both. That joint-ness
        # is the paper's word and it is why there is no factor of 2 above.
        self.W_dkv, W_ukv = low_rank_factors(W_kv, d_c)
        self.W_uk = W_ukv[:, :d_all]      # (d_c, d_all)
        self.W_uv = W_ukv[:, d_all:]      # (d_c, d_all)

    def heads(self, M):
        """(n, n_heads*head_dim) -> (n_heads, n, head_dim)."""
        n = M.shape[0]
        return M.reshape(n, self.n_heads, self.head_dim).transpose(1, 0, 2)

    def latent(self, X):
        """The ONLY thing that gets cached: (n, d_c)."""
        return X @ self.W_dkv

    def attention_standard(self, X):
        """Textbook multi-head attention. The reference answer."""
        Q, K, V = self.heads(X @ self.W_q), self.heads(X @ self.W_k), \
            self.heads(X @ self.W_v)
        w = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(self.head_dim))
        out = w @ V                                  # (H, n, head_dim)
        return out.transpose(1, 0, 2).reshape(X.shape[0], -1)

    def attention_via_latent(self, X):
        """MLA the naive way: compress, then re-expand K and V and carry on.
        Correct, but it materialises the full K and V -- no saving at all
        except in what you STORE. Demo 4 removes the expansion."""
        C = self.latent(X)                           # (n, d_c)  <- cached
        Q = self.heads(X @ self.W_q)
        K = self.heads(C @ self.W_uk)                # re-expanded
        V = self.heads(C @ self.W_uv)                # re-expanded
        w = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(self.head_dim))
        out = w @ V
        return out.transpose(1, 0, 2).reshape(X.shape[0], -1)


# ---------------------------------------------------------------------------
# STAGE 3 -- ABSORPTION. The trick that makes MLA free.
#
#     (x W_q) . (c W_uk)^T  =  x W_q W_uk^T c^T  =  (x (W_q W_uk^T)) . c^T
#
# Matrix multiplication is associative, so the up-projection can be folded
# into the query projection ONCE, at load time. At inference you project the
# query straight into latent space and dot it against the cached latent. The
# full keys are never built. It is the compiler trick of constant-folding two
# adjacent multiplies, applied to a weight matrix.
# ---------------------------------------------------------------------------
def absorb_q_into_latent(W_q_head, W_uk_head):
    """(d_model, head_dim) x (d_c, head_dim) -> (d_model, d_c). Done ONCE."""
    return W_q_head @ W_uk_head.T


def scores_expanded(x, W_q_head, C, W_uk_head):
    """The obvious way: expand the keys, then dot. Materialises (n, head_dim)."""
    q = x @ W_q_head                     # (1, head_dim)
    K = C @ W_uk_head                    # (n, head_dim)   <- the expansion
    return q @ K.T                       # (1, n)


def scores_absorbed(x, W_absorbed, C):
    """The MLA way: project into latent space, dot against the cache."""
    q_c = x @ W_absorbed                 # (1, d_c)
    return q_c @ C.T                     # (1, n)


def macs_expanded(n_cached, d_model, head_dim, d_c):
    """Per head, per decode step: q projection + key expansion + the dot."""
    return d_model * head_dim + n_cached * d_c * head_dim + n_cached * head_dim


def macs_absorbed(n_cached, d_model, head_dim, d_c):
    """Per head, per decode step: latent-space q projection + the dot.
    The n_cached * d_c * head_dim expansion term is simply gone."""
    return d_model * d_c + n_cached * d_c


# ---------------------------------------------------------------------------
# STAGE 4 -- RoPE, and why it breaks absorption.
#
# RoPE rotates q and k by an angle proportional to their POSITION (see the
# RoPE page). Rotation is a per-position matrix R(m), so
#
#     q_m . k_n  =  (R(m) x W_q) . (R(n) c W_uk)
#
# and there is no way to pull R(n) out and fold it into W_q at load time --
# R(n) is different for every cached token, and W_q is fixed. Absorption dies.
#
# DeepSeek's fix: split the head. Most dimensions carry NO position and stay
# absorbable; a small extra block of d_rope dimensions carries RoPE and is
# cached separately, shared across all heads. The score is the sum of the two.
# ---------------------------------------------------------------------------
def rope_angles(pos, dim, base=10000.0):
    i = np.arange(dim // 2)
    theta = base ** (-2.0 * i / dim)
    return np.asarray(pos)[:, None] * theta[None, :]


def apply_rope(M, positions, base=10000.0):
    """Rotate each (2i, 2i+1) pair of every row by position * theta_i."""
    ang = rope_angles(positions, M.shape[-1], base)
    cos, sin = np.cos(ang), np.sin(ang)
    even, odd = M[..., 0::2], M[..., 1::2]
    out = np.empty_like(M)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


def decoupled_score(x_q, pos_q, C, positions, W_absorbed, W_qr, K_rope):
    """DeepSeek's score: absorbable content part + decoupled RoPE part.

    K_rope is (n, d_rope), already rotated and cached -- ONE copy for all
    heads. That sharing is why the extra cache is cheap."""
    content = (x_q @ W_absorbed) @ C.T                       # no position
    q_r = apply_rope(x_q @ W_qr, np.array([pos_q]))          # (1, d_rope)
    return content + q_r @ K_rope.T


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def gib(b):
    return b / (1024 ** 3)


def demo_1_the_cache_table():
    line("DEMO 1: the cache table -- this is the whole argument")

    print("Formula from the KV-cache page, unchanged:")
    print("  MHA/GQA bytes = 2 * layers * kv_heads * head_dim * seq * dtype")
    print("  MLA     bytes =     layers * (d_c + d_rope)      * seq * dtype")
    print("  (no factor of 2: K and V share ONE latent)\n")

    for name, L, H, hd, d_c, d_r in [
        ("Llama-2-7B shape", 32, 32, 128, 512, 64),
        ("DeepSeek-V2 shape", 60, 128, 128, 512, 64),
    ]:
        mha = kv_cache_bytes(L, H, hd, 1)
        gqa = kv_cache_bytes(L, 8, hd, 1)
        mla = mla_cache_bytes(L, d_c, d_r, 1)
        print(f"{name}: layers={L} heads={H} head_dim={hd} "
              f"d_c={d_c} d_rope={d_r}, fp16")
        print(f"    multi-head     : {mha:>10,d} bytes/token   1.0x")
        print(f"    grouped-query 8: {gqa:>10,d} bytes/token "
              f"{mha / gqa:>5.1f}x smaller")
        print(f"    MLA            : {mla:>10,d} bytes/token "
              f"{mha / mla:>5.1f}x smaller")
        print(f"    MLA vs GQA-8   : {gqa / mla:>5.1f}x smaller\n")

    print("READ THIS: the DeepSeek-V2 shape has 128 heads. Multi-head caching")
    print("would cost 3.75 MiB per token per sequence, which is unservable.")
    print("MLA caches 576 floats per layer instead of 32768, and that number")
    print("does not move when you add heads.")


def demo_2_correctness_and_the_sweep():
    line("DEMO 2: is it the same attention? at full rank, yes")

    d_model, n_heads, head_dim = 32, 4, 8
    d_all = n_heads * head_dim
    X = np.random.default_rng(1).normal(0, 1, (6, d_model))

    full = min(d_model, 2 * d_all)          # the rank the factorisation can hit
    m = ToyMLA(d_model, n_heads, head_dim, d_c=full, seed=0)
    ref = m.attention_standard(X)
    via = m.attention_via_latent(X)
    print(f"d_model={d_model} n_heads={n_heads} head_dim={head_dim}")
    print(f"full-rank latent d_c = {full}")
    print(f"  max |standard - through the latent| = "
          f"{np.abs(ref - via).max():.3e}   <- float noise")
    print(f"  identical to 1e-10?                 {np.allclose(ref, via, atol=1e-10)}")

    print("\nnow shrink the latent and watch what you are actually trading.")
    print("errors are relative: max abs error / max abs value.")
    print("   d_c | K reconstruction | attention output | cache bytes/token")
    K_true = X @ m.W_k
    for d_c in (32, 24, 16, 12, 8, 4, 2):
        mm = ToyMLA(d_model, n_heads, head_dim, d_c=d_c, seed=0)
        K_hat = mm.latent(X) @ mm.W_uk
        k_err = np.abs(K_true - K_hat).max() / np.abs(K_true).max()
        out = mm.attention_via_latent(X)
        o_err = np.abs(ref - out).max() / np.abs(ref).max()
        b = mla_cache_bytes(1, d_c, 0, 1)
        print(f"  {d_c:>4d} | {k_err:>16.4f} | {o_err:>16.4f} | {b:>10,d} B")

    print("\nREAD THIS: at full rank the latent is a lossless re-encoding --")
    print("1e-15 is float64 addition order, nothing more. Below full rank it")
    print("is a lossy compression, and the error in the attention OUTPUT is")
    print("the number that matters, not the error in the keys. Here the output")
    print("tracks the key error closely here -- sometimes a little under it,")
    print("sometimes a little over, because softmax renormalises. Halving the")
    print("cache (d_c 32 -> 16) costs about 5% on the output. Quartering it")
    print("costs 60%. That curve, on real learned weights instead of a")
    print("shaped-spectrum toy, is the only thing choosing d_c = 512 against")
    print("128*128 = 16384 expanded dimensions.")


def demo_3_absorption_identity():
    line("DEMO 3: absorption -- (x Wq)(c Wuk)^T == x (Wq Wuk^T) c^T")

    d_model, n_heads, head_dim, d_c = 32, 4, 8, 16
    m = ToyMLA(d_model, n_heads, head_dim, d_c, seed=2)
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (64, d_model))       # 64 tokens already cached
    C = m.latent(X)                           # (64, d_c) -- the cache
    x_new = rng.normal(0, 1, (1, d_model))    # the token being decoded

    W_q_h = m.W_q[:, :head_dim]               # head 0
    W_uk_h = m.W_uk[:, :head_dim]
    W_abs = absorb_q_into_latent(W_q_h, W_uk_h)

    s_exp = scores_expanded(x_new, W_q_h, C, W_uk_h)
    s_abs = scores_absorbed(x_new, W_abs, C)

    print(f"cached tokens n = {X.shape[0]}, d_c = {d_c}, head_dim = {head_dim}")
    print(f"  W_q       {W_q_h.shape}   W_uk {W_uk_h.shape}")
    print(f"  W_absorbed = W_q @ W_uk.T -> {W_abs.shape}  (built ONCE at load)")
    print(f"  expanded scores[0][:4] = {s_exp[0][:4]}")
    print(f"  absorbed scores[0][:4] = {s_abs[0][:4]}")
    print(f"  max abs difference     = {np.abs(s_exp - s_abs).max():.3e}   <- same thing")

    print("\noperation counts, per head per decode step:")
    print("   cached n |  expanded MACs |  absorbed MACs | saving")
    for n in (64, 512, 4096, 32768):
        e = macs_expanded(n, d_model, head_dim, d_c)
        a = macs_absorbed(n, d_model, head_dim, d_c)
        print(f"  {n:>9,d} | {e:>14,d} | {a:>14,d} | {e / a:>5.2f}x")

    print("\nREAD THIS: associativity, that is all. The keys are never built.")
    print("With a realistic d_c the absorbed path is not always fewer MACs --")
    print("what it removes is the whole expansion tensor and the memory")
    print("traffic behind it, and decode is memory-bound, not compute-bound.")


def demo_4_rope_breaks_absorption():
    line("DEMO 4: RoPE does not commute with absorption")

    d_model, n_heads, head_dim, d_c, d_rope = 32, 4, 8, 16, 8
    m = ToyMLA(d_model, n_heads, head_dim, d_c, seed=4)
    rng = np.random.default_rng(5)
    n = 8
    X = rng.normal(0, 1, (n, d_model))
    C = m.latent(X)
    positions = np.arange(n)
    x_new = rng.normal(0, 1, (1, d_model))
    pos_new = n

    W_q_h, W_uk_h = m.W_q[:, :head_dim], m.W_uk[:, :head_dim]
    W_abs = absorb_q_into_latent(W_q_h, W_uk_h)

    # The honest reference: expand the keys, rotate each by ITS position.
    q_rot = apply_rope(x_new @ W_q_h, np.array([pos_new]))
    K_rot = apply_rope(C @ W_uk_h, positions)
    ref = q_rot @ K_rot.T

    # The naive absorbed attempt: absorb first, rotate the latent-space query.
    # Shapes line up. The answer does not.
    naive = apply_rope(x_new @ W_abs, np.array([pos_new])) @ C.T

    print("scores over 8 cached tokens, head 0, with RoPE applied:")
    print(f"  correct (expand, then rotate) : {ref[0][:4]}")
    print(f"  absorbed then rotated         : {naive[0][:4]}")
    print(f"  max abs difference            = {np.abs(ref - naive).max():.4f}"
          "   <- wrong")
    print(f"  ranking preserved?              "
          f"{np.array_equal(np.argsort(ref[0]), np.argsort(naive[0]))}")

    # DeepSeek's fix: a small decoupled block that carries the rotation.
    W_kr = rng.normal(0, 0.4, (d_model, d_rope))    # shared by ALL heads
    W_qr = rng.normal(0, 0.4, (d_model, d_rope))
    K_rope = apply_rope(X @ W_kr, positions)        # (n, d_rope) -- cached
    fixed = decoupled_score(x_new, pos_new, C, positions, W_abs, W_qr, K_rope)

    content = (x_new @ W_abs) @ C.T
    q_r = apply_rope(x_new @ W_qr, np.array([pos_new]))
    check = content + q_r @ K_rope.T
    print("\nDeepSeek's fix -- split the head in two:")
    print(f"  content part (absorbed, no position) : {content[0][:4]}")
    print(f"  decoupled RoPE part (d_rope={d_rope})   : {(q_r @ K_rope.T)[0][:4]}")
    print(f"  score = sum                          : {fixed[0][:4]}")
    print(f"  matches the explicit two-part sum?     "
          f"{np.allclose(fixed, check)}, max diff "
          f"{np.abs(fixed - check).max():.3e}")

    print("\nwhat the decoupled block costs, per token, DeepSeek-V2 shape:")
    L, d_c2, d_r2 = 60, 512, 64
    base = mla_cache_bytes(L, d_c2, 0, 1)
    with_r = mla_cache_bytes(L, d_c2, d_r2, 1)
    print(f"  latent only        : {base:>7,d} bytes/token")
    print(f"  + decoupled RoPE   : {with_r:>7,d} bytes/token "
          f"(+{with_r - base:,d}, +{100 * (with_r - base) / base:.1f}%)")
    print(f"  still vs multi-head: "
          f"{kv_cache_bytes(L, 128, 128, 1) / with_r:.1f}x smaller")

    print("\nREAD THIS: R(n) is a different matrix for every cached token, so")
    print("it cannot be folded into a fixed W_q. The absorption and the")
    print("rotation genuinely do not commute. DeepSeek pays 64 extra")
    print("dimensions per token -- ONE copy shared by all heads -- to keep")
    print("the other 512 absorbable. Twelve percent more cache for the trick")
    print("to survive. See ../rope/ for why the rotation has to be per")
    print("position in the first place.")


def demo_5_where_the_savings_land():
    line("DEMO 5: where the savings land -- sequences per GPU")

    L, H, hd, d_c, d_r = 60, 128, 128, 512, 64
    budget = 80 * 1024 ** 3          # one 80 GB card, cache only
    print(f"DeepSeek-V2 shape, fp16. Budget: {gib(budget):.0f} GiB of KV cache.\n")

    print("   context | multi-head |    GQA-8 |      MLA | MLA seqs in 80 GiB")
    for seq in (4096, 16384, 32768, 131072):
        mha = kv_cache_bytes(L, H, hd, seq)
        gqa = kv_cache_bytes(L, 8, hd, seq)
        mla = mla_cache_bytes(L, d_c, d_r, seq)
        print(f"  {seq:>8,d} | {gib(mha):>9.2f}G | {gib(gqa):>7.2f}G | "
              f"{gib(mla):>7.2f}G | {budget // mla:>6,d}")

    print("\n  sequences that fit in 80 GiB at 32k context:")
    seq = 32768
    for name, b in [("multi-head", kv_cache_bytes(L, H, hd, seq)),
                    ("GQA-8", kv_cache_bytes(L, 8, hd, seq)),
                    ("MLA", mla_cache_bytes(L, d_c, d_r, seq))]:
        print(f"    {name:<12s}: {budget // b:>6,d} concurrent sequences")

    print("\nREAD THIS: concurrency IS throughput. At 32k context this shape")
    print("cannot serve a SINGLE multi-head sequence in 80 GiB of cache. GQA-8")
    print("fits 10. MLA fits 37. Same latency per token in all three cases --")
    print("nearly 4x the tokens per second per GPU between the last two. That")
    print("is the commercial argument, and it is why a lab that serves its own")
    print("model at scale is the one that invented this.")


def demo_6_honest_comparison_with_gqa():
    line("DEMO 6: honest comparison -- MLA is not free, it is a trade")

    d, L, H, hd, d_c, d_r = 5120, 60, 128, 128, 512, 64
    d_all = H * hd
    mha_p = 4 * d * d_all
    gqa_p = 2 * d * d_all + 2 * d * (8 * hd)
    mla_p = (d * d_all            # W_q
             + d * d_c            # W_dkv
             + d_c * d_all        # W_uk
             + d_c * d_all        # W_uv
             + d * d_r            # W_kr, shared
             + d * (H * d_r)      # W_qr, per head
             + d * d_all)         # W_o

    print(f"per attention layer, d_model={d}, {H} heads x {hd}:")
    print(f"  multi-head    params: {mha_p:>13,d}")
    print(f"  GQA-8         params: {gqa_p:>13,d}")
    print(f"  MLA           params: {mla_p:>13,d}  "
          f"({mla_p / mha_p:.2f}x multi-head, {mla_p / gqa_p:.2f}x GQA-8)")

    seq = 32768
    print(f"\n  cache at {seq:,d} tokens, batch 1:")
    print(f"  multi-head : {gib(kv_cache_bytes(L, H, hd, seq)):>7.2f} GiB")
    print(f"  GQA-8      : {gib(kv_cache_bytes(L, 8, hd, seq)):>7.2f} GiB")
    print(f"  MLA        : {gib(mla_cache_bytes(L, d_c, d_r, seq)):>7.2f} GiB")

    print("\nREAD THIS: be careful with the parameter claim. Against 128-head")
    print("multi-head, MLA is actually CHEAPER in parameters, because the")
    print("latent is a bottleneck. Against GQA-8 it costs 1.29x -- and it")
    print("costs arithmetic too: an extra projection on every token, and a")
    print("wider per-head dimension once the RoPE block is bolted on.")
    print("Weights are paid once and shared across the whole batch; cache is")
    print("paid per sequence per token. Trading a fixed cost for a per-request")
    print("cost is a good trade at serving scale and a bad one at batch 1.")
    print("DeepSeek report MLA beating GQA on quality at comparable cache in")
    print("their ablations. This script does not measure quality and cannot.")


def demo_7_break_it():
    line("DEMO 7: five ways to get this wrong")

    d_model, n_heads, head_dim, d_c = 32, 4, 8, 16
    m = ToyMLA(d_model, n_heads, head_dim, d_c, seed=6)
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (64, d_model))
    C = m.latent(X)
    x_new = rng.normal(0, 1, (1, d_model))
    W_q_h, W_uk_h, W_uv_h = (m.W_q[:, :head_dim], m.W_uk[:, :head_dim],
                             m.W_uv[:, :head_dim])
    W_abs = absorb_q_into_latent(W_q_h, W_uk_h)
    good = scores_absorbed(x_new, W_abs, C)

    print("BREAK 1 -- latent too small (d_c = 2 instead of full rank):")
    ref = ToyMLA(d_model, n_heads, head_dim, 32, seed=6).attention_standard(X)
    tiny = ToyMLA(d_model, n_heads, head_dim, 2, seed=6).attention_via_latent(X)
    print(f"    attention output max abs error = {np.abs(ref - tiny).max():.4f}"
          "   <- not a rounding error")

    print("\nBREAK 2 -- cache the expanded K and V instead of the latent:")
    L2, d_c2, d_r2 = 60, 512, 64
    lat = mla_cache_bytes(L2, d_c2, d_r2, 1)
    exp = kv_cache_bytes(L2, 128, 128, 1)
    print(f"    latent cached  : {lat:>10,d} bytes/token")
    print(f"    expanded cached: {exp:>10,d} bytes/token  "
          f"<- {exp / lat:.1f}x worse, the entire point gone")

    print("\nBREAK 3 -- absorb, then apply RoPE, with no decoupled dims:")
    pos = np.arange(64)
    q_rot = apply_rope(x_new @ W_q_h, np.array([64]))
    ref_s = q_rot @ apply_rope(C @ W_uk_h, pos).T
    bad_s = apply_rope(x_new @ W_abs, np.array([64])) @ C.T
    print(f"    correct scores[:3] = {ref_s[0][:3]}")
    print(f"    absorbed+rotated   = {bad_s[0][:3]}")
    print(f"    max abs difference = {np.abs(ref_s - bad_s).max():.4f}")

    print("\nBREAK 4 -- absorb the wrong matrix. Two orders, two outcomes:")
    W_flip = W_uk_h @ W_q_h.T                # (d_c, d_model) -- shapes fit
    flipped = (x_new @ W_flip.T) @ C.T
    W_wrong = W_q_h @ W_uv_h.T               # (d_model, d_c) -- shapes fit too
    wrong = (x_new @ W_wrong) @ C.T
    print(f"    correct  W_q @ W_uk.T    : shape {W_abs.shape}")
    print(f"    flipped  W_uk @ W_q.T    : shape {W_flip.shape}, "
          "used transposed")
    print(f"    wrong    W_q @ W_uv.T    : shape {W_wrong.shape}")
    print(f"    correct scores[:3] = {good[0][:3]}")
    print(f"    flipped scores[:3] = {flipped[0][:3]}")
    print(f"    wrong   scores[:3] = {wrong[0][:3]}")
    print(f"    correct vs flipped, max abs diff = "
          f"{np.abs(good - flipped).max():.4f}   <- the transpose is harmless")
    print(f"    correct vs wrong,   max abs diff = "
          f"{np.abs(good - wrong).max():.4f}   <- silently a different model")

    print("\nBREAK 5 -- rebuild the absorbed matrix every decode step:")
    per_step = d_model * head_dim * d_c
    for steps in (1, 1024, 32768):
        print(f"    {steps:>6,d} steps x {n_heads} heads: "
              f"{steps * n_heads * per_step:>14,d} extra MACs "
              f"({'correct, and free' if steps == 1 else 'correct, and wasted'})")

    print("\nREAD THIS: break 4 is the nastiest, and not in the way you")
    print("expect. Reversing the order gives back the transpose, so it is")
    print("harmless -- 0.0000. Reaching for W_uv instead of W_uk has the")
    print("identical shape, raises nothing, produces plausible-looking")
    print("scores, and is a different model. Shape checks are not")
    print("correctness checks. Only the second path shows up in your eval.")


if __name__ == "__main__":
    demo_1_the_cache_table()
    demo_2_correctness_and_the_sweep()
    demo_3_absorption_identity()
    demo_4_rope_breaks_absorption()
    demo_5_where_the_savings_land()
    demo_6_honest_comparison_with_gqa()
    demo_7_break_it()

    line("MLA, COMPRESSED")
    print("""
  1. cache      = one latent c = x @ W_dkv per token per layer, not K and V
  2. jointly    = K and V share the latent, so no factor of 2 in the bytes
  3. expand     = k = c @ W_uk, v = c @ W_uv, only if you actually need them
  4. absorption = W_q @ W_uk.T folded once at load, so you never need them
  5. RoPE       = does not commute with (4), so a small decoupled block of
                  dimensions carries the rotation and is cached alongside
  6. query lora = a separate low-rank compression of Q. Saves ACTIVATION
                  memory in training. Nothing to do with the inference cache.
  7. the trade  = more parameters and more arithmetic, far less cache.

  Everything else in the DeepSeek-V2 paper is the MoE, the routing and the
  training recipe. Real, but a different paper.
""")
