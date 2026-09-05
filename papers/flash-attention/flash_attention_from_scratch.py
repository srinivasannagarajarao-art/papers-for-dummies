"""
FlashAttention -- for programmers, not researchers.

Run it:      python3 flash_attention_from_scratch.py
Debug it:    breakpoint inside flash_attention() and watch m_i / l_i move.

No torch. No GPU. You don't need one, and pretending to benchmark a GPU I
don't have would be a lie. Instead the memory hierarchy is SIMULATED with
explicit byte counters: every read from and write to "slow memory" (HBM) is
counted by hand. That is the paper's actual subject -- reads and writes --
so counting them is the honest experiment.

Every core function is short. The demos print the evidence.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)

BYTES = 4          # float32
KB = 1024
MB = 1024 * 1024


# ---------------------------------------------------------------------------
# STAGE 0 -- the memory hierarchy, as a counter.
#
# A GPU has two memories: HBM (big, ~40GB, slow) and SRAM (tiny, ~100KB per
# streaming multiprocessor, ~10x the bandwidth). Arithmetic happens in SRAM.
# Anything that doesn't fit in SRAM has to be shipped in and out of HBM.
#
# I can't measure that from Python, so I count it instead. Every algorithm
# below reports how many bytes it would move and how many multiply-adds it
# would do. Those two numbers are the whole paper.
# ---------------------------------------------------------------------------
class Counters:
    def __init__(self):
        self.hbm_bytes = 0      # traffic between slow and fast memory
        self.flops = 0          # multiply-accumulate operations
        self.peak_extra = 0     # biggest scratch buffer we ever hold, in floats

    def move(self, n_floats):
        self.hbm_bytes += n_floats * BYTES

    def matmul(self, m, k, n):
        self.flops += 2 * m * k * n     # a multiply and an add per element

    def scratch(self, n_floats):
        self.peak_extra = max(self.peak_extra, n_floats)


# ---------------------------------------------------------------------------
# STAGE 1 -- softmax, the ordinary one, exactly as on the attention page.
#
# The `- max` is not in anybody's formula; it is overflow safety. exp(1000)
# is inf. exp(1000-1000) is 1. It cancels in the ratio, so it changes
# nothing mathematically. Remember this trick -- FlashAttention is built on
# doing it INCREMENTALLY.
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# STAGE 2 -- standard attention, with the traffic written down.
#
# The shape of the problem: Q, K, V are (n, d). The score matrix is (n, n).
# For n=8192 that matrix is 268MB in float32 -- it does not fit in SRAM, so
# it gets written to HBM and read back. Three times, roughly: written after
# the matmul, read+written by the softmax, read again by the @V.
#
# The multiplying is n^2*d. The moving is n^2. When d is 64, moving costs
# more wall-clock than multiplying, because HBM is that much slower.
# ---------------------------------------------------------------------------
def standard_attention(Q, K, V, c):
    n, d = Q.shape

    c.move(3 * n * d)               # read Q, K, V in
    S = Q @ K.T                     # (n, n) scores
    c.matmul(n, d, n)
    c.move(n * n)                   # WRITE the score matrix to HBM
    c.scratch(n * n)                # ...and we are holding it

    c.move(n * n)                   # read it back for softmax
    P = softmax(S, axis=-1)
    c.move(n * n)                   # write the probabilities back

    c.move(n * n)                   # read them again for the @V
    O = P @ V
    c.matmul(n, n, d)
    c.move(n * d)                   # write the output
    return O


# ---------------------------------------------------------------------------
# STAGE 3 -- ONLINE SOFTMAX. This is the whole trick. Everything else is
# bookkeeping around it.
#
# You want softmax over a row you are only allowed to see in blocks. Carry
# two numbers as you stream:
#     m = the biggest score seen so far
#     l = the running sum of exp(score - m)
#
# When a new block contains a bigger maximum, the old sum was computed
# against the OLD m. Rescale it by exp(m_old - m_new) and it is correct
# again. That rescale is the load-bearing line. Drop it and the answer is
# silently wrong.
#
# This is Milakov & Gimelshein's online normalizer calculation (2018);
# FlashAttention's contribution is applying it to the whole attention
# pipeline so the (n, n) matrix never has to exist.
# ---------------------------------------------------------------------------
def online_softmax(row, block=8, rescale=True):
    m = -np.inf        # running max
    l = 0.0            # running sum of exp(x - m)
    for j in range(0, len(row), block):
        chunk = row[j:j + block]
        m_new = max(m, chunk.max())
        if rescale:
            l = l * np.exp(m - m_new) if m > -np.inf else l
        l = l + np.exp(chunk - m_new).sum()
        m = m_new
    # Second pass only because we want the full vector back for comparison;
    # attention does not need it, because it accumulates O as it goes.
    return np.exp(row - m) / l


# ---------------------------------------------------------------------------
# STAGE 4 -- FLASHATTENTION. Tile Q into row blocks, K/V into column blocks.
#
# For each query block we walk the key blocks, and we carry three things per
# query row: m (running max), l (running sum), O (running output). When m
# moves, BOTH l and O must be rescaled -- O is a weighted sum whose weights
# were computed against the old maximum.
#
# The (n, n) score matrix is never materialised. Only a (Br, Bc) tile ever
# exists, and that tile is chosen to fit in SRAM. Same function, same
# answer, a fraction of the traffic.
# ---------------------------------------------------------------------------
def flash_attention(Q, K, V, c, Br=64, Bc=64, rescale=True, keep_full_S=False):
    n, d = Q.shape
    O = np.zeros((n, d))
    debug_S = np.zeros((n, n)) if keep_full_S else None

    for i in range(0, n, Br):
        Qi = Q[i:i + Br]
        c.move(Qi.size)                       # load the query tile
        Oi = np.zeros_like(Qi)
        mi = np.full((Qi.shape[0], 1), -np.inf)
        li = np.zeros((Qi.shape[0], 1))
        # working set: one Q tile, one K tile, one V tile, one score tile,
        # plus the two running statistics. Independent of n.
        c.scratch(Qi.size + 2 * Bc * d + Qi.shape[0] * Bc + 2 * Qi.shape[0])

        for j in range(0, n, Bc):
            Kj, Vj = K[j:j + Bc], V[j:j + Bc]
            c.move(Kj.size + Vj.size)         # load the key/value tile

            Sij = Qi @ Kj.T                   # (Br, Bc) -- lives in SRAM only
            c.matmul(Qi.shape[0], d, Kj.shape[0])
            if keep_full_S:
                debug_S[i:i + Br, j:j + Bc] = Sij
                c.move(Sij.size)              # ...and now you're paying again
                c.scratch(n * n)

            m_new = np.maximum(mi, Sij.max(axis=1, keepdims=True))
            P = np.exp(Sij - m_new)           # unnormalised weights
            if rescale:
                correction = np.exp(mi - m_new)
                correction[np.isnan(correction)] = 0.0   # -inf minus -inf
                li = correction * li + P.sum(axis=1, keepdims=True)
                Oi = correction * Oi + P @ Vj            # rescale the output too
            else:
                li = li + P.sum(axis=1, keepdims=True)   # THE BUG
                Oi = Oi + P @ Vj
            c.matmul(Qi.shape[0], Kj.shape[0], d)
            mi = m_new

        O[i:i + Br] = Oi / li                 # normalise once, at the end
        c.move(Oi.size)

    return (O, debug_S) if keep_full_S else O


# ---------------------------------------------------------------------------
# STAGE 5 -- the backward pass trade.
#
# Standard backward keeps the (n, n) probability matrix from the forward
# pass so it can reuse it. FlashAttention throws it away and RECOMPUTES the
# tiles from the stored m and l. That costs one extra matmul's worth of
# arithmetic and saves the entire n^2 buffer.
#
# It is exactly the trade gradient checkpointing makes, and the answer is
# the same either way.
# ---------------------------------------------------------------------------
def backward_costs(n, d):
    standard = {
        "stored_floats": n * n + 3 * n * d,   # the P matrix plus Q, K, V
        "flops": 5 * 2 * n * n * d,           # the usual backward matmuls
    }
    flash = {
        "stored_floats": 3 * n * d + 2 * n,   # Q, K, V plus m and l per row
        "flops": 6 * 2 * n * n * d,           # one extra: recompute the scores
    }
    return standard, flash


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def toy(n, d, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1, (n, d)) / np.sqrt(d),
            rng.normal(0, 1, (n, d)) / np.sqrt(d),
            rng.normal(0, 1, (n, d)))


def demo_1_where_the_time_goes():
    """Count bytes moved vs arithmetic done. The n^2 matrix dominates."""
    line("DEMO 1: attention is memory-bound, not compute-bound")

    d = 64
    print(f"head dim d = {d}, float32\n")
    print(f"{'n':>7} {'HBM MB':>10} {'GFLOPs':>9} {'bytes/flop':>11} "
          f"{'n^2 share':>10}")
    for n in (128, 512, 2048, 8192):
        c = Counters()
        Q, K, V = toy(n, d)
        standard_attention(Q, K, V, c)
        n2_bytes = 4 * n * n * BYTES         # the four n^2 trips
        print(f"{n:>7} {c.hbm_bytes / MB:>10.2f} {c.flops / 1e9:>9.3f} "
              f"{c.hbm_bytes / c.flops:>11.3f} "
              f"{100 * n2_bytes / c.hbm_bytes:>9.1f}%")

    print("\nREAD THIS: arithmetic is n^2*d, traffic is n^2. Their ratio is a")
    print("constant ~0.06 bytes per flop -- but an A100 does ~19.5 TFLOP/s of")
    print("fp32 against ~1.5 TB/s of HBM, i.e. it can only feed ~0.08 bytes per")
    print("flop. Attention sits the wrong side of that line, so the multiplying")
    print("waits on the moving. And look at the last column: by n=2048 over 98%")
    print("of everything moved is the score matrix, the one thing you never")
    print("actually wanted to keep.")


def demo_2_online_softmax():
    """The running-max trick, and what happens without the rescale."""
    line("DEMO 2: online softmax -- the whole idea, in two variables")

    rng = np.random.default_rng(1)
    row = rng.normal(0, 3, 32)

    ref = softmax(row)
    good = online_softmax(row, block=8, rescale=True)
    bad = online_softmax(row, block=8, rescale=False)

    print("one row of 32 scores, streamed in 4 blocks of 8\n")
    print(f"ordinary softmax vs online softmax, max abs diff: "
          f"{np.abs(ref - good).max():.2e}  <- exact to float precision")
    print(f"ordinary softmax vs NO-RESCALE version, max abs diff: "
          f"{np.abs(ref - bad).max():.2e}  <- wrong")
    print(f"no-rescale weights sum to {bad.sum():.6f}, not 1.0")

    print("\nfirst 6 weights, side by side:")
    print("  correct: ", np.array2string(ref[:6], precision=6))
    print("  online:  ", np.array2string(good[:6], precision=6))
    print("  broken:  ", np.array2string(bad[:6], precision=6))

    print("\nREAD THIS: the only difference between `online` and `broken` is one")
    print("line -- multiplying the running sum by exp(m_old - m_new) when a new")
    print("block raises the maximum. Without it you are adding numbers that were")
    print("scaled against different baselines, like summing a column of prices")
    print("in three currencies. Nothing errors. The answer is just wrong.")


def demo_3_tiled_attention():
    """Same output as standard attention, flat memory as n grows."""
    line("DEMO 3: tiled attention -- same answer, flat memory")

    d = 64
    print(f"{'n':>7} {'max abs diff':>14} {'std peak KB':>13} "
          f"{'flash peak KB':>15} {'std HBM MB':>12} {'flash HBM MB':>13}")
    for n in (128, 256, 512, 1024, 2048):
        Q, K, V = toy(n, d, seed=2)
        cs, cf = Counters(), Counters()
        ref = standard_attention(Q, K, V, cs)
        out = flash_attention(Q, K, V, cf, Br=64, Bc=64)
        print(f"{n:>7} {np.abs(ref - out).max():>14.2e} "
              f"{cs.peak_extra * BYTES / KB:>13.1f} "
              f"{cf.peak_extra * BYTES / KB:>15.1f} "
              f"{cs.hbm_bytes / MB:>12.2f} {cf.hbm_bytes / MB:>13.2f}")

    print("\nREAD THIS: the difference column is float noise -- this is the same")
    print("function, not an approximation of it. The standard peak grows with")
    print("n^2 (16x for every 4x in n); the flash peak does not move at all,")
    print("because the working set is two tiles and two running statistics.")
    print("That flat column is why long context became affordable.")


def demo_4_block_size():
    """There is an optimum block size, and SRAM sets it."""
    line("DEMO 4: block size -- the trade-off, and who sets it")

    n, d = 1024, 64
    Q, K, V = toy(n, d, seed=3)
    sram_budget = 96 * KB          # roughly one A100 SM's shared memory

    cs = Counters()
    standard_attention(Q, K, V, cs)
    print(f"n = {n}, d = {d}, SRAM budget assumed {sram_budget // KB} KB")
    print(f"standard attention traffic: {cs.hbm_bytes / MB:.2f} MB\n")
    print(f"{'block':>7} {'HBM MB':>10} {'tile working set KB':>21}  fits SRAM?")
    for b in (16, 32, 64, 128, 256, 512, 1024):
        c = Counters()
        flash_attention(Q, K, V, c, Br=b, Bc=b)
        ws = c.peak_extra * BYTES
        print(f"{b:>7} {c.hbm_bytes / MB:>10.2f} {ws / KB:>21.1f}  "
              f"{'yes' if ws <= sram_budget else 'NO -- spills'}")

    print("\nREAD THIS: traffic falls as the block grows, because a bigger block")
    print("means fewer passes over K and V. But the tile has to live in SRAM.")
    print("Past the budget the tile spills back to HBM and you have reinvented")
    print("standard attention with extra steps. On real hardware the ceiling is")
    print("shared memory per streaming multiprocessor -- on an A100, 164 KB, of")
    print("which a kernel can typically use around 96-100 KB. That number, not")
    print("the maths, is what picks the block size.")


def demo_5_backward():
    """Recomputation: more arithmetic, less memory. Same trade as checkpointing."""
    line("DEMO 5: the backward pass -- pay flops, save bytes")

    d = 64
    print(f"{'n':>7} {'std stored MB':>15} {'flash stored MB':>17} "
          f"{'std GFLOPs':>12} {'flash GFLOPs':>14} {'extra flops':>12}")
    for n in (512, 2048, 8192):
        s, f = backward_costs(n, d)
        print(f"{n:>7} {s['stored_floats'] * BYTES / MB:>15.2f} "
              f"{f['stored_floats'] * BYTES / MB:>17.2f} "
              f"{s['flops'] / 1e9:>12.2f} {f['flops'] / 1e9:>14.2f} "
              f"{100 * (f['flops'] / s['flops'] - 1):>11.0f}%")

    print("\nREAD THIS: FlashAttention stores only m and l per row -- two floats")
    print("-- and recomputes the score tiles during the backward pass. That is")
    print("20% more arithmetic and, at n=8192, 256MB less memory held per head.")
    print("It still runs faster in wall-clock, because the arithmetic was never")
    print("the bottleneck. This is gradient checkpointing's trade, applied to")
    print("one operator instead of a whole layer.")


def demo_6_exactness():
    """Exact, not approximate. Contrast with a method that isn't."""
    line("DEMO 6: this is not an approximation")

    n, d = 512, 64
    Q, K, V = toy(n, d, seed=5)
    c = Counters()
    ref = standard_attention(Q, K, V, Counters())
    flash = flash_attention(Q, K, V, c, Br=64, Bc=64)

    # A stand-in for the sparse/local approximations FlashAttention replaced:
    # only look at a 128-wide window. Cheap, and a different function.
    S = Q @ K.T
    band = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) <= 64
    approx = softmax(np.where(band, S, -np.inf), axis=-1) @ V

    print(f"flash vs standard, max abs diff:  {np.abs(ref - flash).max():.3e}")
    print(f"flash vs standard, allclose:      "
          f"{np.allclose(ref, flash, atol=1e-10)}")
    print(f"local-window vs standard, max abs diff: "
          f"{np.abs(ref - approx).max():.3e}")
    print(f"local-window vs standard, allclose:     "
          f"{np.allclose(ref, approx, atol=1e-10)}")

    print("\nREAD THIS: the sparse and low-rank methods that came before bought")
    print("their speed by computing a different, cheaper function and hoping the")
    print("quality survived. FlashAttention buys its speed by moving fewer bytes")
    print("while computing the identical function. You can drop it into a trained")
    print("model and the logits do not change. That is why it won.")


def demo_7_not_fewer_flops():
    """Kill the myth: the speedup is not from less arithmetic."""
    line("DEMO 7: no, it is not doing less arithmetic")

    d = 64
    print(f"{'n':>7} {'std GFLOPs':>12} {'flash GFLOPs':>14} "
          f"{'std HBM MB':>12} {'flash HBM MB':>14} {'traffic cut':>12}")
    for n in (512, 1024, 2048, 4096):
        Q, K, V = toy(n, d, seed=6)
        cs, cf = Counters(), Counters()
        standard_attention(Q, K, V, cs)
        flash_attention(Q, K, V, cf, Br=64, Bc=64)
        print(f"{n:>7} {cs.flops / 1e9:>12.3f} {cf.flops / 1e9:>14.3f} "
              f"{cs.hbm_bytes / MB:>12.2f} {cf.hbm_bytes / MB:>14.2f} "
              f"{cs.hbm_bytes / cf.hbm_bytes:>11.1f}x")

    print("\nREAD THIS: the flop columns are identical. Every multiply the")
    print("standard version does, the tiled version also does. The only column")
    print("that moves is traffic. If you take one sentence from this page into")
    print("an interview, take that one: FlashAttention is an IO result, not an")
    print("arithmetic one. The paper's own title says IO-awareness, not")
    print("efficiency.")


def demo_8_debug_matrix():
    """Keep the full score matrix 'just for debugging' -- saving gone."""
    line("DEMO 8: keeping the n x n matrix 'just for debugging'")

    n, d = 1024, 64
    Q, K, V = toy(n, d, seed=7)
    c_clean, c_dirty = Counters(), Counters()
    flash_attention(Q, K, V, c_clean, Br=64, Bc=64)
    flash_attention(Q, K, V, c_dirty, Br=64, Bc=64, keep_full_S=True)

    print(f"tiled, matrix discarded: peak extra "
          f"{c_clean.peak_extra * BYTES / KB:8.1f} KB, "
          f"traffic {c_clean.hbm_bytes / MB:.2f} MB")
    print(f"tiled, matrix kept:      peak extra "
          f"{c_dirty.peak_extra * BYTES / KB:8.1f} KB, "
          f"traffic {c_dirty.hbm_bytes / MB:.2f} MB")
    print(f"memory saving lost: "
          f"{c_dirty.peak_extra / c_clean.peak_extra:.0f}x worse peak")

    print("\nREAD THIS: the tiling did not save you anything. The saving was")
    print("never in the loop structure -- it was in never allocating the n x n")
    print("buffer. One debug line puts it straight back.")


def demo_9_no_running_max():
    """Skip the running max entirely: overflow, exactly as on the attention page."""
    line("DEMO 9: skip the running max -- exp() overflows to inf")

    scores = np.array([800.0, 810.0, 805.0, 790.0])
    print("scores:", scores)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.exp(scores)
        naive = raw / raw.sum()
    print("exp(scores) with no max subtracted:", raw)
    print("resulting weights:                 ", naive)
    print("safe softmax (max subtracted):     ",
          np.array2string(softmax(scores), precision=6))
    print("\nREAD THIS: same failure as the attention page, one level down. In")
    print("FlashAttention the max is not a single number you can compute up")
    print("front -- you only ever see a block at a time -- which is precisely")
    print("why the running max exists. It is overflow safety that had to learn")
    print("to work incrementally.")


if __name__ == "__main__":
    demo_1_where_the_time_goes()
    demo_2_online_softmax()
    demo_3_tiled_attention()
    demo_4_block_size()
    demo_5_backward()
    demo_6_exactness()
    demo_7_not_fewer_flops()
    demo_8_debug_matrix()
    demo_9_no_running_max()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. the cost   = moving the n x n score matrix to HBM and back, not the
                  multiplying. Attention is memory-bound.
  2. the fix    = tile Q into row blocks and K/V into column blocks, so only
                  a (Br, Bc) tile ever exists, and it lives in SRAM.
  3. the maths  = online softmax: carry a running max m and a running sum l,
                  and rescale l AND the output O whenever m changes.
  4. backward   = don't store the n x n probabilities; recompute the tiles
                  from m and l. More flops, far less memory, still faster.
  5. exactness  = same function, bit-for-bit-ish. Not a sparse or low-rank
                  approximation. That is the point.
  6. flash-2    = same idea, better work partitioning across warps and
                  thread blocks, fewer non-matmul operations.

  What is NOT in the idea: fewer flops, a different attention pattern, any
  loss of quality. If an explanation tells you FlashAttention approximates
  attention, close the tab.
""")
