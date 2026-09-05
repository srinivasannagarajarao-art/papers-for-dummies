"""
Quantisation -- for programmers, not researchers.

Run it:      python3 quantization_from_scratch.py
Debug it:    put a breakpoint in quantize_symmetric and watch the round().

No torch. No training. Quantisation is arithmetic: pick a scale, divide,
round, clamp, store, multiply back. Every core function is under 15 lines.

The papers stitched together here:
  LLM.int8()  Dettmers et al. 2022   arXiv 2208.07339
  GPTQ        Frantar et al. 2022    arXiv 2210.17323
  AWQ         Lin et al. 2023        arXiv 2306.00978
  QLoRA/NF4   Dettmers et al. 2023   arXiv 2305.14314
"""

import numpy as np
from statistics import NormalDist

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=80)


# ---------------------------------------------------------------------------
# STAGE 1 -- the whole operation, symmetric. Six lines.
#
# int8 gives you 255 usable slots from -127..127. A scale maps your float
# range onto those slots. Dividing by the scale is "how many slots is this
# number worth"; rounding is where the information dies.
# ---------------------------------------------------------------------------
def quantize_symmetric(x, bits=8):
    qmax = 2 ** (bits - 1) - 1              # 127 for int8
    scale = np.abs(x).max() / qmax          # one float per tensor
    q = np.clip(np.round(x / scale), -qmax, qmax)   # the lossy step
    return q.astype(np.int32), scale


def dequantize_symmetric(q, scale):
    return q * scale                        # back to float. Not the original.


# ---------------------------------------------------------------------------
# STAGE 2 -- asymmetric, with a zero point. Symmetric wastes half its range
# on data that never goes negative (a ReLU output, say). Asymmetric shifts
# the grid with an integer offset so both ends of the range are used.
# ---------------------------------------------------------------------------
def quantize_asymmetric(x, bits=8):
    qmin, qmax = 0, 2 ** bits - 1           # 0..255 for uint8
    scale = (x.max() - x.min()) / (qmax - qmin)
    zero_point = np.round(qmin - x.min() / scale)   # integer, not a float
    q = np.clip(np.round(x / scale + zero_point), qmin, qmax)
    return q.astype(np.int32), scale, zero_point


def dequantize_asymmetric(q, scale, zero_point):
    return (q - zero_point) * scale


# ---------------------------------------------------------------------------
# STAGE 3 -- where the scale lives. One scale for the whole tensor means the
# loudest channel sets the resolution for every quiet one. A scale per column
# costs one float per column and fixes it.
# ---------------------------------------------------------------------------
def quantize_per_channel(W, bits=8, axis=0):
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(W).max(axis=axis, keepdims=True) / qmax   # one per column
    q = np.clip(np.round(W / scale), -qmax, qmax)
    return q.astype(np.int32), scale


# ---------------------------------------------------------------------------
# STAGE 4 -- group-wise scales. Between "one scale" and "one per column":
# chop each column into groups of g values and give each group its own scale.
# GPTQ and AWQ both ship with g=128 as the default for a reason.
# ---------------------------------------------------------------------------
def quantize_grouped(x, bits=4, group=128):
    flat = x.reshape(-1, group)                     # rows of g values
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(flat).max(axis=1, keepdims=True) / qmax
    q = np.clip(np.round(flat / scale), -qmax, qmax)
    return (q * scale).reshape(x.shape), scale.size  # recovered, n_scales


# ---------------------------------------------------------------------------
# STAGE 5 -- the levels themselves. Linear int4 spaces 16 levels evenly.
# NF4 (QLoRA) puts the 16 levels at quantiles of a normal distribution,
# because that is what weights actually look like: crowded near zero.
#
# The real NF4 table is built asymmetrically so that exact 0 is a level;
# this is the same construction idea in five lines.
# ---------------------------------------------------------------------------
def linear_levels(bits=4):
    n = 2 ** bits
    return np.linspace(-1.0, 1.0, n)                # 16 evenly spaced levels


def normal_levels(bits=4):
    n = 2 ** bits
    ps = [(i + 0.5) / n for i in range(n)]          # equal probability mass
    lv = np.array([NormalDist().inv_cdf(p) for p in ps])
    return lv / np.abs(lv).max()                    # squash into [-1, 1]


def quantize_to_levels(x, levels):
    """Snap each value to the nearest level, after absmax-scaling to [-1, 1]."""
    scale = np.abs(x).max()
    idx = np.abs(x[:, None] / scale - levels[None, :]).argmin(axis=1)
    return levels[idx] * scale


# ---------------------------------------------------------------------------
# STAGE 6 -- LLM.int8()'s fix: find the columns that hold outliers, keep those
# in 16-bit, quantise the other 99-odd percent to int8. Mixed precision, one
# matrix multiply split in two.
# ---------------------------------------------------------------------------
def outlier_columns(X, threshold=6.0):
    return np.where(np.abs(X).max(axis=0) >= threshold)[0]


def quantize_mixed(X, outlier_cols, bits=8):
    keep = np.zeros(X.shape[1], dtype=bool)
    keep[outlier_cols] = True
    out = X.copy()                                  # outlier cols stay fp16-ish
    rest = X[:, ~keep]
    q, s = quantize_symmetric(rest, bits)           # the rest goes to int8
    out[:, ~keep] = dequantize_symmetric(q, s)
    return out


def rel_err(x, xq):
    """One honest error number: RMS error as a fraction of the RMS signal."""
    return float(np.sqrt(np.mean((x - xq) ** 2)) / np.sqrt(np.mean(x ** 2)))


def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ===========================================================================
# DEMOS
# ===========================================================================

def demo_1_the_operation():
    line("DEMO 1: the whole operation -- divide, round, clamp, store, multiply")

    x = np.array([-0.83, -0.20, 0.00, 0.11, 0.47, 1.60, 0.05, -0.62])
    q, s = quantize_symmetric(x, 8)
    xq = dequantize_symmetric(q, s)

    print(f"scale = max|x| / 127 = {np.abs(x).max():.2f} / 127 = {s:.8f}")
    print("original :", x)
    print("as int8  :", q)
    print("recovered:", np.round(xq, 5))
    print("error    :", np.round(xq - x, 5))
    print(f"\nsymmetric  relative error: {rel_err(x, xq):.6f}")

    # Same data, shifted so it is not centred on zero -- a ReLU output.
    y = x + 1.0
    qs, ss = quantize_symmetric(y, 8)
    qa, sa, zp = quantize_asymmetric(y, 8)
    ya_s = dequantize_symmetric(qs, ss)
    ya_a = dequantize_asymmetric(qa, sa, zp)
    print("\nnow shift the data to [0.17, 2.60] -- nothing is negative:")
    print(f"  symmetric  relative error: {rel_err(y, ya_s):.6f}  (half the grid unused)")
    print(f"  asymmetric relative error: {rel_err(y, ya_a):.6f}  zero_point={zp:.0f}")
    print(f"  symmetric  int8 codes: {qs.min()} .. {qs.max()} "
          f"(of -127..127, {100 * (qs.max() - qs.min() + 1) // 255}% of the grid used)")
    print(f"  asymmetric int8 codes: {qa.min()} .. {qa.max()} "
          f"(of 0..255, {100 * (qa.max() - qa.min() + 1) // 256}% used)")

    print("\nREAD THIS: quantisation is lossy compression of the weights.")
    print("The scale is the only thing you get to choose. Choose it badly and")
    print("you throw away the wrong bits -- exactly like a JPEG quality slider.")


def demo_2_per_tensor_vs_per_channel():
    line("DEMO 2: one scale for the tensor vs one scale per channel")

    rng = np.random.default_rng(1)
    d = 64
    W = rng.normal(0, 1, (d, 8))
    W[:, 0] *= 50.0          # one very loud channel
    W[:, 1] *= 10.0
    W[:, 7] *= 0.02          # one very quiet channel

    qt, st = quantize_symmetric(W, 8)
    Wt = dequantize_symmetric(qt, st)
    qc, sc = quantize_per_channel(W, 8, axis=0)
    Wc = qc * sc

    print(f"channel absmax: {np.abs(W).max(axis=0).round(3)}")
    print(f"\nper-tensor  one scale  = {st:.6f}")
    print(f"per-channel scales     = {sc.ravel().round(6)}")
    print(f"\nper-tensor  relative error: {rel_err(W, Wt):.6f}")
    print(f"per-channel relative error: {rel_err(W, Wc):.6f}")

    print("\nper-channel error, column by column:")
    for j in range(8):
        et = rel_err(W[:, j], Wt[:, j])
        ec = rel_err(W[:, j], Wc[:, j])
        print(f"  col {j}: per-tensor {et:9.6f}   per-channel {ec:.6f}")

    print(f"\ndistinct int8 codes in the quietest column (col 7):")
    print(f"  per-tensor : {len(np.unique(qt[:, 7]))} of 255")
    print(f"  per-channel: {len(np.unique(qc[:, 7]))} of 255")
    print("\nREAD THIS: this is the single most useful fact on the page. One")
    print("scale for a whole tensor lets the loudest channel decide the")
    print("resolution of every quiet one. Extra storage for per-channel: one")
    print(f"float per column, {sc.size} floats for a {W.shape[0]}x{W.shape[1]} matrix.")


def demo_3_outliers():
    line("DEMO 3: the outlier problem -- LLM.int8()'s finding")

    rng = np.random.default_rng(2)
    n_tok, d = 32, 16
    X = rng.normal(0, 1, (n_tok, d))
    clean = X.copy()

    qc, scl = quantize_symmetric(clean, 8)
    print(f"clean activations, absmax {np.abs(clean).max():.3f}")
    print(f"  per-tensor int8 relative error: {rel_err(clean, dequantize_symmetric(qc, scl)):.6f}")

    # Inject outliers in two feature dimensions. This is what LLM.int8()
    # observed in real transformers past roughly 6.7B parameters.
    X[:, 3] *= 40.0
    X[:, 11] *= 25.0

    q, s = quantize_symmetric(X, 8)
    Xq = dequantize_symmetric(q, s)
    print(f"\nwith outliers in dims 3 and 11, absmax {np.abs(X).max():.3f}")
    print(f"  per-tensor scale went {scl:.6f} -> {s:.6f}  ({s / scl:.1f}x coarser)")
    print(f"  per-tensor int8 relative error: {rel_err(X, Xq):.6f}")

    normal_dims = [j for j in range(d) if j not in (3, 11)]
    print(f"  error on the NON-outlier dims only: "
          f"{rel_err(X[:, normal_dims], Xq[:, normal_dims]):.6f}")
    print(f"  distinct int8 codes used by dim 0: {len(np.unique(q[:, 0]))} of 255")

    cols = outlier_columns(X, threshold=6.0)
    Xm = quantize_mixed(X, cols)
    print(f"\nmixed precision: keep columns {cols.tolist()} in 16-bit,")
    print(f"  that is {len(cols)}/{d} = {100 * len(cols) / d:.1f}% of the dimensions")
    print(f"  relative error, whole tensor : {rel_err(X, Xm):.6f}")
    print(f"  relative error, non-outlier  : "
          f"{rel_err(X[:, normal_dims], Xm[:, normal_dims]):.6f}")

    print("\nREAD THIS: one outlier does not damage itself, it damages")
    print("everything else. The scale is set by the max, so a single huge")
    print("value spends the whole int8 grid on empty space. LLM.int8() found")
    print("these outlier features are systematic in large transformers, and")
    print("that holding a tiny fraction of dimensions in 16-bit fixes it.")


def demo_4_memory():
    line("DEMO 4: the memory arithmetic -- the number that decides everything")

    GB = 1024 ** 3
    models = [("7B", 7e9), ("13B", 13e9), ("70B", 70e9)]
    print("weight memory, params x bytes-per-param:\n")
    print(f"{'model':>6}  {'fp16':>10}  {'int8':>10}  {'int4':>10}")
    for name, p in models:
        print(f"{name:>6}  {p * 2 / GB:8.1f} GB  {p * 1 / GB:8.1f} GB  "
              f"{p * 0.5 / GB:8.1f} GB")

    print("\nfits on the card? weights only, nothing left for activations:\n")
    print(f"{'model':>6}  {'fp16 24/80':>12}  {'int8 24/80':>12}  {'int4 24/80':>12}")
    for name, p in models:
        row = []
        for bpp in (2, 1, 0.5):
            g = p * bpp / GB
            row.append(f"{'yes' if g < 24 else 'NO':>4} "
                       f"{'yes' if g < 80 else 'NO':>4}")
        print(f"{name:>6}  " + "  ".join(f"{c:>12}" for c in row))

    # KV cache, Llama-2 shapes. bytes = 2 (K and V) * layers * kv_heads
    # * head_dim * 2 bytes (fp16) per token. See ../kv-cache/ for why.
    print("\nnow add the KV cache at 4096 tokens, one sequence, fp16")
    print("(Llama-2 shapes; 70B uses grouped-query attention, 8 KV heads):\n")
    cfg = [("7B", 7e9, 32, 32, 128), ("13B", 13e9, 40, 40, 128),
           ("70B", 70e9, 80, 8, 128)]
    ctx = 4096
    for name, p, layers, kv_heads, hd in cfg:
        kv = 2 * layers * kv_heads * hd * 2 * ctx / GB
        for label, bpp in (("fp16", 2), ("int4", 0.5)):
            w = p * bpp / GB
            print(f"  {name:>4} {label}: weights {w:6.1f} GB + KV {kv:5.2f} GB "
                  f"= {w + kv:6.1f} GB   24GB card: "
                  f"{'fits' if w + kv < 24 else 'no'}")

    print("\nREAD THIS: this is the whole reason quantisation exists. A 70B in")
    print("fp16 is 130 GB and needs two 80GB cards. At int4 it is 32.6 GB and")
    print("fits on one. A 13B at int4 is 6.1 GB and fits on a laptop GPU. The")
    print("KV cache is a separate budget that grows with context, not weights.")


def demo_5_nf4_vs_int4():
    line("DEMO 5: NF4 vs linear int4 -- put the levels where the data is")

    rng = np.random.default_rng(3)
    w = rng.normal(0, 1, 100000)

    lin = linear_levels(4)
    nf4 = normal_levels(4)
    print("linear int4 levels (evenly spaced):")
    print(" ", np.round(lin, 4))
    print("normal-quantile levels (NF4-style):")
    print(" ", np.round(nf4, 4))

    w_lin = quantize_to_levels(w, lin)
    w_nf4 = quantize_to_levels(w, nf4)
    print(f"\nweights ~ N(0,1), n={w.size}, absmax {np.abs(w).max():.3f}")
    print(f"  linear int4 relative error: {rel_err(w, w_lin):.6f}")
    print(f"  NF4-style   relative error: {rel_err(w, w_nf4):.6f}")
    print(f"  improvement: {100 * (1 - rel_err(w, w_nf4) / rel_err(w, w_lin)):.1f}%")

    print("\nhow much data each of the 16 levels actually gets:")
    for name, lv, wq in (("linear", lin, w_lin), ("NF4   ", nf4, w_nf4)):
        counts = np.array([int((wq == lv_i * np.abs(w).max()).sum()) for lv_i in lv])
        print(f"  {name}: emptiest level {counts.min():6d}  busiest {counts.max():6d}"
              f"  levels under 1% of the data: "
              f"{int((counts < 0.01 * w.size).sum())}/16")

    print("\nREAD THIS: weights are not spread evenly, they are piled up near")
    print("zero. Evenly spaced levels put most of the resolution out in the")
    print("tails where almost nothing lives. QLoRA's NF4 puts each level at an")
    print("equal-probability quantile instead, so every level earns its keep.")


def demo_6_group_size():
    line("DEMO 6: group size -- error against the cost of storing scales")

    rng = np.random.default_rng(4)
    n = 4096
    w = rng.normal(0, 1, n)
    w[rng.integers(0, n, 8)] *= 30.0        # a handful of large weights

    print(f"{n} weights, 8 of them 30x larger than the rest, int4\n")
    print(f"{'group':>8}  {'rel error':>10}  {'scales':>7}  {'bits/weight':>12}")
    for g in (n, 128, 64, 32):
        wq, n_scales = quantize_grouped(w, bits=4, group=g)
        overhead = 16 * n_scales / n        # one fp16 scale per group
        label = "whole" if g == n else str(g)
        print(f"{label:>8}  {rel_err(w, wq):10.6f}  {n_scales:7d}  "
              f"{4 + overhead:12.3f}")

    print("\nsame sweep with NO outliers, so you can see what the outliers cost:")
    w2 = rng.normal(0, 1, n)
    for g in (n, 128, 64, 32):
        wq, _ = quantize_grouped(w2, bits=4, group=g)
        label = "whole" if g == n else str(g)
        print(f"{label:>8}  {rel_err(w2, wq):10.6f}")

    print("\nREAD THIS: a group is just a smaller blast radius for one outlier.")
    print("g=128 is the usual default because it buys most of the accuracy for")
    print("0.125 extra bits per weight. QLoRA's double quantisation then")
    print("quantises those scales too, saving about 0.37 bits per weight.")


def demo_7_you_cannot_train_on_a_grid():
    line("DEMO 7: why QLoRA needs LoRA -- you cannot fine-tune on a grid")

    rng = np.random.default_rng(5)
    w = rng.normal(0, 1, 8)
    q, s = quantize_symmetric(w, 4)
    wq = dequantize_symmetric(q, s)
    print(f"4-bit grid step (the scale) = {s:.6f}")
    print("quantised weights:", np.round(wq, 5))

    grad = rng.normal(0, 1, 8)
    lr = 1e-3
    updated = wq - lr * grad
    q2, _ = quantize_symmetric(updated, 4)
    # re-quantise onto the SAME grid, which is what a 4-bit store forces
    re_q = np.clip(np.round(updated / s), -7, 7) * s
    print(f"\nafter one SGD step at lr={lr}:")
    print("  moved by         :", np.round(updated - wq, 6))
    print("  after re-quantise:", np.round(re_q - wq, 6))
    print(f"  weights that actually changed: {int((re_q != wq).sum())} of 8")

    lr_big = 0.5
    re_q_big = np.clip(np.round((wq - lr_big * grad) / s), -7, 7) * s
    print(f"\nsame gradient at lr={lr_big}: "
          f"{int((re_q_big != wq).sum())} of 8 weights changed")
    print("  step sizes taken :", np.round(re_q_big - wq, 5))

    # The QLoRA answer: leave the base frozen on its grid, add a small
    # higher-precision adapter that CAN take a small step. See ../lora/.
    A = np.zeros((8, 2))
    B = rng.normal(0, 0.02, (2, 1))
    delta = (A @ B).ravel()
    A = A - lr * np.outer(grad, B.ravel())   # one step on the adapter only
    delta_after = (A @ B).ravel()
    print(f"\nfrozen 4-bit base + fp16 LoRA adapter, same lr={lr}:")
    print("  adapter output before: all zeros (A starts at 0, so it is")
    print("                         a no-op until the first step)")
    print(f"  adapter output after : max |delta| = "
          f"{np.abs(delta_after).max():.3e}")
    print(f"  weights that actually changed: "
          f"{int((delta_after != delta).sum())} of 8, by steps smaller than")
    print(f"  the 4-bit grid step of {s:.6f}")

    print("\nREAD THIS: a 4-bit weight lives on 15 allowed values. A gradient")
    print("step smaller than half the grid step rounds straight back to where")
    print("it started -- the update silently disappears. Crank the learning")
    print("rate until it does move and every step is a whole grid step, which")
    print("is not fine-tuning, it is vandalism. QLoRA's answer: freeze the")
    print("4-bit base, train a small fp16 LoRA adapter on top of it.")


if __name__ == "__main__":
    demo_1_the_operation()
    demo_2_per_tensor_vs_per_channel()
    demo_3_outliers()
    demo_4_memory()
    demo_5_nf4_vs_int4()
    demo_6_group_size()
    demo_7_you_cannot_train_on_a_grid()

    line("THE WHOLE FAMILY, COMPRESSED")
    print("""
  the operation  q = round(x / scale); x' = q * scale. That is all of it.
  the question   which numbers am I allowed to be sloppy about?

  LLM.int8()  outlier features are systematic past roughly 6.7B params.
              Keep those few dimensions in 16-bit, the rest in int8.
  GPTQ        one-shot post-training, weights only. Quantise column by
              column, using approximate second-order (Hessian) info to
              correct the not-yet-quantised weights as it goes.
  AWQ         a small fraction of weights are salient. Find them by
              activation magnitude, not weight magnitude, and protect
              them by scaling. No backprop, no reconstruction data needed.
  QLoRA       4-bit NormalFloat levels, double quantisation of the
              scales, paged optimisers. Train a LoRA adapter over a
              frozen 4-bit base.

  Weight-only quantisation shrinks the checkpoint. Activation quantisation
  is what makes the matmul itself cheaper, and it is much harder, because
  activations are where the outliers live.
""")
