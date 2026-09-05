"""
RoFormer: Rotary Position Embedding (RoPE) -- for programmers, not researchers.

Run it:      python3 rope_from_scratch.py
Debug it:    set a breakpoint in rotate_pairs and watch one pair of dims turn.

No torch. No autograd. No training. Just dot products, printed, so you can
SEE that "the score only depends on the gap" is a real, testable claim.
Every function is <15 lines.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)


# STAGE 0 -- softmax, copied from the attention page. Row -> percentages.
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


# STAGE 1 -- the tick rates. One per PAIR of dims, so d/2 of them:
#     theta_i = base^(-2i/d),   i = 0 .. d/2 - 1
# Pair 0 turns 1 radian per position (the fast hand); the last pair about
# 1/base (the slow hand). Same frequencies as the 2017 sinusoidal encoding.
# RoPE reuses them and does something different with them.
def rope_freqs(d, base=10000.0):
    i = np.arange(d // 2)
    return base ** (-2.0 * i / d)


# STAGE 2 -- THE WHOLE PAPER. Turn each 2D pair of dims by its own angle:
#     [x1']   [cos a  -sin a] [x1]
#     [x2'] = [sin a   cos a] [x2]
# Pair (0,1) gets angles[0], pair (2,3) gets angles[1], and so on. That is
# the paper's block-diagonal matrix R, never materialised: just a cos/sin
# multiply. No parameters. Nothing learned.
def rotate_pairs(x, angles):
    """x: (d,), angles: (d/2,). Pair i = dims (2i, 2i+1), turned by angles[i]."""
    c, s = np.cos(angles), np.sin(angles)
    x1, x2 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x1 * c - x2 * s
    out[1::2] = x1 * s + x2 * c
    return out


def rope(x, pos, base=10000.0):
    """Rotary embedding of vector x at integer position pos. Same shape out."""
    return rotate_pairs(x, pos * rope_freqs(x.shape[-1], base))


# The OTHER pairing convention. Hugging Face's LLaMA pairs dim i with dim
# i + d/2 (its `rotate_half`); Meta's original code pairs (0,1), (2,3), ...
# Each is fine alone. Mixed, the dims are scrambled -- which is why the HF
# weight-conversion script permutes W_q and W_k.
def rope_half(x, pos, base=10000.0):
    d = x.shape[-1]
    angles = pos * rope_freqs(d, base)
    c, s = np.cos(angles), np.sin(angles)
    x1, x2 = x[: d // 2], x[d // 2 :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c])


# The 2017 scheme, for comparison: a position VECTOR that gets ADDED. Same
# tick rates; sin in even dims, cos in odd. This is one row of the attention
# page's positional_encoding().
def sinusoidal_pe(pos, d, base=10000.0):
    angles = pos * rope_freqs(d, base)
    pe = np.empty(d)
    pe[0::2] = np.sin(angles)
    pe[1::2] = np.cos(angles)
    return pe


# STAGE 3 -- attention with RoPE. The attention page's function plus two
# lines: turn each row of Q and K by its own position, right after the
# projections and before the dot product. V is left alone.
def attention_with_rope(Q, K, V, positions, base=10000.0):
    Qr = np.stack([rope(q, m, base) for q, m in zip(Q, positions)])
    Kr = np.stack([rope(k, n, base) for k, n in zip(K, positions)])
    scores = Qr @ Kr.T / np.sqrt(Q.shape[-1])
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_1_clock_hands():
    """One hand first, then the paper's d/2 hands at their real speeds."""
    line("DEMO 1: RoPE is a clock. Each pair of dims is one hand.")

    # ONE hand: a 2D vector, turned by pos * theta. theta = 30 deg so the
    # numbers are recognisable.
    theta = np.pi / 6
    v = np.array([1.0, 0.0])
    print("one hand, 30 deg per position, starting at", v)
    for pos in range(4):
        r = rotate_pairs(v, np.array([pos * theta]))
        print(f"  pos {pos}: turned {np.degrees(pos * theta):5.1f} deg -> {r}"
              f"   length {np.linalg.norm(r):.4f}")

    # d/2 hands, each with its own speed, from the paper's frequency set.
    d = 8
    theta = rope_freqs(d)
    print(f"\nd = {d}, so {d // 2} hands. theta_i = 10000^(-2i/d) =", theta)
    for pos in (1, 10):
        print(f"\n  at position {pos}, each hand has turned:")
        for i, t in enumerate(theta):
            a = pos * t
            print(f"    pair {i}: {a:8.4f} rad = {np.degrees(a):8.2f} deg"
                  f" = {a / (2 * np.pi):6.4f} turns")

    print("\nREAD THIS: rotation never changes a vector's length, only its")
    print("direction. Pair 0 has lapped the clock by position 10 (1.59 turns).")
    print("Pair 3 has barely moved (0.0016 turns). Fast hands resolve nearby")
    print("positions; slow hands tell position 10 from position 1000.")


def demo_2_relative_position():
    """THE claim: q_m . k_n depends on m - n and nothing else."""
    line("DEMO 2: the dot product only sees the GAP between positions")

    d = 8
    rng = np.random.default_rng(1)
    q = rng.normal(0, 1, d)
    k = rng.normal(0, 1, d)

    print("same q and k, placed at (3, 7) and then at (13, 17). Gap = 4 both times.")
    a = rope(q, 3) @ rope(k, 7)
    b = rope(q, 13) @ rope(k, 17)
    print(f"\nRoPE, rotate:      q@3  . k@7  = {a:.12f}")
    print(f"                   q@13 . k@17 = {b:.12f}")
    print(f"                   difference  = {abs(a - b):.1e}   <- float noise")

    c = (q + sinusoidal_pe(3, d)) @ (k + sinusoidal_pe(7, d))
    e = (q + sinusoidal_pe(13, d)) @ (k + sinusoidal_pe(17, d))
    print(f"\n2017, add:         q@3  . k@7  = {c:.12f}")
    print(f"                   q@13 . k@17 = {e:.12f}")
    print(f"                   difference  = {abs(c - e):.1e}   <- absolute position leaks")

    print(f"\nunrotated q . k             = {q @ k:.12f}")
    print(f"gap 0, e.g. q@5 . k@5       = {rope(q, 5) @ rope(k, 5):.12f}   <- identical")

    print("\nREAD THIS: turning q by m and k by n, then dotting, is the same as")
    print("turning k by (n - m) and dotting with the original q. Rotations")
    print("compose by adding angles: R(m)^T R(n) = R(n - m). That one identity")
    print("is the paper. The add scheme has cross terms q.pe[n] and pe[m].k that")
    print("carry absolute position, so the score moves when the sentence does.")


def demo_3_decay_with_distance():
    """The paper's long-term decay figure, reproduced as numbers."""
    line("DEMO 3: far-apart tokens score lower -- on average")

    d = 64
    q = np.ones(d)          # all ones, so the sin cross terms cancel and
    k = np.ones(d)          # the curve is the pure sum of cos(gap * theta_i)
    gaps = np.arange(65)
    dots = np.array([rope(q, 0) @ rope(k, g) for g in gaps])

    print(f"q = k = all ones, d = {d}. dot(q at 0, k at gap):")
    for g in (0, 1, 2, 3, 4, 6, 7, 8, 12, 16, 24, 32, 48, 64):
        print(f"  gap {g:2d}: {dots[g]:8.3f}")

    print("\nmean dot over gap buckets (the trend):")
    for lo in range(0, 64, 8):
        print(f"  gaps {lo:2d}-{lo + 7:2d}: {dots[lo:lo + 8].mean():8.3f}")

    rises = [g for g in range(64) if dots[g + 1] > dots[g]]
    print(f"\nnot monotonic: the dot RISES at {len(rises)} of the 64 steps, e.g.")
    for g in rises[:3]:
        print(f"  gap {g:2d} -> {g + 1:2d}: {dots[g]:.3f} -> {dots[g + 1]:.3f}")

    print("\nREAD THIS: at gap 0 every hand lines up and you get d back. As the")
    print("gap grows, the fast hands point in random directions and cancel;")
    print("only the slow hands still agree. The paper's long-term decay figure")
    print("plots an upper bound on this sum; it has the same shape. It is a")
    print("trend, not a guarantee: the wiggles are real.")


def demo_4_only_q_and_k():
    """Rotate q and k. Never v. And never after the softmax."""
    line("DEMO 4: rotate q and k. Never v. Never after the softmax.")

    n, d = 5, 8
    rng = np.random.default_rng(2)
    Q = rng.normal(0, 1, (n, d))
    K = rng.normal(0, 1, (n, d))
    V = rng.normal(0, 1, (n, d))
    pos_a = np.arange(n)          # the 5 tokens at positions 0..4
    pos_b = np.arange(n) + 100    # the same 5 tokens at positions 100..104

    out_a, w_a = attention_with_rope(Q, K, V, pos_a)
    out_b, w_b = attention_with_rope(Q, K, V, pos_b)
    print("same 5 tokens at positions 0-4, then at 100-104")
    print(f"  CORRECT (q, k only):  weights max|diff| = {np.abs(w_a - w_b).max():.1e}")
    print(f"                        output  max|diff| = {np.abs(out_a - out_b).max():.1e}"
          "   <- shift-invariant")

    # BREAK 1: rotate V as well.
    Vr_a = np.stack([rope(v, m) for v, m in zip(V, pos_a)])
    Vr_b = np.stack([rope(v, m) for v, m in zip(V, pos_b)])
    outv_a, _ = attention_with_rope(Q, K, Vr_a, pos_a)
    outv_b, _ = attention_with_rope(Q, K, Vr_b, pos_b)
    print(f"  BROKEN (v too):       output  max|diff| = {np.abs(outv_a - outv_b).max():.4f}"
          "   <- moves with the sentence")
    print(f"                        vs correct output = {np.abs(outv_a - out_a).max():.4f}")
    print("  token 0, correct:", np.round(out_a[0], 3))
    print("  token 0, v too:  ", np.round(outv_a[0], 3))

    # BREAK 2: no rotation on q, k; rotate the OUTPUT instead.
    w_plain = softmax(Q @ K.T / np.sqrt(d))
    out_after = np.stack([rope(o, m) for o, m in zip(w_plain @ V, pos_a)])
    print("\n  BROKEN (after softmax):")
    print(f"    weights vs plain attention, no position at all: max|diff| ="
          f" {np.abs(w_plain - w_plain).max():.4f}")
    print(f"    weights vs the correct RoPE weights:            max|diff| ="
          f" {np.abs(w_plain - w_a).max():.4f}")
    print("    token 0, after:  ", np.round(out_after[0], 3))

    print("\nREAD THIS: the property lives in the SCORE q.k, so the rotation")
    print("goes on q and k, right after W_q and W_k, before the dot product.")
    print("V is the content being blended; turn it and each token's output")
    print("depends on where the sentence starts. Rotate after the softmax and")
    print("the weights were computed blind -- you are back to the attention")
    print("page's 'cat ate food' == 'food ate cat' problem, plus a scramble.")


def demo_5_base_scaling():
    """Why changing 10000 -> 100000 is a context-extension trick."""
    line("DEMO 5: change the base, slow the clock -- context extension")

    d = 128             # LLaMA's per-head dim
    pos = 4096
    for base in (10000.0, 100000.0):
        th = rope_freqs(d, base)
        slow, fast = pos * th[-1], pos * th[0]
        print(f"base {base:>7.0f}: slowest hand theta = {th[-1]:.3e} rad/pos")
        print(f"              at pos {pos}: {slow:.4f} rad = {np.degrees(slow):5.2f} deg"
              f" = {slow / (2 * np.pi):.4f} turns")
        print(f"              fastest hand at pos {pos}: {fast:.0f} rad"
              f" = {fast / (2 * np.pi):.1f} turns, any base")

    th4 = rope_freqs(d, 10000.0)[-1]
    th5 = rope_freqs(d, 100000.0)[-1]
    print(f"\nslow hand, base 100000: reaches the base-10000 pos-4096 angle"
          f" at pos {pos * th4 / th5:.0f}")
    print(f"position interpolation instead: feed pos/8, so 4096 -> {pos / 8:.0f},"
          f" slow hand {pos / 8 * th4:.4f} rad")
    print("  but EVERY hand is slowed 8x, the fast ones too")

    print("\nREAD THIS: a model trained to 4096 has only ever seen the slow hand")
    print("between 0 and 27 degrees. Ask for position 8192 and the slow hand")
    print("points somewhere it has never pointed. Raise the base and the slow")
    print("hands slow down (the fast ones barely change), so 8192 lands inside")
    print("the range training covered. That is the whole NTK-aware trick.")


def demo_6_pairing_conventions():
    """Adjacent pairs vs half-split pairs. Both fine. Mixed: scrambled."""
    line("DEMO 6: two pairing conventions -- either is fine, mixing is not")

    d = 8
    rng = np.random.default_rng(3)
    q = rng.normal(0, 1, d)
    k = rng.normal(0, 1, d)

    print("dot product at (3, 7) and at (13, 17), gap 4 both times:\n")
    cases = [("adjacent q, adjacent k", rope, rope),
             ("half q,     half k    ", rope_half, rope_half),
             ("adjacent q, half k    ", rope, rope_half)]
    for name, fq, fk in cases:
        a = fq(q, 3) @ fk(k, 7)
        b = fq(q, 13) @ fk(k, 17)
        tag = "relative" if abs(a - b) < 1e-9 else "SCRAMBLED"
        print(f"  {name}:  {a:9.5f}   {b:9.5f}   |diff| {abs(a - b):.1e}  {tag}")

    print("\nREAD THIS: adjacent pairing gives one number, half-split another,")
    print("and each is stable under a shift, so each is a valid RoPE. The")
    print("learned W_q, W_k absorb the difference. Mix them and dim 1 is turned")
    print("by theta_0 on one side and theta_1 on the other: the gap identity")
    print("no longer holds. Hugging Face LLaMA uses half-split (rotate_half).")


def demo_7_one_frequency():
    """A clock with only second hands cannot tell 1 o'clock from 2."""
    line("DEMO 7: one tick rate for every pair -- positions alias")

    d = 8
    rng = np.random.default_rng(4)
    q = rng.normal(0, 1, d)
    k = rng.normal(0, 1, d)

    one_rate = np.full(d // 2, 2 * np.pi / 8)     # every hand ticks 45 deg
    print("every pair turns 45 deg per position (period: 8 positions):")
    for gap in (1, 9, 17, 2, 10):
        a = rotate_pairs(q, 0 * one_rate) @ rotate_pairs(k, gap * one_rate)
        print(f"  gap {gap:2d}: {a:.6f}")

    print("\nthe paper's frequencies, same q, k:")
    for gap in (1, 9, 17):
        print(f"  gap {gap:2d}: {rope(q, 0) @ rope(k, gap):.6f}")

    print("\nREAD THIS: with one rate, gap 1 and gap 9 are the same angle, so")
    print("the model cannot tell them apart. Multiple rates are hour, minute")
    print("and second hands: the second hand aliases every minute, but the")
    print("hour hand disambiguates. The slowest hand in a d=128 head takes")
    print("about 54,000 positions to lap.")


if __name__ == "__main__":
    demo_1_clock_hands()
    demo_2_relative_position()
    demo_3_decay_with_distance()
    demo_4_only_q_and_k()
    demo_5_base_scaling()
    demo_6_pairing_conventions()
    demo_7_one_frequency()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. split q (and k) into d/2 pairs of dims. Each pair is a clock hand.
  2. hand i ticks theta_i = 10000^(-2i/d) radians per position.
  3. at position m, turn hand i by m * theta_i. That is R(m) q.
  4. q_m . k_n = q^T R(n - m) k  -- only the gap survives. Zero parameters.
  5. apply it to q and k after their projections. Never to v.
  6. far-apart tokens score lower on average (long-term decay). A trend.
  7. raise the base and the slow hands slow down: that is context extension.

  Everything else in the paper is: a proof of step 4, a plug-in for linear
  attention, and experiments. Real, but not the idea.
""")
