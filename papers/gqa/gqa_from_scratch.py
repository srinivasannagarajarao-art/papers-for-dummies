"""
Grouped-Query Attention -- for programmers, not researchers.

Run it:      python3 gqa_from_scratch.py
Debug it:    breakpoint in repeat_kv_heads and look at which query head got
             which key/value head.

No torch. No training. Just the forward pass, plus the paper's actual
contribution: converting a multi-head checkpoint into a grouped-query one by
mean-pooling. Every core function is short.

The cache arithmetic is NOT re-derived here -- ../kv-cache/ already does it.
This script is about the forward pass and what sharing costs.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- softmax. The `- max` is overflow safety, not maths.
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# STAGE 1 -- THE WHOLE IMPLEMENTATION. One repeat, in the right place.
#
# kv is (n_kv_heads, n, head_dim). We need (n_heads, n, head_dim) so every
# query head has a key head to talk to. group = n_heads // n_kv_heads.
#
#   "block"      np.repeat  -> [kv0 kv0 kv0 kv0 kv1 kv1 kv1 kv1]
#                              query head i uses kv head i // group   <-- right
#   "interleave" np.tile    -> [kv0 kv1 kv0 kv1 kv0 kv1 kv0 kv1]
#                              query head i uses kv head i %  n_kv    <-- wrong
#
# Both produce the correct SHAPE. Only one matches the convention every
# checkpoint on disk was saved with. Getting it wrong raises nothing.
# ---------------------------------------------------------------------------
def repeat_kv_heads(kv, n_heads, mode="block"):
    n_kv_heads = kv.shape[0]
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}: "
            "groups must be equal-sized"
        )
    group = n_heads // n_kv_heads
    if mode == "block":
        return np.repeat(kv, group, axis=0)      # head i -> kv head i // group
    if mode == "interleave":
        return np.tile(kv, (group, 1, 1))        # head i -> kv head i % n_kv
    raise ValueError(f"unknown repeat mode {mode!r}")


# ---------------------------------------------------------------------------
# STAGE 2 -- attention over head-split tensors. This ONE function is
# multi-head, multi-query and grouped-query. The only difference is how many
# rows K and V arrive with; the repeat above makes them meet the queries.
#
#   Q  (n_heads,    n, head_dim)
#   K  (n_kv_heads, n, head_dim)     n_kv_heads == n_heads  -> multi-head
#   V  (n_kv_heads, n, head_dim)     n_kv_heads == 1        -> multi-query
#                                    anything between       -> grouped-query
# ---------------------------------------------------------------------------
def attention(Q, K, V, mode="block", causal=False, verbose=False):
    n_heads, n, head_dim = Q.shape
    K = repeat_kv_heads(K, n_heads, mode)
    V = repeat_kv_heads(V, n_heads, mode)

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(head_dim)   # (H, n, n)
    if causal:
        mask = np.tril(np.ones((n, n), dtype=bool))
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    out = weights @ V                                        # (H, n, head_dim)

    if verbose:
        print(f"    Q {Q.shape}  K after repeat {K.shape}  V after repeat "
              f"{V.shape}")
        print(f"    scores {scores.shape}  weights {weights.shape}  "
              f"out {out.shape}")
    return out, weights


# ---------------------------------------------------------------------------
# STAGE 3 -- the projections. Q always has n_heads heads. K and V have
# n_kv_heads heads, and that is the entire architectural change: W_k and W_v
# get narrower. W_q and W_o do not move.
# ---------------------------------------------------------------------------
def make_weights(d_model, n_heads, n_kv_heads, head_dim, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.5 / np.sqrt(d_model)
    return {
        "W_q": rng.normal(0, s, (d_model, n_heads * head_dim)),
        "W_k": rng.normal(0, s, (d_model, n_kv_heads * head_dim)),
        "W_v": rng.normal(0, s, (d_model, n_kv_heads * head_dim)),
        "W_o": rng.normal(0, s, (n_heads * head_dim, d_model)),
    }


def split_heads(x, n_h, head_dim):
    # (n, n_h*head_dim) -> (n_h, n, head_dim). Heads are contiguous blocks.
    n = x.shape[0]
    return x.reshape(n, n_h, head_dim).transpose(1, 0, 2)


def forward(X, W, n_heads, n_kv_heads, head_dim, mode="block", verbose=False):
    """One attention layer. n_kv_heads picks the variant."""
    Q = split_heads(X @ W["W_q"], n_heads, head_dim)
    K = split_heads(X @ W["W_k"], n_kv_heads, head_dim)
    V = split_heads(X @ W["W_v"], n_kv_heads, head_dim)
    if verbose:
        print(f"    X {X.shape} -> Q {Q.shape}  K {K.shape}  V {V.shape}")
    out, _ = attention(Q, K, V, mode=mode, verbose=verbose)
    merged = out.transpose(1, 0, 2).reshape(X.shape[0], n_heads * head_dim)
    return merged @ W["W_o"]


# ---------------------------------------------------------------------------
# STAGE 4 -- UPTRAINING, the paper's actual contribution. You do not train a
# GQA model from scratch. You take an existing multi-head checkpoint and
# MEAN-POOL the key and value heads inside each group, then train briefly.
#
# W_k is (d_model, n_heads*head_dim). View it as (d_model, n_heads, head_dim),
# average over the heads in each group, and you have (d_model, n_kv, head_dim).
# The mean is the projection that minimises squared error to all group members;
# picking one head throws the other g-1 away.
# ---------------------------------------------------------------------------
def convert_checkpoint(W_mha, n_heads, n_kv_heads, head_dim, how="mean"):
    group = n_heads // n_kv_heads
    out = {"W_q": W_mha["W_q"].copy(), "W_o": W_mha["W_o"].copy()}
    for name in ("W_k", "W_v"):
        w = W_mha[name].reshape(-1, n_kv_heads, group, head_dim)
        if how == "mean":
            w = w.mean(axis=2)               # mean-pool inside each group
        elif how == "first":
            w = w[:, :, 0, :]                # keep head 0, bin the rest
        else:
            raise ValueError(how)
        out[name] = w.reshape(-1, n_kv_heads * head_dim)
    return out


# ===========================================================================
# DEMOS
# ===========================================================================

D_MODEL, N_HEADS, HEAD_DIM, N_TOK = 64, 8, 8, 6


def demo_1_one_function_three_variants():
    print("=" * 74)
    print("DEMO 1 -- one function, three variants. Watch the K/V shapes only.")
    print("=" * 74)
    X = np.random.default_rng(1).normal(0, 1, (N_TOK, D_MODEL))
    for n_kv, label in [(8, "multi-head  (MHA, 2017)"),
                        (2, "grouped-query (GQA, 2023)"),
                        (1, "multi-query (MQA, 2019)")]:
        print(f"\n  n_kv_heads={n_kv}  {label}")
        W = make_weights(D_MODEL, N_HEADS, n_kv, HEAD_DIM, seed=7)
        y = forward(X, W, N_HEADS, n_kv, HEAD_DIM, verbose=True)
        print(f"    layer output {y.shape}   <- identical for all three")


def demo_2_the_repeat_is_the_implementation():
    print()
    print("=" * 74)
    print("DEMO 2 -- the repeat IS the implementation. Block vs interleave.")
    print("=" * 74)
    n_kv, group = 2, N_HEADS // 2
    tag = np.arange(n_kv).reshape(n_kv, 1, 1) * np.ones((n_kv, 1, 1))
    blk = repeat_kv_heads(tag, N_HEADS, "block").ravel().astype(int)
    itl = repeat_kv_heads(tag, N_HEADS, "interleave").ravel().astype(int)
    print(f"\n  n_heads={N_HEADS}, n_kv_heads={n_kv}, group={group}")
    print("  query head       :", list(range(N_HEADS)))
    print("  block  -> kv head:", list(blk), " matches i // group")
    print("  interl -> kv head:", list(itl), " matches i %  n_kv")
    print("  convention (i // group):", [i // group for i in range(N_HEADS)])

    X = np.random.default_rng(2).normal(0, 1, (N_TOK, D_MODEL))
    W = make_weights(D_MODEL, N_HEADS, n_kv, HEAD_DIM, seed=3)
    y_ok = forward(X, W, N_HEADS, n_kv, HEAD_DIM, mode="block")
    y_bad = forward(X, W, N_HEADS, n_kv, HEAD_DIM, mode="interleave")
    print(f"\n  output shapes equal?            {y_ok.shape == y_bad.shape}")
    print("  no exception raised?            True")
    print(f"  max |block - interleave|        {np.abs(y_ok - y_bad).max():.4f}")
    print(f"  mean |block|                    {np.abs(y_ok).mean():.4f}")
    print("  -> same shape, no error, different answer. That is the bug.")


def demo_3_equivalence_with_multi_head():
    print()
    print("=" * 74)
    print("DEMO 3 -- with n_kv_heads == n_heads, this IS multi-head attention.")
    print("=" * 74)
    X = np.random.default_rng(4).normal(0, 1, (N_TOK, D_MODEL))
    W = make_weights(D_MODEL, N_HEADS, N_HEADS, HEAD_DIM, seed=5)

    # plain multi-head: no repeat at all, head i uses key head i
    Q = split_heads(X @ W["W_q"], N_HEADS, HEAD_DIM)
    K = split_heads(X @ W["W_k"], N_HEADS, HEAD_DIM)
    V = split_heads(X @ W["W_v"], N_HEADS, HEAD_DIM)
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(HEAD_DIM)
    ref = (softmax(s) @ V).transpose(1, 0, 2).reshape(N_TOK, -1) @ W["W_o"]

    got = forward(X, W, N_HEADS, N_HEADS, HEAD_DIM)
    print(f"\n  max |grouped(n_kv=8) - plain multi-head| = "
          f"{np.abs(ref - got).max():.3e}")
    print(f"  bit-identical?                            {np.array_equal(ref, got)}")
    print("  group size is 1, so np.repeat is a no-op. Same code path.")


def demo_4_what_sharing_costs():
    print()
    print("=" * 74)
    print("DEMO 4 -- what sharing costs, when heads want different keys.")
    print("=" * 74)
    rng = np.random.default_rng(11)
    n = 8
    # Each head's query points at a DIFFERENT token: head h wants token h.
    Q = np.zeros((N_HEADS, 1, HEAD_DIM))
    K_full = np.zeros((N_HEADS, n, HEAD_DIM))
    V = rng.normal(0, 1, (1, n, HEAD_DIM)) * np.ones((N_HEADS, 1, 1))
    dirs = np.eye(HEAD_DIM)
    for h in range(N_HEADS):
        Q[h, 0] = dirs[h % HEAD_DIM] * 6.0
        K_full[h] = rng.normal(0, 0.1, (n, HEAD_DIM))
        K_full[h, h % n] += dirs[h % HEAD_DIM] * 6.0     # head h's ideal key

    ideal, _ = attention(Q, K_full, V)
    print("\n  ideal: every head has its own key head (n_kv=8).")
    print("  now mean-pool the key heads into groups and re-run:\n")
    print("    n_kv | group |  max |out - ideal| | mean |out - ideal|")
    for n_kv in (8, 4, 2, 1):
        g = N_HEADS // n_kv
        K_sh = K_full.reshape(n_kv, g, n, HEAD_DIM).mean(axis=1)
        V_sh = V.reshape(n_kv, g, n, HEAD_DIM).mean(axis=1)
        out, _ = attention(Q, K_sh, V_sh)
        d = np.abs(out - ideal)
        print(f"    {n_kv:4d} | {g:5d} |         {d.max():8.4f} |"
              f"         {d.mean():8.4f}")

    print("\n  per-head error at n_kv=2 (heads 0-3 share, heads 4-7 share):")
    K_sh = K_full.reshape(2, 4, n, HEAD_DIM).mean(axis=1)
    V_sh = V.reshape(2, 4, n, HEAD_DIM).mean(axis=1)
    out, w = attention(Q, K_sh, V_sh)
    for h in range(N_HEADS):
        print(f"    head {h}  group {h // 4}  error {np.abs(out[h] - ideal[h]).max():.4f}"
              f"   argmax attends to token {int(w[h, 0].argmax())}"
              f" (wanted {h % n})")
    print("\n  Heads in the same group are pushed toward the same answer.")
    print("  This is a mechanical illustration on constructed vectors, NOT a")
    print("  language-modelling quality measurement. The paper's quality")
    print("  numbers are the paper's; see ../kv-cache/ for the memory side.")


def demo_5_parameters_and_compute():
    print()
    print("=" * 74)
    print("DEMO 5 -- parameters shrink a little; attention FLOPs do not move.")
    print("=" * 74)
    d, h, dh, n = 4096, 32, 128, 2048
    print(f"\n  Llama-2-7B-shaped layer: d_model={d}, n_heads={h}, "
          f"head_dim={dh}, seq={n}")
    print("\n    variant     | n_kv | W_k+W_v params | vs MHA | attn MACs/token")
    for n_kv, label in [(32, "multi-head "), (8, "GQA-8      "),
                        (4, "GQA-4      "), (1, "multi-query")]:
        kv_params = 2 * d * n_kv * dh
        # attention itself: QK^T and weights@V, over all n_heads query heads
        attn_macs = 2 * h * dh * n
        print(f"    {label} | {n_kv:4d} | {kv_params:14,d} | "
              f"{2 * d * 32 * dh / kv_params:5.1f}x | {attn_macs:,d}")
    print(f"\n  W_q and W_o are untouched: {d * h * dh:,d} params each, "
          "in every variant.")
    print("  The attention arithmetic is identical -- the keys are expanded")
    print("  back to 32 heads before the matmul. Only the CACHE shrinks,")
    print("  by exactly n_heads / n_kv_heads. That table lives in ../kv-cache/.")


def demo_6_uptraining_by_mean_pooling():
    print()
    print("=" * 74)
    print("DEMO 6 -- uptraining: convert an MHA checkpoint, don't retrain it.")
    print("=" * 74)
    X = np.random.default_rng(21).normal(0, 1, (N_TOK, D_MODEL))
    W_mha = make_weights(D_MODEL, N_HEADS, N_HEADS, HEAD_DIM, seed=9)
    ref = forward(X, W_mha, N_HEADS, N_HEADS, HEAD_DIM)
    scale = np.abs(ref).mean()
    print(f"\n  original multi-head output, mean |y| = {scale:.4f}")
    print("\n    n_kv | mean-pooled err | first-head err | first/mean")
    for n_kv in (4, 2, 1):
        y_mean = forward(X, convert_checkpoint(W_mha, N_HEADS, n_kv, HEAD_DIM,
                                               "mean"),
                         N_HEADS, n_kv, HEAD_DIM)
        y_first = forward(X, convert_checkpoint(W_mha, N_HEADS, n_kv, HEAD_DIM,
                                                "first"),
                          N_HEADS, n_kv, HEAD_DIM)
        em = np.abs(y_mean - ref).mean()
        ef = np.abs(y_first - ref).mean()
        print(f"    {n_kv:4d} |        {em:8.4f} |       {ef:8.4f} |"
              f"      {ef / em:5.2f}x")
    print("\n  Mean-pooling starts closer, every time. That is the whole")
    print("  argument for it: a shorter distance for the brief uptraining")
    print("  run to cover. The GQA paper reports doing this with a small")
    print("  fraction of the original pre-training compute.")


def demo_7_break_it():
    print()
    print("=" * 74)
    print("DEMO 7 -- breaking it on purpose.")
    print("=" * 74)
    print("\n  (a) n_kv_heads that does not divide n_heads:")
    try:
        kv = np.zeros((3, N_TOK, HEAD_DIM))
        repeat_kv_heads(kv, N_HEADS, "block")
    except ValueError as e:
        print(f"      ValueError: {e}")
    print("      Groups must be equal-sized, or head->group is not a function.")

    print("\n  (b) share keys but not values (half the saving):")
    d, h, dh = 4096, 32, 128
    both = 2 * 8 * dh
    keys_only = (8 + 32) * dh
    print(f"      cache floats/token/layer, GQA-8 both K and V : {both:,d}")
    print(f"      cache floats/token/layer, K shared, V not    : {keys_only:,d}")
    print(f"      full multi-head                              : "
          f"{2 * 32 * dh:,d}")
    print(f"      reduction: 4.0x -> {2 * 32 * dh / keys_only:.1f}x")

    print("\n  (c) expand K and V to n_heads BEFORE caching them:")
    print(f"      cached floats/token/layer, expanded : {2 * 32 * dh:,d}")
    print(f"      cached floats/token/layer, correct  : {both:,d}")
    print("      Identical maths, zero saving. See ../kv-cache/.")


if __name__ == "__main__":
    demo_1_one_function_three_variants()
    demo_2_the_repeat_is_the_implementation()
    demo_3_equivalence_with_multi_head()
    demo_4_what_sharing_costs()
    demo_5_parameters_and_compute()
    demo_6_uptraining_by_mean_pooling()
    demo_7_break_it()
