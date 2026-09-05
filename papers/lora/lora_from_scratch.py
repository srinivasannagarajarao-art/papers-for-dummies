"""
LoRA: Low-Rank Adaptation of Large Language Models -- for programmers,
not researchers.

Run it:      python3 lora_from_scratch.py
Debug it:    set a breakpoint in lora_forward and step through.

No torch. No autograd. No training loop. The "fine-tune" is solved in closed
form (least squares) and the low-rank part is an SVD, so you can SEE the
weight diff and how little of it matters. Every function is <15 lines.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=80)


# ---------------------------------------------------------------------------
# STAGE 0 -- the frozen base. A pretrained linear layer is one matrix W and
# the forward pass is h = x @ W. A full fine-tune edits every number in W.
# LoRA never touches W. It stores a patch next to it.
# ---------------------------------------------------------------------------
def make_base(d_in, d_out, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_out))   # "pretrained"


# ---------------------------------------------------------------------------
# STAGE 1 -- how many numbers you train. Full fine-tune: every entry of W.
# LoRA: two thin matrices, A (d_in, r) and B (r, d_out). That is
# r * (d_in + d_out) numbers instead of d_in * d_out.
# ---------------------------------------------------------------------------
def param_count(d_in, d_out, r):
    full = d_in * d_out
    lora = r * (d_in + d_out)
    return full, lora


# ---------------------------------------------------------------------------
# STAGE 2 -- THE WHOLE PAPER. Equation 3, section 4.1:
#
#     h = W0 x + dW x = W0 x + B A x        dW = B A,  rank(dW) <= r
#
# I use row vectors (x @ W), so my patch reads A then B -- the order x meets
# them -- and my A @ B is the transpose of the paper's B A. Same two matrices.
#
# Init (section 4.1): A random Gaussian, B zero. So dW = 0 at step 0 and the
# adapted model IS the base model until training moves B.
# ---------------------------------------------------------------------------
def lora_init(d_in, d_out, r, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1 / np.sqrt(d_in), (d_in, r))   # random, like any layer
    B = np.zeros((r, d_out))                           # zero: the patch is empty
    return A, B


def lora_forward(x, W, A, B, alpha, r):
    # (1) BASE: the frozen pretrained layer, untouched
    base = x @ W

    # (2) PATCH: squeeze x down to r numbers, expand back out. Two thin matmuls.
    patch = (x @ A) @ B

    # (3) SCALE by alpha/r, so changing r doesn't change how big a step is.
    #     Section 4.1: "we simply set alpha to the first r we try".
    return base + (alpha / r) * patch


def merge(W, A, B, alpha, r):
    """Fold the patch into the weight. One matrix again: zero extra latency."""
    return W + (alpha / r) * (A @ B)


# ---------------------------------------------------------------------------
# STAGE 3 -- a "fine-tune" with no training loop. For one linear layer the
# weight diff a full fine-tune converges to has a closed form: least squares
# on the residual the base model gets wrong. Then an SVD tells you how many
# independent directions that diff actually has -- its rank.
# ---------------------------------------------------------------------------
def fit_delta(X, Y, W):
    """What full fine-tuning would learn: the dW that best maps X to Y."""
    residual = Y - X @ W                          # what the base gets wrong
    dW, *_ = np.linalg.lstsq(X, residual, rcond=None)
    return dW


def low_rank(dW, r):
    """Keep the top-r directions of dW. Returns A (d_in, r), B (r, d_out)."""
    U, s, Vt = np.linalg.svd(dW, full_matrices=False)
    A = U[:, :r] * s[:r]      # fold the singular values into A
    B = Vt[:r]
    return A, B, s            # A @ B is the best rank-r approximation of dW


# ---------------------------------------------------------------------------
# STAGE 4 -- one gradient, by hand, so you can see WHY B starts at zero.
# h = x@W + s*(x@A)@B.  With G = dL/dh (same shape as h), the chain rule:
#     dL/dB = s * (x@A).T @ G          dL/dA = s * x.T @ G @ B.T
# Look at dL/dA: it is multiplied by B. If B = 0, A gets no gradient at step 0
# but B does -- so training starts. If BOTH were zero, nothing would ever move.
# ---------------------------------------------------------------------------
def lora_grads(x, A, B, G, alpha, r):
    s = alpha / r
    grad_A = s * x.T @ G @ B.T
    grad_B = s * (x @ A).T @ G
    return grad_A, grad_B


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_1_param_count():
    line("DEMO 1: how many numbers are you training?")
    d = 4096
    full = d * d
    print(f"one square weight, d = {d}. Full fine-tune trains d*d = {full:,}\n")
    print(f"  {'r':>3}   {'LoRA params':>11}   {'full / LoRA':>11}")
    for r in (1, 4, 8, 64):
        full, lora = param_count(d, d, r)
        print(f"  {r:>3}   {lora:>11,}   {full / lora:>10,.0f}x")

    # The paper's GPT-3 numbers fall out of the same formula.
    L, dm = 96, 12288        # GPT-3 175B: 96 layers, d_model 12288
    print(f"\nGPT-3 175B shape: {L} layers, d_model {dm}. LoRA on Wq and Wv only:")
    for r in (1, 4, 8):
        _, lora = param_count(dm, dm, r)
        total = lora * 2 * L
        print(f"  r = {r}:  {total:>12,} trainable = {total / 1e6:5.1f}M"
              f"   (175,000M full, ratio {175e9 / total:,.0f}x)")

    # BREAK IT: patch every matrix instead of just Wq and Wv.
    r, vocab, dff = 8, 50257, 4 * dm
    wq_wv = param_count(dm, dm, r)[1] * 2 * L
    attn4 = param_count(dm, dm, r)[1] * 4 * L
    mlp2 = param_count(dm, dff, r)[1] * 2 * L
    emb2 = param_count(vocab, dm, r)[1] * 2          # embedding + LM head
    print(f"\nBREAK IT: LoRA on every matrix at r = {r}, same GPT-3 shape:")
    print(f"  Wq, Wv only (the paper) ........ {wq_wv / 1e6:7.1f}M")
    print(f"  + Wk, Wo ........................ {attn4 / 1e6:7.1f}M")
    print(f"  + both MLP matrices ............. {(attn4 + mlp2) / 1e6:7.1f}M")
    print(f"  + embedding and LM head ......... {(attn4 + mlp2 + emb2) / 1e6:7.1f}M"
          f"   ({(attn4 + mlp2 + emb2) / wq_wv:.1f}x the paper's count)")

    # What LoRA does NOT shrink: the frozen W is still loaded.
    full, lora = param_count(d, d, 8)
    print(f"\nWhat LoRA does NOT shrink (d = {d}, r = 8, fp16 weights, fp32 Adam):")
    print(f"  the frozen W itself ............ {2 * full / 1e6:6.1f} MB  either way")
    print(f"  full FT  grads + Adam m, v ..... {10 * full / 1e6:6.1f} MB")
    print(f"  LoRA     grads + Adam m, v ..... {10 * lora / 1e6:6.1f} MB")
    print("\nREAD THIS: LoRA cuts what you TRAIN (gradients, optimizer state,")
    print("checkpoints), not what you LOAD. The base model costs the same memory")
    print("at inference. Shrinking the base is quantisation's job, not LoRA's.")


def make_task(d, n, dW_true, seed):
    """Toy data: the base W is wrong, the task wants W + dW_true, plus noise."""
    rng = np.random.default_rng(seed)
    W = make_base(d, d)
    X = rng.normal(0, 1, (n, d))
    Y = X @ (W + dW_true) + rng.normal(0, 0.05, (n, d))
    return W, X, Y


def rank_table(W, X, Y, X_test, Y_test, ranks):
    dW = fit_delta(X, Y, W)                       # the "full fine-tune"
    _, s, _ = np.linalg.svd(dW, full_matrices=False)
    print("  singular values of the fitted dW (first 10):")
    print("  ", s[:10])
    base_rmse = np.sqrt(np.mean((X_test @ W - Y_test) ** 2))
    print(f"\n  {'rank r':>6}   {'dW error':>8}   {'test RMSE':>9}")
    print(f"  {'none':>6}   {'':>8}   {base_rmse:9.4f}   <- the base model")
    for r in ranks:
        A, B, _ = low_rank(dW, r)
        err = np.linalg.norm(dW - A @ B) / np.linalg.norm(dW)
        rmse = np.sqrt(np.mean((X_test @ merge(W, A, B, r, r) - Y_test) ** 2))
        tag = "  <- full fine-tune" if r == dW.shape[0] else ""
        print(f"  {r:>6}   {err:8.4f}   {rmse:9.4f}{tag}")


def demo_2_the_diff_is_low_rank():
    line("DEMO 2: the fine-tune diff is low-rank (and what that means)")
    d, n = 64, 512
    rng = np.random.default_rng(1)

    # The task genuinely needs a rank-4 change, plus a little full-rank noise.
    U, _ = np.linalg.qr(rng.normal(0, 1, (d, 4)))     # 4 orthonormal directions
    V, _ = np.linalg.qr(rng.normal(0, 1, (d, 4)))
    dW_true = U @ np.diag([3.0, 2.0, 1.0, 0.5]) @ V.T
    dW_true = dW_true + rng.normal(0, 0.01, (d, d))
    W, X, Y = make_task(d, n, dW_true, seed=10)
    _, X_test, Y_test = make_task(d, 256, dW_true, seed=11)
    print(f"d = {d}, {n} training rows. The task needs W + dW, dW rank 4 + noise.\n")
    rank_table(W, X, Y, X_test, Y_test, ranks=(1, 2, 4, 8, 64))

    print("\nREAD THIS: the singular values fall off a cliff after 4. The dW that a")
    print("full fine-tune finds has 4096 numbers in it but only 4 directions that")
    print("matter. Rank 4 gets you the whole fine-tune; rank 8 is already there.")

    # BREAK IT: a task whose diff is genuinely full-rank.
    dW_full = rng.normal(0, 1, (d, d))
    dW_full *= np.linalg.norm(dW_true) / np.linalg.norm(dW_full)  # same size
    W, X, Y = make_task(d, n, dW_full, seed=12)
    _, X_test, Y_test = make_task(d, 256, dW_full, seed=13)
    print(f"\nBREAK IT: same size of dW, but full-rank (random Gaussian):\n")
    rank_table(W, X, Y, X_test, Y_test, ranks=(1, 2, 4, 8, 64))
    print("\nREAD THIS: no cliff. Rank 8 still leaves most of the diff on the")
    print("floor and the test error barely moves. LoRA works because real")
    print("fine-tune diffs look like the first table, not this one (section 7).")


def demo_3_merge_costs_nothing():
    line("DEMO 3: merge the patch and pay nothing at inference")
    d, r, alpha = 64, 8, 8
    rng = np.random.default_rng(2)
    W = make_base(d, d)
    A, _ = lora_init(d, d, r)
    B = rng.normal(0, 0.1, (r, d))            # pretend training moved B
    x = rng.normal(0, 1, (512, d))            # 512 rows, so lstsq is a real fit

    unmerged = lora_forward(x, W, A, B, alpha, r)      # 3 matmuls
    W_merged = merge(W, A, B, alpha, r)
    merged = x @ W_merged                              # 1 matmul
    print(f"max |unmerged - merged| = {np.abs(unmerged - merged).max():.2e}")
    print(f"flops per token, base W alone ......... {2 * d * d:>6,}")
    patch_flops = 2 * d * d + 2 * (d * r + r * d)
    print(f"flops per token, base + unmerged patch  {patch_flops:>6,}")
    print(f"flops per token, merged ............... {2 * d * d:>6,}   <- same as base")

    # Switch tasks: subtract this patch, add the next one. Base is recovered.
    W_back = W_merged - (alpha / r) * (A @ B)
    back = np.abs(W_back - W).max()
    print(f"un-merge, max |W_back - W| = {back:.2e}   <- base recovered")

    # BREAK IT: an adapter (Houlsby 2019) is W then a small MLP with a
    # nonlinearity. Try to fold it into one matrix: you can't.
    D = rng.normal(0, 0.3, (d, r))
    Up = rng.normal(0, 0.3, (r, d))
    h = x @ W
    adapter_out = h + np.maximum(h @ D, 0) @ Up        # the relu is the problem
    W_fit, *_ = np.linalg.lstsq(x, adapter_out, rcond=None)
    resid = np.linalg.norm(x @ W_fit - adapter_out) / np.linalg.norm(adapter_out)
    W_fit2, *_ = np.linalg.lstsq(x, unmerged, rcond=None)
    resid2 = np.linalg.norm(x @ W_fit2 - unmerged) / np.linalg.norm(unmerged)
    print(f"\nbest single matrix that reproduces a LoRA layer:  residual {resid2:.2e}")
    print(f"best single matrix that reproduces an adapter:    residual {resid:.2e}")
    print("\nREAD THIS: LoRA's patch is linear, so it folds into W and the served")
    print("model is one matmul, same as before. An adapter has a relu between")
    print("two matmuls; no single matrix reproduces it, so it stays as extra")
    print("layers and extra latency. That is the paper's 'no inference latency'.")


def demo_4_zero_init():
    line("DEMO 4: B = 0 makes the adapted model the base model at step 0")
    d, n, r, alpha = 64, 16, 8, 8
    rng = np.random.default_rng(3)
    W = make_base(d, d)
    x = rng.normal(0, 1, (n, d))
    h_base = x @ W

    A, B = lora_init(d, d, r)
    h_lora = lora_forward(x, W, A, B, alpha, r)
    print(f"A random, B zero:    identical to base? {np.array_equal(h_base, h_lora)}"
          f"   max |diff| = {np.abs(h_base - h_lora).max()}")

    # BREAK IT: init B randomly too.
    B_bad = rng.normal(0, 1 / np.sqrt(d), (r, d))
    h_bad = lora_forward(x, W, A, B_bad, alpha, r)
    moved = np.linalg.norm(h_bad - h_base) / np.linalg.norm(h_base)
    print(f"A random, B random:  identical to base? {np.array_equal(h_base, h_bad)}"
          f"   outputs moved {moved:.1%}, untrained")

    # Which init can actually start training? One gradient at step 0.
    Y = h_base + rng.normal(0, 1, (n, d))         # some target the base misses
    G = 2 * (h_base - Y) / n                      # dL/dh for a mean squared error
    print(f"\n  {'init':<20} {'|dL/dA|':>8} {'|dL/dB|':>8}   what happens")
    inits = [("A random, B zero", A, B, "B moves first, then A. Trains."),
             ("A zero,   B zero", np.zeros_like(A), B, "both zero forever. Stuck."),
             ("A random, B random", A, B_bad, "trains, but step 0 is wrong.")]
    for name, A_, B_, verdict in inits:
        gA, gB = lora_grads(x, A_, B_, G, alpha, r)
        nA, nB = np.linalg.norm(gA), np.linalg.norm(gB)
        print(f"  {name:<20} {nA:8.3f} {nB:8.3f}   {verdict}")
    print("\nREAD THIS: zero B, not zero A. dL/dA is multiplied by B, so with B = 0")
    print("A sits still for one step while B gets a real gradient. Zero both and")
    print("every gradient is zero: the patch never leaves the origin.")


def demo_5_alpha_over_r():
    line("DEMO 5: what alpha / r is for")
    d, n, alpha, lr = 64, 256, 4, 1e-3
    rng = np.random.default_rng(4)
    W = make_base(d, d)
    x = rng.normal(0, 1, (n, d))
    Y = x @ (W + rng.normal(0, 0.1, (d, d)))      # a task the base gets wrong
    h0 = x @ W
    G = 2 * (h0 - Y) / n
    loss0 = np.mean((h0 - Y) ** 2)

    print(f"one Adam step (lr = {lr}) on B, from B = 0. Adam's first step is")
    print(f"exactly lr * sign(grad). alpha = {alpha}.\n")
    print(f"  {'r':>3}   {'no scale: |dh|':>14} {'loss drop':>10}"
          f"   {'alpha/r: |dh|':>13} {'loss drop':>10}")
    drops = {}
    for r in (1, 4, 16, 64):
        A, B = lora_init(d, d, r, seed=r)
        row = f"  {r:>3}  "
        for a in (r, alpha):                       # alpha = r means scale 1
            _, gB = lora_grads(x, A, B, G, a, r)
            B1 = -lr * np.sign(gB)                 # Adam, step 1
            dh = lora_forward(x, W, A, B1, a, r) - h0
            drop = loss0 - np.mean((h0 + dh - Y) ** 2)
            drops[(r, a == alpha)] = drop
            row += f"   {np.linalg.norm(dh):14.4f} {drop:10.5f}"
        print(row)
    raw = drops[(64, False)] / drops[(1, False)]
    scaled = drops[(64, True)] / drops[(1, True)]
    print(f"\nloss drop at r = 64 vs r = 1:  no scale {raw:.0f}x   alpha/r {scaled:.1f}x")
    print("\nREAD THIS: without the scale, the same optimizer step does far more to")
    print("the loss at r = 64 than at r = 1: change r and you have silently changed")
    print("your learning rate. With alpha/r the loss drop stays put, so you can")
    print("try a new r without retuning. (The r = 4 row matches: alpha/r = 1.)")


if __name__ == "__main__":
    demo_1_param_count()
    demo_2_the_diff_is_low_rank()
    demo_3_merge_costs_nothing()
    demo_4_zero_init()
    demo_5_alpha_over_r()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. a fine-tune  = W0 + dW. The diff is the whole story.
  2. the claim    = dW is low-rank: a few directions carry the change.
  3. LoRA         = store dW as B @ A, two thin matrices, rank r. Freeze W0.
  4. init         = A random, B zero, so step 0 is exactly the base model.
  5. alpha / r    = keeps the update size fixed when you change r.
  6. inference    = merge: W0 + B @ A is one matrix. No extra latency.
  7. where        = Wq and Wv in attention, r = 1 to 8, in the GPT-3 runs.

  Everything not listed above is evaluation: GLUE, E2E, WikiSQL, SAMSum.
  Real, but not the idea.
""")
