"""
LLaMA: Open and Efficient Foundation Language Models -- for programmers.

Run it:      python3 llama_from_scratch.py
Debug it:    set a breakpoint in pre_norm_block and watch 20 layers go by.

No torch. No training. LLaMA is not a new mechanism, it is a recipe: the 2017
transformer with four config changes, trained on far more data than the
scaling laws said to. So every demo here is either a forward pass you can
watch, or arithmetic you can check on a napkin. Every function is short.
"""

import inspect

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# DELTA 1a -- RMSNorm (Zhang & Sennrich 2019) instead of LayerNorm.
#
# Both put every token's vector on the same scale before it hits a matmul.
# LayerNorm: subtract the mean, divide by the standard deviation, then a
# learned scale (gamma) and shift (beta). RMSNorm: skip the mean, divide by
# the root-mean-square, learned scale only, no beta. Fewer passes over the
# vector, and nobody could measure a quality loss. gamma is 1 and beta is 0
# at init, so they are left out here.
# ---------------------------------------------------------------------------
def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def rms_norm(x, eps=1e-5):
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / rms


# Elementwise passes over a length-d vector, counted off the code above.
#   layer_norm: mean, subtract, square, mean, divide, *gamma, +beta  = 7
#   rms_norm:   square, mean, divide, *gamma                          = 4
PASSES_LAYERNORM, PASSES_RMSNORM = 7, 4


# ---------------------------------------------------------------------------
# DELTA 2 -- SwiGLU (Shazeer 2020) instead of ReLU in the feed-forward.
#
#   vanilla:  relu(x @ W1) @ W2                    2 matrices, hidden = 4d
#   SwiGLU:   (silu(x @ W_gate) * (x @ W_up)) @ W_down
#                                                  3 matrices, hidden = 2/3*4d
#
# A gate: one branch decides HOW MUCH gets through, the other decides WHAT,
# multiplied elementwise. Three matrices instead of two, so the hidden width
# is cut to 2/3 to keep the parameter count level, then rounded up to a
# multiple of 256 for the GPU. This is the arithmetic in the released model.py.
# ---------------------------------------------------------------------------
def silu(x):
    return x / (1.0 + np.exp(-x))                    # x * sigmoid(x)


def llama_hidden(dim, multiple_of=256):
    hidden = int(2 * (4 * dim) / 3)                  # 2/3 of the vanilla 4d
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


def relu_ffn(x, W1, W2):
    return np.maximum(x @ W1, 0.0) @ W2              # (n,d)->(n,4d)->(n,d)


def swiglu_ffn(x, W_gate, W_up, W_down):
    return (silu(x @ W_gate) * (x @ W_up)) @ W_down  # (n,d)->(n,h)->(n,d)


def ffn_params(dim, hidden, gated):
    return (3 if gated else 2) * dim * hidden        # no bias terms anywhere


# ---------------------------------------------------------------------------
# DELTA 3 -- rotary position embeddings (Su et al. 2021) instead of adding a
# position vector to the embeddings once at the bottom.
#
# Pair up dims (0,1), (2,3), ... and rotate each pair by position * theta_i,
# with theta_i shrinking geometrically across pairs. Applied to Q and K only,
# inside EVERY attention layer. Never to V. A dot product of two rotated
# vectors depends only on the DIFFERENCE of their positions, so attention
# scores get relative position for free. The full story lives at ../rope/.
# ---------------------------------------------------------------------------
def rope(x, positions, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)        # (d/2,) one per pair
    angle = positions[:, None] * theta[None, :]      # (n, d/2)
    cos, sin = np.cos(angle), np.sin(angle)
    x1, x2 = x[..., 0::2], x[..., 1::2]              # even dims, odd dims
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin             # a 2-D rotation per pair
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# Single-head causal attention, lifted from ../attention/. Unchanged by LLaMA.
def attention(Q, K, V):
    n, d_k = Q.shape
    scores = Q @ K.T / np.sqrt(d_k)
    scores = np.where(np.tril(np.ones((n, n), bool)), scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ V


# ---------------------------------------------------------------------------
# DELTA 4 -- no bias terms. Every matrix below is a plain matrix. The paper
# never argues for this; it is just how model.py is written (bias=False on
# every Linear). Demo 5 shows what it saves, which is nothing worth having.
# ---------------------------------------------------------------------------
def init_block(d, hidden, rng):
    """One block's weights, variance-preserving: std = 1/sqrt(fan_in)."""
    w = lambda i, o: rng.normal(0.0, 1.0 / np.sqrt(i), (i, o))
    return dict(Wq=w(d, d), Wk=w(d, d), Wv=w(d, d), Wo=w(d, d),
                Wg=w(d, hidden), Wu=w(d, hidden), Wd=w(hidden, d))


def attn_sublayer(x, p):
    pos = np.arange(len(x), dtype=float)
    Q, K, V = x @ p["Wq"], x @ p["Wk"], x @ p["Wv"]
    return attention(rope(Q, pos), rope(K, pos), V) @ p["Wo"]  # rope: Q, K. Not V


def ffn_sublayer(x, p):
    return swiglu_ffn(x, p["Wg"], p["Wu"], p["Wd"])


# ---------------------------------------------------------------------------
# DELTA 1b -- pre-normalisation: WHERE the norm sits relative to the skip link.
#
#   2017 (post-norm):  x = norm(x + f(x))      norm on the trunk
#   LLaMA (pre-norm):  x = x + f(norm(x))      norm on the branch
#
# Pre-norm leaves the trunk as a straight wire from the embedding to the
# output: every block ADDS to it and nothing rescales it. Post-norm rescales
# the trunk twice per block, so whatever each block does to a signal gets
# MULTIPLIED down the stack. Demo 3 measures both.
# ---------------------------------------------------------------------------
def pre_norm_block(x, p):                            # LLaMA
    x = x + attn_sublayer(rms_norm(x), p)
    x = x + ffn_sublayer(rms_norm(x), p)
    return x


def post_norm_block(x, p):                           # 2017
    x = rms_norm(x + attn_sublayer(x, p))
    x = rms_norm(x + ffn_sublayer(x, p))
    return x


def no_norm_block(x, p):                             # neither. For the break
    x = x + attn_sublayer(x, p)
    x = x + ffn_sublayer(x, p)
    return x


def run_stack(x, blocks, block_fn):
    """Push x through the blocks. Returns the output and, per layer, the
    residual-stream norm."""
    norms = []
    for p in blocks:
        x = block_fn(x, p)
        norms.append(np.linalg.norm(x, axis=-1).mean())   # per token, averaged
    return rms_norm(x), norms                             # final norm, as LLaMA


def sensitivity(x, blocks, block_fn, delta):
    """||change in output|| / ||change in input||. A finite-difference gradient:
    the same trick you would use to unit-test a hand-written backward pass."""
    base, _ = run_stack(x, blocks, block_fn)
    bumped, _ = run_stack(x + delta, blocks, block_fn)
    return np.linalg.norm(bumped - base) / np.linalg.norm(delta)


# ---------------------------------------------------------------------------
# THE ARITHMETIC. Every LLaMA-1 size is this one formula with the config
# numbers from Table 2 of the paper plugged in.
# ---------------------------------------------------------------------------
VOCAB = 32000
LLAMA = {  # dim, layers, heads, and the size the paper reports
    "7B":  dict(dim=4096, n_layers=32, n_heads=32, paper="6.7B",  tokens=1.0e12),
    "13B": dict(dim=5120, n_layers=40, n_heads=40, paper="13.0B", tokens=1.0e12),
    "33B": dict(dim=6656, n_layers=60, n_heads=52, paper="32.5B", tokens=1.4e12),
    "65B": dict(dim=8192, n_layers=80, n_heads=64, paper="65.2B", tokens=1.4e12),
}


def count_params(dim, n_layers, hidden, vocab=VOCAB, bias=False):
    attn = 4 * dim * dim                             # Wq, Wk, Wv, Wo
    ffn = 3 * dim * hidden                           # gate, up, down
    norms = 2 * dim                                  # two RMSNorm gammas
    if bias:                                          # the 2017 model had these
        attn += 4 * dim                              # one bias per projection
        ffn += 2 * hidden + dim                      # gate, up, down
        norms += 2 * dim                             # LayerNorm's beta
    per_layer = attn + ffn + norms
    embed = vocab * dim                              # input embedding table
    head = vocab * dim                               # output projection, untied
    return embed + n_layers * per_layer + dim + head  # + dim: the final norm


def chinchilla_tokens(n_params, tokens_per_param=20):
    """Hoffmann et al. 2022: compute-optimal is about 20 tokens per parameter."""
    return tokens_per_param * n_params


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_1_rmsnorm_vs_layernorm():
    line("DEMO 1: RMSNorm is LayerNorm minus the mean subtraction")

    x = np.array([1.0, 2.0, 3.0, 4.0])
    ln, rms = layer_norm(x), rms_norm(x)
    print(f"x           = {x}   mean {x.mean():.4f}")
    print(f"layer_norm  = {ln}   mean {ln.mean():.4f}")
    print(f"rms_norm    = {rms}   mean {rms.mean():.4f}")
    print("same numbers?", np.allclose(ln, rms))

    xc = x - x.mean()
    print(f"\nx centred   = {xc}   mean {xc.mean():.4f}")
    print(f"layer_norm  = {layer_norm(xc)}")
    print(f"rms_norm    = {rms_norm(xc)}")
    print("same numbers?", np.allclose(layer_norm(xc), rms_norm(xc)))

    d, layers = 4096, 32
    ln_ops, rms_ops = PASSES_LAYERNORM * d, PASSES_RMSNORM * d
    print(f"\nelementwise ops per token per norm at d={d}:")
    saved = ln_ops - rms_ops
    print(f"  layer_norm {ln_ops:,}   rms_norm {rms_ops:,}   saved {saved:,}")
    print(f"  x 2 norms per layer x {layers} layers = "
          f"{2 * layers * saved:,} ops saved per token")

    print("\nBREAK: eps=0 on a vector with nothing in it")
    with np.errstate(invalid="ignore", divide="ignore"):
        print("  rms_norm(zeros, eps=0)    =", rms_norm(np.zeros(4), eps=0.0))
        print("  rms_norm(zeros, eps=1e-5) =", rms_norm(np.zeros(4)))
        print("  layer_norm([3,3,3,3], eps=0) =",
              layer_norm(np.array([3.0, 3.0, 3.0, 3.0]), eps=0.0))

    print("\nREAD THIS: on a centred vector the two are the same function.")
    print("The only thing RMSNorm drops is the mean subtraction, and the")
    print("outputs differ exactly by that: rms_norm's output keeps x's mean.")
    print("The eps is not decoration. 0/0 is NaN, and one NaN in the residual")
    print("stream poisons every layer above it.")


def demo_2_swiglu_vs_relu():
    line("DEMO 2: SwiGLU costs three matrices, so the hidden layer shrinks")

    d = 4096
    h_relu, h_swiglu = 4 * d, llama_hidden(d)
    p_relu = ffn_params(d, h_relu, gated=False)
    p_swiglu = ffn_params(d, h_swiglu, gated=True)
    print(f"d = {d}")
    print(f"  vanilla ReLU FFN:  hidden = 4d   = {h_relu:5d}"
          f"   params = 2*d*h = {p_relu:,}")
    print(f"  LLaMA SwiGLU FFN:  hidden = 2/3*4d = {h_swiglu:5d}"
          f"   params = 3*d*h = {p_swiglu:,}")
    print(f"  ratio SwiGLU / ReLU = {p_swiglu / p_relu:.3f}")

    print("\nllama_hidden(dim) for every size in Table 2:")
    for name, c in LLAMA.items():
        print(f"  {name:>3s}: dim {c['dim']:5d} -> ffn {llama_hidden(c['dim'])}")

    p_fat = ffn_params(d, 4 * d, gated=True)
    print(f"\nBREAK: SwiGLU with hidden = 4d = {4 * d}:"
          f"   params = {p_fat:,}   ({100 * (p_fat / p_relu - 1):+.1f}%)")

    # A forward pass on toy sizes, to see what a gate does that a switch can't.
    rng = np.random.default_rng(2)
    n, d, h = 6, 8, llama_hidden(8, multiple_of=8)
    x = rng.normal(size=(n, d))
    W1, W2 = rng.normal(size=(d, 4 * d)), rng.normal(size=(4 * d, d))
    Wg, Wu, Wd = (rng.normal(size=(d, h)), rng.normal(size=(d, h)),
                  rng.normal(size=(h, d)))
    hid_relu = np.maximum(x @ W1, 0.0)
    hid_swiglu = silu(x @ Wg) * (x @ Wu)
    print(f"\ntoy forward pass, n={n} tokens, d={d}:")
    print(f"  relu_ffn   out shape {relu_ffn(x, W1, W2).shape}"
          f"   hidden units exactly 0: {100 * np.mean(hid_relu == 0):.1f}%")
    print(f"  swiglu_ffn out shape {swiglu_ffn(x, Wg, Wu, Wd).shape}"
          f"   hidden units exactly 0: {100 * np.mean(hid_swiglu == 0):.1f}%")

    print("\nREAD THIS: the paper's point is parameter-matched. 3 x 11008 is")
    print("within 1% of 2 x 16384, so any benchmark gain is the gate, not extra")
    print("capacity. ReLU is a switch: half the hidden units are exactly off.")
    print("SwiGLU is a dimmer: every unit passes something, scaled by a second")
    print("branch that saw the same input.")


def demo_3_pre_vs_post_norm():
    line("DEMO 3: pre-norm vs post-norm, random weights, 20 blocks deep")

    d, n_tok, depth = 64, 8, 32
    rng = np.random.default_rng(3)
    blocks = [init_block(d, llama_hidden(d, 8), rng) for _ in range(depth)]
    x0 = rms_norm(rng.normal(size=(n_tok, d)))       # a normalised embedding
    delta = 1e-4 * rng.normal(size=x0.shape)         # a tiny bump at layer 0
    orderings = (("pre-norm  (LLaMA)", pre_norm_block),
                 ("post-norm (2017) ", post_norm_block),
                 ("no norm at all   ", no_norm_block))

    print(f"d={d}, sqrt(d)={np.sqrt(d):.1f}. residual-stream norm per token:")
    print(f"{'':20s} layer 1   layer 5   layer 10  layer 20   bump at 0 -> out")
    with np.errstate(all="ignore"):
        for name, fn in orderings:
            _, norms = run_stack(x0, blocks[:20], fn)
            s = sensitivity(x0, blocks[:20], fn, delta)
            print(f"  {name}  {norms[0]:8.3g}  {norms[4]:8.3g}  {norms[9]:8.3g}"
                  f"  {norms[19]:8.3g}   x {s:.3g}")

    print("\nBREAK: same bump at layer 0, deeper and deeper stacks:")
    print(f"{'':20s}  5 deep   10 deep   20 deep   32 deep")
    for name, fn in orderings[:2]:
        s = [sensitivity(x0, blocks[:L], fn, delta) for L in (5, 10, 20, depth)]
        print(f"  {name}  " + "  ".join(f"x {v:6.3g}" for v in s))

    print("\nREAD THIS: post-norm pins the trunk at sqrt(d) -- that is what a")
    print("norm on the trunk does. Pre-norm lets the trunk grow and nobody")
    print("cares, because every sublayer reads rms_norm(x), never x. The")
    print("column that matters is the last: how much a change at the bottom")
    print("is amplified by the top. Pre-norm: about the same at any depth.")
    print("Post-norm: multiplies with depth. That ratio IS the gradient scale")
    print("a training step sees, and a ratio that grows with depth is a")
    print("training run that needs warmup, a small learning rate, and luck.")
    print("With no norm at all the trunk overflows before layer 10.")


def demo_4_rope_on_q_and_k_only():
    line("DEMO 4: RoPE rotates Q and K. Not V.")

    rng = np.random.default_rng(4)
    q, k = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))

    def score(pos_q, pos_k):
        rq = rope(q, np.array([pos_q], dtype=float))
        rk = rope(k, np.array([pos_k], dtype=float))
        return (rq @ rk.T).item()

    print(f"q at 3 . k at 1  (offset 2) = {score(3, 1):.4f}")
    print(f"q at 7 . k at 5  (offset 2) = {score(7, 5):.4f}"
          "   <- same offset, same score")
    print(f"q at 3 . k at 0  (offset 3) = {score(3, 0):.4f}   <- different offset")
    rq = rope(q, np.array([3.0]))
    print(f"\n||q|| before {np.linalg.norm(q):.4f}, after {np.linalg.norm(rq):.4f}"
          "   (a rotation: length unchanged)")

    print("\nwhere it is applied, straight from attn_sublayer's source:")
    src = inspect.getsource(attn_sublayer)
    print("  " + [l for l in src.splitlines() if "rope(" in l][0].strip())

    print("\nREAD THIS: the score between two tokens depends on how far apart")
    print("they are, not where they are. That is why it goes on Q and K, the")
    print("two things being dotted. V is the content handed over once a match")
    print("is made; rotating it would make the content depend on position.")


def demo_5_parameter_arithmetic():
    line("DEMO 5: count the parameters yourself")

    print(f"{'size':>4s}  {'dim':>5s}  {'layers':>6s}  {'ffn':>6s}"
          f"  {'count_params':>16s}  {'paper':>6s}")
    for name, c in LLAMA.items():
        n = count_params(c["dim"], c["n_layers"], llama_hidden(c["dim"]))
        print(f"{name:>4s}  {c['dim']:5d}  {c['n_layers']:6d}"
              f"  {llama_hidden(c['dim']):6d}  {n:16,d}  {c['paper']:>6s}")

    c = LLAMA["7B"]
    d, h, L = c["dim"], llama_hidden(c["dim"]), c["n_layers"]
    attn, ffn, norms = 4 * d * d, 3 * d * h, 2 * d
    total = count_params(d, L, h)
    rows = [(f"embedding    {VOCAB} x {d}", VOCAB * d),
            ("per layer    attn   4*d*d", attn),
            (f"             ffn    3*d*{h}", ffn),
            ("             norms  2*d", norms),
            (f"x {L} layers", L * (attn + ffn + norms)),
            ("final norm", d),
            (f"output head  {VOCAB} x {d}", VOCAB * d)]
    print("\n7B, where it goes:")
    for label, v in rows:
        print(f"  {label:<30s} = {v:14,d}")
    print(f"  {'total':<30s} = {total:14,d}  = {total / 1e9:.2f}B")

    with_bias = count_params(d, L, h, bias=True)
    print(f"\nBREAK: put every bias term back (bias=True):")
    print(f"  {with_bias:,} vs {total:,}: +{with_bias - total:,}"
          f" = +{100 * (with_bias / total - 1):.3f}%")

    print("\nREAD THIS: attention is 4d^2 (no GQA yet, that is LLaMA 2 at 70B),")
    print("the FFN is 3dh and is TWICE the attention. Everything else is")
    print("rounding. Biases would add 0.02%; they were dropped for a simpler")
    print("model.py, not for savings.")


def demo_6_the_tokens_argument():
    line("DEMO 6: 1T tokens for a 7B model is 7x what Chinchilla asked for")

    c = LLAMA["7B"]
    n = count_params(c["dim"], c["n_layers"], llama_hidden(c["dim"]))
    opt = chinchilla_tokens(n)
    print(f"LLaMA-7B parameters                 {n:.3g}")
    print(f"Chinchilla-optimal tokens, 20/param {opt:.3g}  ({opt / 1e9:.0f}B)")
    print(f"LLaMA-7B actually trained on        {c['tokens']:.3g}  (1T)"
          f"   {c['tokens'] / opt:.1f}x more")
    print(f"\ntraining compute ~ 6 * N * D:")
    print(f"  Chinchilla-optimal   {6 * n * opt:.3g} FLOPs")
    print(f"  what LLaMA spent     {6 * n * c['tokens']:.3g} FLOPs"
          f"   {c['tokens'] / opt:.1f}x more")

    gpt3 = 175e9
    print(f"\ninference compute ~ 2 * N per token:")
    print(f"  LLaMA-7B     {2 * n:.3g} FLOPs/token")
    print(f"  GPT-3 175B   {2 * gpt3:.3g} FLOPs/token   {gpt3 / n:.0f}x more")
    print(f"\ntokens per parameter:")
    print(f"  GPT-3 175B 300B / 175.0B = {300e9 / gpt3:5.1f}")
    for name in ("7B", "13B", "33B", "65B"):
        cc = LLAMA[name]
        nn = count_params(cc["dim"], cc["n_layers"], llama_hidden(cc["dim"]))
        print(f"  LLaMA-{name:<4s} {cc['tokens'] / 1e12:.1f}T / {nn / 1e9:4.1f}B"
              f" = {cc['tokens'] / nn:5.1f}")

    print("\nREAD THIS: Chinchilla answers 'given this training budget, what")
    print("size and how many tokens?' LLaMA asks a different question: 'given")
    print("this SERVING budget, how good can the model be?' You pay training")
    print("once and inference per token, forever. So spend more training on a")
    print("smaller model: past the optimum the loss keeps falling, just slower.")
    print("That is the whole reason a 13B model beat a 175B one.")


if __name__ == "__main__":
    demo_1_rmsnorm_vs_layernorm()
    demo_2_swiglu_vs_relu()
    demo_3_pre_vs_post_norm()
    demo_4_rope_on_q_and_k_only()
    demo_5_parameter_arithmetic()
    demo_6_the_tokens_argument()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  Start from the 2017 decoder-only transformer. Apply four hunks:
  1. norm      = RMSNorm, and put it BEFORE each sublayer (pre-norm)
  2. ffn       = SwiGLU: down(silu(gate(x)) * up(x)), hidden 2/3*4d -> x256
  3. position  = RoPE: rotate Q and K in every layer, add nothing at the input
  4. bias      = none. Every Linear is a bare matrix.
  Then train it on 1T-1.4T tokens of public data, 7x past compute-optimal,
  because you serve a model far more often than you train it.

  Sizes 7B, 13B, 33B, 65B. 13B beat GPT-3 175B on most benchmarks.
  Everything not listed is training machinery: AdamW, cosine schedule,
  xformers attention, activation checkpointing. Real, but not the idea.
""")
