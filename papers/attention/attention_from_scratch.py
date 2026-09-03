"""
Attention Is All You Need -- for programmers, not researchers.

Run it:      .venv/bin/python Notebooks/Transformer/attention_from_scratch.py
Debug it:    set a breakpoint in scaled_dot_product_attention and step through.

No torch. No autograd. No training. Just the forward pass, so you can SEE
the numbers move. Every function is <15 lines.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- a toy "vocabulary". Real models learn these; we hardcode them so
# the attention weights come out human-readable.
#
# 4 dimensions, each one a hand-assigned "meaning slot":
#   [0] animal-ness   [1] action-ness   [2] food-ness   [3] pronoun-ness
# ---------------------------------------------------------------------------
EMBEDDINGS = {
    "the":  np.array([0.0, 0.0, 0.0, 0.0]),
    "cat":  np.array([1.0, 0.0, 0.0, 0.0]),
    "ate":  np.array([0.0, 1.0, 0.0, 0.0]),
    "food": np.array([0.0, 0.0, 1.0, 0.0]),
    "it":   np.array([0.0, 0.0, 0.0, 1.0]),
}


def embed(tokens):
    """list[str] -> (n_tokens, d_model) matrix. One row per token."""
    return np.stack([EMBEDDINGS[t] for t in tokens])


# ---------------------------------------------------------------------------
# STAGE 1 -- softmax. Turns any row of numbers into percentages summing to 1.
#
# The `- max` is not in the paper's formula; it's the standard numerical-safety
# trick. exp(1000) overflows to inf; exp(1000-1000)=exp(0)=1. Subtracting the
# max changes nothing mathematically (it cancels in the ratio) but keeps floats
# in range. Same instinct as clamping before a division.
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# STAGE 2 -- THE WHOLE PAPER. Equation 1, section 3.2.1:
#
#     Attention(Q, K, V) = softmax( Q @ K.T / sqrt(d_k) ) @ V
#
# Read it as a 4-step pipeline. That's all it is.
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None, verbose=False):
    d_k = Q.shape[-1]

    # (1) SCORE: how well does each query match each key?
    #     Q is (n_q, d_k), K.T is (d_k, n_k)  ->  scores is (n_q, n_k)
    #     scores[i][j] = "how much should token i care about token j"
    #     A dot product IS a similarity measure: same direction -> big number.
    scores = Q @ K.T

    # (2) SCALE: divide by sqrt(d_k).
    #     Why: dot products of d_k-dim vectors have variance ~d_k, so they grow
    #     as you widen the model. Feed huge numbers into softmax and it
    #     saturates -- one weight becomes ~1.0, the rest ~0.0, and the gradient
    #     through it vanishes. Dividing by sqrt(d_k) pulls variance back to ~1.
    #     Pure numerical hygiene. Footnote 4 in the paper.
    scores = scores / np.sqrt(d_k)

    # (3) MASK (optional): set forbidden positions to -inf so softmax gives
    #     them exactly 0. This is how a decoder is stopped from reading the
    #     future. -inf, not 0, because 0 is a perfectly ordinary score.
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    # (4) NORMALIZE + BLEND: percentages, then weighted average of the values.
    weights = softmax(scores, axis=-1)   # each ROW sums to 1.0
    output = weights @ V

    if verbose:
        print("  raw scores (QK^T):\n", scores)
        print("  attention weights (each row sums to 1):\n", weights)
        print("  row sums:", weights.sum(axis=-1))

    return output, weights


# ---------------------------------------------------------------------------
# STAGE 3 -- self-attention. "Self" just means Q, K and V all come from the
# SAME sentence, each through its own learned linear layer.
#
# Why three different projections of the same input? Because a token plays
# three different roles at once:
#   Q = what I'm looking for       ("I'm a pronoun, I need an antecedent")
#   K = what I advertise            ("I'm a noun, I can BE an antecedent")
#   V = what I hand over if chosen  (the actual content)
# One shared vector couldn't separate those roles.
# ---------------------------------------------------------------------------
class SelfAttention:
    def __init__(self, d_model, d_k, seed=0):
        rng = np.random.default_rng(seed)
        # In a real model these are LEARNED by gradient descent. Random here.
        self.W_q = rng.normal(0, 0.5, (d_model, d_k))
        self.W_k = rng.normal(0, 0.5, (d_model, d_k))
        self.W_v = rng.normal(0, 0.5, (d_model, d_k))

    def __call__(self, X, mask=None, verbose=False):
        Q = X @ self.W_q          # (n, d_model) @ (d_model, d_k) -> (n, d_k)
        K = X @ self.W_k
        V = X @ self.W_v
        return scaled_dot_product_attention(Q, K, V, mask, verbose)


# ---------------------------------------------------------------------------
# STAGE 4 -- multi-head. Run h independent attentions in parallel, concat.
#
# Why: ONE attention pass produces ONE weighted average -- it can only track
# one kind of relationship. 8 heads = 8 relationship types at once (grammar,
# coreference, position...). Mechanically identical to sharding a query across
# 8 workers with different indexes and merging the results.
#
# Note the model does NOT get wider: d_model=512 with 8 heads means each head
# works in 64 dims (512/8). Same total compute, split 8 ways. Section 3.2.2.
# ---------------------------------------------------------------------------
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        d_k = d_model // n_heads
        self.heads = [SelfAttention(d_model, d_k, seed=i) for i in range(n_heads)]
        rng = np.random.default_rng(99)
        self.W_o = rng.normal(0, 0.5, (d_model, d_model))  # the final mix

    def __call__(self, X, mask=None):
        outs, all_weights = [], []
        for h in self.heads:
            o, w = h(X, mask)
            outs.append(o)
            all_weights.append(w)
        concatenated = np.concatenate(outs, axis=-1)   # back to (n, d_model)
        return concatenated @ self.W_o, all_weights


# ---------------------------------------------------------------------------
# STAGE 5 -- positional encoding.
#
# THE PROBLEM: attention is a set operation. "cat ate food" and "food ate cat"
# produce identical outputs, because a weighted sum doesn't care about order.
# An RNN got order for free from its loop; we threw the loop away.
#
# THE FIX: add a unique, deterministic wave-pattern to each position BEFORE
# attention. Even dims use sin, odd dims use cos, wavelengths span 2pi..10000*2pi.
# Not learned -- just computed. Section 3.5.
# ---------------------------------------------------------------------------
def positional_encoding(n_positions, d_model):
    pos = np.arange(n_positions)[:, None]           # (n, 1)
    i = np.arange(d_model)[None, :]                 # (1, d)
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((n_positions, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])            # even dims
    pe[:, 1::2] = np.cos(angle[:, 1::2])            # odd dims
    return pe


# ---------------------------------------------------------------------------
# STAGE 6 -- causal mask. Lower-triangular boolean matrix.
#
# Row i may only attend to columns <= i. This is the single line that separates
# a translator (bidirectional, sees everything) from a text generator like GPT
# (autoregressive, must not peek at the answer during training).
# ---------------------------------------------------------------------------
def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_1_hand_built_lookup():
    """Attention with weights I chose by hand, so the output is predictable."""
    line("DEMO 1: attention is a soft dictionary lookup")

    tokens = ["the", "cat", "ate", "food", "it"]
    X = embed(tokens)
    print("tokens:", tokens)
    print("dims:   [animal, action, food, pronoun]\n")

    # Skip the learned projections. Use the embeddings directly as Q, K, V.
    # Query = the last token, "it". Its vector is [0,0,0,1] (pure pronoun).
    query = X[-1:]

    print('Query = "it", asking all 5 tokens "who are you?"')
    out, w = scaled_dot_product_attention(query, X, X, verbose=True)
    print("\noutput vector:", out[0])
    print("\nREAD THIS: every score is 0 except 'it' matching itself. A pronoun")
    print("vector is orthogonal to the animal/action/food vectors -- dot product")
    print("0. So softmax spreads weight almost evenly, with a bump on 'it'.")
    print("The model hasn't LEARNED anything yet. That is the point: the")
    print("mechanism is fixed, the W_q/W_k/W_v matrices are what training moves.")


def demo_2_scaling_matters():
    """Show empirically why the sqrt(d_k) division exists."""
    line("DEMO 2: why divide by sqrt(d_k)?")

    rng = np.random.default_rng(1)
    for d_k in (4, 64, 512):
        q = rng.normal(0, 1, (1, d_k))
        k = rng.normal(0, 1, (10, d_k))
        raw = (q @ k.T)[0]
        scaled = raw / np.sqrt(d_k)
        print(f"\nd_k = {d_k}")
        print(f"  raw    score std: {raw.std():6.2f}  ->  max softmax weight: {softmax(raw).max():.4f}")
        print(f"  scaled score std: {scaled.std():6.2f}  ->  max softmax weight: {softmax(scaled).max():.4f}")

    print("\nREAD THIS: unscaled, as d_k grows the scores spread out, softmax")
    print("saturates toward a hard argmax (one weight -> 1.0), and the gradient")
    print("through it collapses. Scaled, the distribution stays soft and")
    print("trainable no matter how wide the model gets.")


def demo_3_position_matters():
    """Prove attention is order-blind without positional encoding."""
    line("DEMO 3: attention alone cannot tell 'cat ate food' from 'food ate cat'")

    d_model = 4
    attn = SelfAttention(d_model, d_model, seed=7)

    a = embed(["cat", "ate", "food"])
    b = embed(["food", "ate", "cat"])

    out_a, _ = attn(a)
    out_b, _ = attn(b)

    # Sort rows so we compare the SET of outputs, ignoring row order.
    set_a = np.sort(out_a, axis=0)
    set_b = np.sort(out_b, axis=0)
    print("without positional encoding, output sets identical?",
          np.allclose(set_a, set_b))

    pe = positional_encoding(3, d_model)
    out_a_pe, _ = attn(a + pe)
    out_b_pe, _ = attn(b + pe)
    set_a_pe = np.sort(out_a_pe, axis=0)
    set_b_pe = np.sort(out_b_pe, axis=0)
    print("with    positional encoding, output sets identical?",
          np.allclose(set_a_pe, set_b_pe))

    print("\npositional encoding matrix (3 positions x 4 dims):")
    print(pe)
    print("\nREAD THIS: the same words in a different order produce the same bag")
    print("of outputs -- attention is permutation-equivariant, a set operation.")
    print("Adding the position waves breaks that tie. That single `+ pe` is the")
    print("only thing giving a transformer any notion of word order.")


def demo_4_causal_mask():
    line("DEMO 4: the causal mask -- how GPT is stopped from cheating")

    tokens = ["the", "cat", "ate", "food", "it"]
    X = embed(tokens) + positional_encoding(len(tokens), 4)
    attn = SelfAttention(4, 4, seed=3)

    print("mask (True = allowed to look):")
    m = causal_mask(len(tokens))
    print(m.astype(int))

    _, w = attn(X, mask=m)
    print("\nresulting attention weights:")
    print(w)
    print("\nREAD THIS: strictly lower-triangular. Row 0 ('the') can only see")
    print("itself. Row 4 ('it') sees everything before it. Zero upper triangle")
    print("means no token ever reads a future token. Remove the mask and you")
    print("have BERT; keep it and you have GPT. Same code otherwise.")


def demo_5_multihead():
    line("DEMO 5: multi-head -- 8 different questions, asked at once")

    tokens = ["the", "cat", "ate", "food", "it"]
    d_model, n_heads = 8, 4

    # Pad the 4-dim toy embeddings out to d_model=8.
    X = np.pad(embed(tokens), ((0, 0), (0, d_model - 4)))
    X = X + positional_encoding(len(tokens), d_model)

    mha = MultiHeadAttention(d_model, n_heads)
    out, all_w = mha(X, mask=causal_mask(len(tokens)))

    for i, w in enumerate(all_w):
        print(f"\nhead {i} -- what each token attends to (row = querying token):")
        for tok, row in zip(tokens, w):
            top = tokens[int(np.argmax(row))]
            print(f"   {tok:5s} -> strongest link: {top:5s}  weights: {row}")

    print(f"\nfinal output shape: {out.shape}  (n_tokens, d_model) -- unchanged")
    print("\nREAD THIS: each head has different random W_q/W_k/W_v, so each")
    print("computes a DIFFERENT attention pattern over the same sentence. Concat")
    print("them, multiply by W_o to mix, and the shape is identical to the input.")
    print("Identical shape in and out is why you can stack these 6, 12, 96 deep.")


if __name__ == "__main__":
    demo_1_hand_built_lookup()
    demo_2_scaling_matters()
    demo_3_position_matters()
    demo_4_causal_mask()
    demo_5_multihead()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. attention  = soft dict lookup: softmax(QK^T / sqrt(d_k)) @ V
  2. self-attn  = Q, K, V are three learned projections of the SAME input
  3. multi-head = h of those in parallel, concat, mix with W_o
  4. pos. enc.  = sine waves added in, because attention is order-blind
  5. mask       = lower-triangular, stops a decoder reading the future
  6. a block    = multi-head attn -> add & norm -> feedforward -> add & norm
  7. the model  = stack N of those blocks. N=6 in the paper.

  Everything not listed above is training machinery: Adam, warmup schedule,
  dropout, label smoothing. Real, but not the idea.
""")
