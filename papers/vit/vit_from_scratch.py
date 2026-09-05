"""
An Image Is Worth 16x16 Words (Vision Transformer) -- for programmers.

Run it:      python3 papers/vit/vit_from_scratch.py
Debug it:    set a breakpoint in patchify and watch .shape change per line.

No torch. No autograd. No training. Just the forward pass, so you can SEE
the reshape happen. The transformer block is the one from ../attention/,
re-typed here so this file stands alone. Every function is under 20 lines.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=80)


# ---------------------------------------------------------------------------
# STAGE 0 -- a toy image. 32x32 pixels, 3 channels, values in [0, 1].
#
# Not random noise: a pattern, so you can SEE which pixels land in which
# patch. Red ramps left->right, green ramps top->bottom, blue is a
# checkerboard whose squares are exactly one patch (8x8) in size.
# Layout is (H, W, C): height, width, channel -- PIL and OpenCV order.
# ---------------------------------------------------------------------------
def toy_image(H=32, W=32, p=8):
    y, x = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3))
    img[:, :, 0] = x / (W - 1)                  # R: column index, scaled
    img[:, :, 1] = y / (H - 1)                  # G: row index, scaled
    img[:, :, 2] = ((y // p) + (x // p)) % 2    # B: patch-sized checkerboard
    return img


# ---------------------------------------------------------------------------
# STAGE 1 -- THE WHOLE PAPER. Section 3.1, first paragraph:
#
#     x in R^(H x W x C)  ->  x_p in R^(N x (P^2 . C)),   N = HW / P^2
#
# "Reshape the image into a sequence of flattened 2D patches." That's it.
# The method section IS this function. Everything after it is the encoder
# from Attention Is All You Need, unchanged.
# ---------------------------------------------------------------------------
def patchify(img, p):
    H, W, C = img.shape
    # (1) CUT: split each spatial axis into (how many patches, pixels each).
    #     (H, W, C) -> (H/p, p, W/p, p, C). Throws if p does not divide H, W.
    cut = img.reshape(H // p, p, W // p, p, C)
    # (2) GROUP: bring the two "which patch" axes together, and the two
    #     "which pixel inside it" axes together. This transpose is the line
    #     everyone skips. Skipping it is silent garbage (demo 5).
    grouped = cut.transpose(0, 2, 1, 3, 4)       # (H/p, W/p, p, p, C)
    # (3) FLATTEN: one row per patch. p*p pixels, R G B R G B ... in order.
    return grouped.reshape((H // p) * (W // p), p * p * C)


def unpatchify(patches, H, W, C, p):
    """The exact inverse. patchify then unpatchify is the identity."""
    grid = patches.reshape(H // p, W // p, p, p, C).transpose(0, 2, 1, 3, 4)
    return grid.reshape(H, W, C)


def patchify_wrong(img, p):
    """Same output shape as patchify. Never the same contents. Demo 5."""
    H, W, C = img.shape
    return img.reshape((H // p) * (W // p), p * p * C)   # no cut, no group


# ---------------------------------------------------------------------------
# STAGE 2 -- attention, copied from ../attention/. Nothing image-specific.
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))   # - max: no overflow
    return e / np.sum(e, axis=axis, keepdims=True)


def attention(Q, K, V):
    """softmax(QK^T / sqrt(d_k)) V -- the soft dictionary lookup."""
    weights = softmax(Q @ K.T / np.sqrt(Q.shape[-1]), axis=-1)
    return weights @ V, weights


def layernorm(x, eps=1e-5):
    """Each token's vector to mean 0, std 1. Numerical hygiene."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def gelu(x):
    """The MLP's activation: a smooth ReLU (tanh approximation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ---------------------------------------------------------------------------
# STAGE 3 -- one encoder block. Equations 2 and 3 in the paper:
#
#     z'_l = MSA(LN(z_{l-1})) + z_{l-1}       tokens talk to each other
#     z_l  = MLP(LN(z'_l))    + z'_l          each token thinks alone
#
# Same block as the 2017 paper with one change: LayerNorm goes BEFORE each
# sublayer (pre-norm), not after. Single head here; multi-head lives on the
# attention page and adds nothing image-specific.
# ---------------------------------------------------------------------------
class EncoderBlock:
    def __init__(self, D, seed):
        rng = np.random.default_rng(seed)
        s = 1 / np.sqrt(D)
        self.W_q, self.W_k, self.W_v, self.W_o = [
            rng.normal(0, s, (D, D)) for _ in range(4)]
        self.W_1 = rng.normal(0, s, (D, 4 * D))             # MLP: D -> 4D -> D
        self.W_2 = rng.normal(0, 1 / np.sqrt(4 * D), (4 * D, D))

    def __call__(self, z):
        h = layernorm(z)                                     # LN first
        a, weights = attention(h @ self.W_q, h @ self.W_k, h @ self.W_v)
        z = z + a @ self.W_o                                 # residual: a diff
        z = z + gelu(layernorm(z) @ self.W_1) @ self.W_2     # MLP, residual
        return z, weights


# ---------------------------------------------------------------------------
# STAGE 4 -- the model. Equation 1:
#
#     z_0 = [x_class; x_p^1 E; x_p^2 E; ... ; x_p^N E] + E_pos
#
# E        (p*p*C, D)  patch embedding. A matmul, not a lookup table,
#                      because pixels are continuous, not vocabulary ids.
# x_class  (1, D)      a learned token prepended to the sequence, BERT-style.
#                      Its output row is read as the whole image's summary.
# E_pos    (N+1, D)    learned 1D position embeddings: a table of N+1
#                      vectors. The paper tried 2D-aware ones. No gain.
#
# All three are LEARNED in the paper. Random here, like W_q on the attention
# page: the mechanism is fixed, training only moves the numbers.
# ---------------------------------------------------------------------------
class ViT:
    def __init__(self, H, W, C, p, D, n_blocks, n_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.p, self.N = p, (H // p) * (W // p)
        self.E = rng.normal(0, 1 / np.sqrt(p * p * C), (p * p * C, D))
        self.cls = rng.normal(0, 0.02, (1, D))
        self.pos = rng.normal(0, 0.02, (self.N + 1, D))
        self.blocks = [EncoderBlock(D, seed=i + 1) for i in range(n_blocks)]
        self.W_head = rng.normal(0, 1 / np.sqrt(D), (D, n_classes))

    def __call__(self, img, use_pos=True):
        return self.forward(patchify(img, self.p), use_pos)  # (N, p*p*C) in

    def forward(self, patches, use_pos=True):
        tokens = patches @ self.E                            # (N, D)   x_p E
        z = np.concatenate([self.cls, tokens], axis=0)       # (N+1, D) prepend
        if use_pos:
            z = z + self.pos                                 # (N+1, D) + E_pos
        for block in self.blocks:
            z, weights = block(z)                            # (N+1, D) unchanged
        z = layernorm(z)                                     # Eq. 4: LN(z_L^0)
        return z[0] @ self.W_head, z, weights                # logits, rows, attn


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def show(name, shape, note=""):
    """One line of the shape trace: name, shape, what it means."""
    print(f"  {name:<9}{str(shape):<18}{note}")


def demo_1_the_reshape():
    """Every shape from pixels to logits, stepped by hand. This is the paper."""
    line("DEMO 1: an image becomes a sentence -- every shape on the way")

    img = toy_image()
    vit = ViT(32, 32, 3, p=8, D=64, n_blocks=1, n_classes=10)
    H, W, C, p = 32, 32, 3, 8

    # patchify(), one line at a time, so you can put a breakpoint on each
    cut = img.reshape(H // p, p, W // p, p, C)               # step (1)
    grouped = cut.transpose(0, 2, 1, 3, 4)                   # step (2)
    patches = grouped.reshape((H // p) * (W // p), p * p * C)  # step (3)
    assert np.array_equal(patches, patchify(img, p))

    # ViT.forward(), likewise
    tokens = patches @ vit.E
    z = np.concatenate([vit.cls, tokens], axis=0)
    z = z + vit.pos
    z, weights = vit.blocks[0](z)
    z = layernorm(z)
    logits = z[0] @ vit.W_head

    show("image", img.shape, "(H, W, C)")
    show("cut", cut.shape, "(H/p, p, W/p, p, C)")
    show("grouped", grouped.shape, "(H/p, W/p, p, p, C)")
    show("patches", patches.shape, "(N, p*p*C)          <- the sentence")
    show("tokens", tokens.shape, "(N, D)              patches @ E")
    show("+ cls", z.shape, "(N+1, D)            BERT-style")
    show("+ pos", z.shape, "(N+1, D)            added")
    show("block", z.shape, "(N+1, D)            shape unchanged")
    show("attn", weights.shape, "(N+1, N+1)          all patches see all")
    show("cls out", z[0].shape, "(D,)                the row we read")
    show("logits", logits.shape, "(n_classes,)")

    print("\npatch 0, first 6 values = pixel (0,0) then pixel (0,1), R G B each:")
    print("  ", patches[0, :6])
    back = unpatchify(patches, 32, 32, 3, 8)
    print("unpatchify(patchify(img)) == img?", np.array_equal(back, img))

    print("\nREAD THIS: 32x32x3 = 3072 numbers, regrouped into 16 rows of 192.")
    print("Nothing was computed until `patches @ E`. A patch is a word, the")
    print("row of 192 pixel values is its spelling, E is the embedding table.")
    print("From (16, 64) onward this is the attention page's (n_tokens, d_model).")


def demo_2_shuffle_the_jigsaw():
    """Attention is a set operation. Only the position table sees layout."""
    line("DEMO 2: shuffle the tiles -- without positions, the model can't tell")

    img = toy_image()
    perm = np.random.default_rng(1).permutation(16)
    shuffled = unpatchify(patchify(img, 8)[perm], 32, 32, 3, 8)
    print("tile permutation:", perm)
    print("shuffled image == original?", np.array_equal(img, shuffled))
    print()

    vit = ViT(32, 32, 3, p=8, D=64, n_blocks=1, n_classes=10)
    for use_pos in (False, True):
        _, za, _ = vit(img, use_pos=use_pos)
        _, zb, _ = vit(shuffled, use_pos=use_pos)
        diff = np.abs(za[0] - zb[0]).max()
        print(f"position embeddings {'ON ' if use_pos else 'OFF'}:"
              f" CLS max abs diff = {diff:<9.3g}"
              f" allclose? {np.allclose(za[0], zb[0])}")

    print("\nREAD THIS: same 16 tiles, different places. Position table off, the")
    print("CLS row agrees to 15 decimal places -- the 1e-15 is float rounding,")
    print("because the shuffle changed the order the attention sum runs in. Turn")
    print("the table on and the rows differ in the second decimal place. That")
    print("`+ self.pos` is the only thing telling the model where a patch was.")
    print("Same point as demo 3 on the attention page: a weighted sum is a set.")


def demo_3_which_row_do_you_read():
    """CLS, mean of patches, or one patch row: two are set-reads, one isn't."""
    line("DEMO 3: which row do you read at the end? CLS, mean-pool, or a patch")

    img = toy_image()
    perm = np.random.default_rng(1).permutation(16)
    shuffled = unpatchify(patchify(img, 8)[perm], 32, 32, 3, 8)
    vit = ViT(32, 32, 3, p=8, D=64, n_blocks=1, n_classes=10)
    _, za, _ = vit(img, use_pos=False)
    _, zb, _ = vit(shuffled, use_pos=False)

    reads = {
        "z[0]        CLS token      ": (za[0], zb[0]),
        "z[1:].mean  mean-pool      ": (za[1:].mean(0), zb[1:].mean(0)),
        "z[6]        patch token 6  ": (za[6], zb[6]),
    }
    print("max abs diff under the shuffle, position embeddings OFF:")
    for name, (a, b) in reads.items():
        print(f"  {name} {np.abs(a - b).max():.3g}")
    print(f"  (token 6 now holds tile {perm[5]}, not tile 5)")

    print("\nREAD THIS: CLS and mean-pool are both reads over the whole set, so")
    print("both survive the shuffle (1e-15 is float noise again). A single patch")
    print("row is a read at a position, so it moves by a real amount. The paper's")
    print("appendix tried mean-pool instead of CLS and found it works as well --")
    print("once the learning rate is re-tuned.")


def demo_4_patch_size_sets_the_bill():
    """Why 16x16. Tokens n = (224/p)^2, and attention costs n^2 per head."""
    line("DEMO 4: patch size sets the token count, and attention bills n^2")

    base = (224 // 16) ** 2
    print(f"  {'patch':>6} {'per token':>10} {'tokens n':>9} {'n x n':>14}"
          f" {'vs 16x16':>10}")
    for p in (1, 4, 8, 16, 32):
        n = (224 // p) ** 2
        print(f"  {p:>3}x{p:<2} {p * p * 3:>10} {n:>9} {n * n:>14,}"
              f" {n * n / (base * base):>9.2f}x")

    print("\nREAD THIS: halve the patch, quadruple the tokens, 16x the attention")
    print("matrix. One pixel per token is 2.5 billion scores per head per layer.")
    print("16x16 on 224x224 is 196 tokens: a paragraph, not a book. And")
    print("16*16*3 = 768, which is exactly ViT-Base's D, so E is square there.")


def demo_5_the_wrong_reshape():
    """Skip the transpose. Shapes pass. Contents are strips, not squares."""
    line("DEMO 5: the wrong reshape -- same shape, scrambled contents, no error")

    img = toy_image()
    right = patchify(img, 8)
    wrong = patchify_wrong(img, 8)
    print("shapes:", right.shape, wrong.shape,
          " equal?", right.shape == wrong.shape)

    print("\npatch 5, first 8 values (R G B R G B R G):")
    print("  right:", right[5, :8])
    print("  wrong:", wrong[5, :8])

    print("\nwhat is 'patch 5', really?")
    print("  right == img[8:16, 8:16]  an 8x8 square? ",
          np.array_equal(right[5], img[8:16, 8:16].ravel()))
    print("  wrong == img[10:12, :]    a 2x32 strip?  ",
          np.array_equal(wrong[5], img[10:12, :].ravel()))

    vit = ViT(32, 32, 3, p=8, D=64, n_blocks=1, n_classes=10)
    logits, _, _ = vit.forward(wrong)
    print("\nmodel runs on the wrong patches anyway: logits", logits.shape)

    print("\nREAD THIS: reshape without the transpose reads the image row by row,")
    print("so 'patch 5' is two full-width strips. Every shape check passes, the")
    print("model trains, and the accuracy is just quietly bad. The loud failure")
    print("is the good one: demo 6.")


def demo_6_does_not_divide():
    """A patch size that does not divide the image. NumPy refuses."""
    line("DEMO 6: a patch size that does not divide the image")

    img = toy_image()
    for p in (8, 5):
        try:
            print(f"  p={p}: patches {patchify(img, p).shape}")
        except ValueError as e:
            print(f"  p={p}: ValueError: {e}")

    print("\nREAD THIS: 32/5 is not an integer, so the cut in step (1) cannot")
    print("happen. Real pipelines resize to 224 first for exactly this reason.")


if __name__ == "__main__":
    demo_1_the_reshape()
    demo_2_shuffle_the_jigsaw()
    demo_3_which_row_do_you_read()
    demo_4_patch_size_sets_the_bill()
    demo_5_the_wrong_reshape()
    demo_6_does_not_divide()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. patchify   = reshape (H,W,C) -> (N, p*p*C). 16x16 on 224 -> 196 tokens
  2. embed      = patches @ E. A matmul, because pixels aren't vocab ids
  3. cls        = one learned token prepended; its output row is the answer
  4. pos        = learned 1D table added in, because attention is order-blind
  5. encoder    = the 2017 block, pre-norm, stacked 12 / 24 / 32 deep
  6. head       = LayerNorm(z[0]) @ W. Mean-pool works too, re-tune the LR
  7. the catch  = no locality, no translation equivariance built in. So it
                  loses to a ResNet on ImageNet-1k alone, and wins once
                  pretrained on ImageNet-21k (14M) or JFT-300M. Less
                  image-specific inductive bias: fewer assumptions, more data.

  Everything not listed above is training machinery: Adam, warmup, weight
  decay, fine-tuning at higher resolution. Real, but not the idea.
""")
