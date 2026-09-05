"""
Sentence-BERT and CLIP -- for programmers, not researchers.

Run it:      python3 embeddings_from_scratch.py
Debug it:    breakpoint in info_nce_loss and look at the (B, B) score matrix.

The maths is NumPy: cosine similarity, the anisotropy measurement, the
retrieval evaluation. Torch appears in exactly one place -- the training
loop -- because the idea of both papers IS a training objective, and
hand-rolled backprop would bury it. CPU only, a few seconds.
"""

import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- toy data.
#
# Every item is one "concept" -- a latent vector z. Each concept shows up in
# two different surface forms, the way one thing shows up as a photo and as a
# caption. Modality A is 32 raw features, modality B is 24. Different widths,
# different random mixing matrices, same underlying z.
#
# These stand in for images and text. They are not images and not text. The
# point is only that two unrelated input types share a hidden cause, which is
# exactly the assumption CLIP makes about a photo and its caption.
# ---------------------------------------------------------------------------
D_LATENT, D_A, D_B, D_EMB = 8, 32, 24, 16

_rng = np.random.default_rng(0)
MIX_A = _rng.normal(0, 1, (D_LATENT, D_A))
MIX_B = _rng.normal(0, 1, (D_LATENT, D_B))
# A large shared offset per modality: every A vector carries this, every B
# vector carries that. Real encoder outputs do the same thing.
BIAS_A = _rng.normal(0, 1, D_A) * 3.0
BIAS_B = _rng.normal(0, 1, D_B) * 3.0


def make_pairs(n, seed):
    """n matched (A, B) pairs. Row i of A and row i of B share a concept."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (n, D_LATENT))
    a = z @ MIX_A + BIAS_A + rng.normal(0, 1.2, (n, D_A))
    b = z @ MIX_B + BIAS_B + rng.normal(0, 1.2, (n, D_B))
    return a, b


# ---------------------------------------------------------------------------
# STAGE 1 -- cosine similarity, and the two things people get wrong about it.
#
# cos(u, v) = (u . v) / (|u| |v|). Dividing by the lengths is what makes it
# a similarity and not a dot product. Skip the division and the longest
# vector wins every time -- the same bug the RAG page demonstrates on
# unnormalised document scores.
# ---------------------------------------------------------------------------
def l2_normalise(X, eps=1e-9):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + eps)


def cosine_matrix(A, B):
    """(n, d) x (m, d) -> (n, m) of cosines. Normalise, then one matmul."""
    return l2_normalise(A) @ l2_normalise(B).T


def mean_pairwise_cosine(X):
    """Anisotropy, measured. Average cosine between distinct random vectors.

    If the vectors are spread over the sphere this sits near 0.0. If they all
    crowd into a narrow cone it sits near 1.0, and 'close' stops meaning
    anything -- everything is close to everything.
    """
    C = cosine_matrix(X, X)
    n = C.shape[0]
    off_diagonal = C[~np.eye(n, dtype=bool)]
    return float(off_diagonal.mean())


# ---------------------------------------------------------------------------
# STAGE 2 -- retrieval evaluation. The only number that settles an argument.
#
# Query with row i of A, score every row of B, and check whether row i comes
# out on top. Chance level is 1/n.
# ---------------------------------------------------------------------------
def top1_accuracy(A, B, normalise=True):
    scores = cosine_matrix(A, B) if normalise else A @ B.T
    return float((scores.argmax(axis=1) == np.arange(len(A))).mean())


# ---------------------------------------------------------------------------
# STAGE 3 -- the two towers. One encoder per input type, one shared output
# space. CLIP's image encoder and text encoder; Sentence-BERT's two copies of
# BERT with tied weights and a pooling step on top.
#
# Nothing here is clever. Two small MLPs. What matters is the objective they
# are trained under, not the architecture.
# ---------------------------------------------------------------------------
class Tower(nn.Module):
    def __init__(self, d_in, d_out=D_EMB, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out)
        )

    def forward(self, x):
        # L2-normalise on the way out, so the space we train is the space we
        # search: cosine similarity becomes a plain dot product.
        return nn.functional.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# STAGE 4 -- InfoNCE with in-batch negatives. The whole contribution, in six
# lines.
#
# For a batch of B matched pairs, build the (B, B) matrix of similarities.
# The diagonal holds the B correct pairings. Every off-diagonal cell is a
# negative you got for free -- no negative sampling, no mining, just the rest
# of the batch. Then it is a B-way classification problem, and the label of
# row i is i.
#
# The temperature divides the scores before the softmax. Small temperature =
# sharp, the loss cares only about the hardest negative. Large temperature =
# flat, the loss barely distinguishes anything.
# ---------------------------------------------------------------------------
def info_nce_loss(emb_a, emb_b, temperature):
    logits = emb_a @ emb_b.T / temperature      # (B, B) -- diagonal is correct
    labels = torch.arange(len(emb_a))
    loss_a = nn.functional.cross_entropy(logits, labels)      # A retrieves B
    loss_b = nn.functional.cross_entropy(logits.T, labels)    # B retrieves A
    return (loss_a + loss_b) / 2                # symmetric, as in CLIP


def train_towers(temperature=0.07, batch_size=64, steps=400, seed=0,
                 log_every=0):
    """Torch lives here and nowhere else. CPU, a couple of seconds."""
    torch.manual_seed(seed)
    a_np, b_np = make_pairs(1024, seed=1)
    A = torch.tensor(a_np, dtype=torch.float32)
    B = torch.tensor(b_np, dtype=torch.float32)

    tower_a, tower_b = Tower(D_A), Tower(D_B)
    params = list(tower_a.parameters()) + list(tower_b.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    g = torch.Generator().manual_seed(seed)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(A), (batch_size,), generator=g)
        loss = info_nce_loss(tower_a(A[idx]), tower_b(B[idx]), temperature)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step == 1 or step % log_every == 0):
            print(f"  step {step:4d}   loss {loss.item():.4f}")
    return tower_a, tower_b


def encode(tower, x_np):
    """Torch model -> NumPy vectors. Everything downstream is NumPy again."""
    with torch.no_grad():
        return tower(torch.tensor(x_np, dtype=torch.float32)).numpy()


def untrained_towers(seed=0):
    """Randomly initialised encoders. The stand-in for raw CLS vectors:
    a real network that was never trained for this distance."""
    torch.manual_seed(seed)
    return Tower(D_A), Tower(D_B)


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


EVAL_A, EVAL_B = make_pairs(200, seed=42)


def demo_1_anisotropy():
    """Untrained vectors all point roughly the same way."""
    line("DEMO 1: the cone -- why raw encoder output is not an embedding")

    ta, _ = untrained_towers()
    raw = encode(ta, EVAL_A)
    print(f"untrained encoder, mean pairwise cosine: {mean_pairwise_cosine(raw):+.4f}")
    print(f"                   min / max cosine:     "
          f"{cosine_matrix(raw, raw).min():+.4f} / "
          f"{cosine_matrix(raw, raw)[~np.eye(200, dtype=bool)].max():+.4f}")

    ta2, _ = train_towers()
    trained = encode(ta2, EVAL_A)
    print(f"\ntrained   encoder, mean pairwise cosine: "
          f"{mean_pairwise_cosine(trained):+.4f}")
    print(f"                   min / max cosine:     "
          f"{cosine_matrix(trained, trained).min():+.4f} / "
          f"{cosine_matrix(trained, trained)[~np.eye(200, dtype=bool)].max():+.4f}")

    print("\nREAD THIS: before training, two unrelated items average cosine 0.68")
    print("and some pairs sit above 0.98. Everything looks similar to")
    print("everything, so 'nearest' is mostly noise. That is the anisotropy")
    print("Sentence-BERT is answering: BERT's CLS vector is not a sentence")
    print("embedding, and cosine between raw outputs is close to meaningless.")
    print("After training, the cosines spread across the whole range.")


def demo_2_the_objective():
    """Watch InfoNCE fall on data where the right pairing is known."""
    line("DEMO 2: InfoNCE with in-batch negatives -- the loss falling")

    print("batch 64, temperature 0.07. Chance loss = ln(64) = "
          f"{np.log(64):.4f}\n")
    train_towers(log_every=50)
    print("\nREAD THIS: the batch is the negative set. For 64 pairs you get a")
    print("64x64 score matrix, 64 correct cells on the diagonal and 4032 free")
    print("negatives off it. Cross-entropy with labels 0..63. That is all the")
    print("objective is, and it is the whole of both papers' contribution.")


def demo_3_retrieval_before_after():
    """The argument, in two numbers."""
    line("DEMO 3: retrieval accuracy, untrained vs trained")

    ua, ub = untrained_towers()
    acc_before = top1_accuracy(encode(ua, EVAL_A), encode(ub, EVAL_B))
    ta, tb = train_towers()
    acc_after = top1_accuracy(encode(ta, EVAL_A), encode(tb, EVAL_B))

    print(f"chance (1/200)                 : {1/200:.4f}")
    print(f"untrained encoders, top-1      : {acc_before:.4f}")
    print(f"trained encoders,   top-1      : {acc_after:.4f}")
    print("\nREAD THIS: same architecture, same data, same cosine. The only")
    print("difference is that one pair of encoders was trained for the property")
    print("we then measure. You do not get a usable distance by accident.")


def demo_4_temperature():
    """Sweep the temperature. Both extremes fail, differently."""
    line("DEMO 4: temperature -- both ends of the dial are broken")

    print(f"{'tau':>6}  {'final loss':>10}  {'top-1':>7}  {'sim std':>8}"
          f"  {'mean top score':>14}")
    for tau in (0.005, 0.02, 0.07, 0.3, 1.0, 10.0):
        ta, tb = train_towers(temperature=tau)
        ea, eb = encode(ta, EVAL_A), encode(tb, EVAL_B)
        sims = cosine_matrix(ea, eb)
        acc = top1_accuracy(ea, eb)
        loss = float(info_nce_loss(torch.tensor(ea[:64]),
                                   torch.tensor(eb[:64]), tau))
        print(f"{tau:>6.3f}  {loss:>10.4f}  {acc:>7.4f}  {sims.std():>8.4f}"
              f"  {sims.max(axis=1).mean():>14.4f}")

    print("\nREAD THIS: 'sim std' is how spread out the similarity distribution")
    print("is. Too cold and the loss chases single hard negatives and the space")
    print("collapses to a few directions; too warm and the gradient cannot tell")
    print("a right pair from a wrong one, so nothing separates. The usable band")
    print("is broad and the ends are cliffs. CLIP does not pick a number: it")
    print("makes the temperature a learned parameter and lets training find it.")


def demo_5_normalisation():
    """Drop the L2 division and magnitude eats the ranking."""
    line("DEMO 5: L2 normalisation -- or the longest vector wins")

    ta, tb = train_towers()
    ea, eb = encode(ta, EVAL_A), encode(tb, EVAL_B)

    # Give the items different lengths, the way documents have different
    # word counts. Cosine ignores this. A raw dot product does not.
    scale = np.linspace(1.0, 8.0, len(eb))[:, None]
    eb_scaled = eb * scale

    print(f"cosine (normalised),   top-1 : {top1_accuracy(ea, eb_scaled):.4f}")
    print(f"raw dot product,       top-1 : "
          f"{top1_accuracy(ea, eb_scaled, normalise=False):.4f}")
    winners = (ea @ eb_scaled.T).argmax(axis=1)
    print(f"raw dot product, distinct items ever returned: "
          f"{len(np.unique(winners))} of {len(eb)}")
    print(f"raw dot product, most-returned item index    : "
          f"{np.bincount(winners).argmax()} (the longest vectors sit at 199)")

    print("\nREAD THIS: the same bug the RAG page shows on documents. Without")
    print("the division by |v|, the score is length times direction, and length")
    print("has nothing to do with relevance. One long item wins every query.")


def demo_6_two_towers():
    """A query from one side retrieving the right item from the other."""
    line("DEMO 6: two towers, one space -- CLIP's trick")

    ta, tb = train_towers()
    ea, eb = encode(ta, EVAL_A), encode(tb, EVAL_B)
    sims = cosine_matrix(ea, eb)

    print("modality A is 32 features, modality B is 24. They stand in for an")
    print("image and its caption. After training they live in one 16-dim space.\n")
    print(f"{'query (A)':>10}  {'best match (B)':>15}  {'its cosine':>11}"
          f"  {'runner-up':>10}  {'correct?':>9}")
    for i in range(6):
        order = np.argsort(-sims[i])
        print(f"{i:>10}  {order[0]:>15}  {sims[i, order[0]]:>11.4f}"
              f"  {sims[i, order[1]]:>10.4f}  {str(order[0] == i):>9}")

    print(f"\nA->B top-1: {top1_accuracy(ea, eb):.4f}     "
          f"B->A top-1: {top1_accuracy(eb, ea):.4f}")
    print("\nREAD THIS: nothing in either encoder knows about the other's input")
    print("format. They only ever met through the loss. That is how zero-shot")
    print("classification works in CLIP: embed the image, embed 'a photo of a")
    print("{class}' for every class, take the nearest text.")


def demo_7_breaks():
    """Three breaks, printed next to the working version."""
    line("DEMO 7: break it on purpose")

    ta, tb = train_towers()
    good = top1_accuracy(encode(ta, EVAL_A), encode(tb, EVAL_B))
    print(f"baseline (batch 64, tau 0.07)          top-1: {good:.4f}\n")

    # BREAK 1 -- batch too small, so almost no in-batch negatives.
    for bs in (2, 4, 16, 64):
        ta_s, tb_s = train_towers(batch_size=bs)
        acc = top1_accuracy(encode(ta_s, EVAL_A), encode(tb_s, EVAL_B))
        print(f"batch size {bs:>3} ({bs - 1:>2} negatives per query) top-1: {acc:.4f}")

    # BREAK 2 -- temperature far too high.
    ta_h, tb_h = train_towers(temperature=10.0)
    print(f"\ntemperature 10.0 (far too warm)        top-1: "
          f"{top1_accuracy(encode(ta_h, EVAL_A), encode(tb_h, EVAL_B)):.4f}")

    # BREAK 3 -- mixing two models. Same architecture, same data, other seed.
    ta2, tb2 = train_towers(seed=7)
    mixed = top1_accuracy(encode(ta, EVAL_A), encode(tb2, EVAL_B))
    print(f"\nquery encoder from run A, index from run B (seed 7):")
    print(f"  matched pair of encoders             top-1: {good:.4f}")
    print(f"  mismatched encoders                  top-1: {mixed:.4f}   "
          f"(chance {1/200:.4f})")
    print("\nREAD THIS: the last one is the production incident. Two runs of the")
    print("SAME code on the SAME data produce two unrelated spaces. Re-embed")
    print("half your corpus with a new model version and the halves cannot be")
    print("compared. Nothing errors. The numbers just stop meaning anything.")


if __name__ == "__main__":
    demo_1_anisotropy()
    demo_2_the_objective()
    demo_3_retrieval_before_after()
    demo_4_temperature()
    demo_5_normalisation()
    demo_6_two_towers()
    demo_7_breaks()

    line("THE WHOLE THING, COMPRESSED")
    print("""
  1. a network trained for something else does not hand you a usable
     distance -- raw output crowds into a narrow cone
  2. so train for the property you want: contrastive, InfoNCE
  3. negatives are free -- the rest of the batch is the negative set,
     so a bigger batch is a harder and better task
  4. temperature sets how sharp the softmax over similarities is;
     both extremes fail
  5. L2-normalise, or length beats relevance
  6. two towers, two input types, one space -- that is CLIP
  7. embeddings from different models are not comparable, ever

  Sentence-BERT's other half is speed: a bi-encoder embeds each item once
  and compares vectors, instead of running a cross-encoder over every pair.
  A cross-encoder is usually more accurate. That is why rerankers exist.
""")
