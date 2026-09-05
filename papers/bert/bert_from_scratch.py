"""
BERT: Pre-training of Deep Bidirectional Transformers -- for programmers.

Run it:      python3 bert_from_scratch.py
Debug it:    set a breakpoint in mask_batch and step through one sentence.

NumPy for the mechanism: the masking recipe and the attention masks.
PyTorch for one thing only: the training loop, because the paper's idea IS a
training objective and hand-written backprop would bury it. CPU, ~30 seconds.
"""

import copy
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(1)   # deterministic; a tiny model gains nothing from more
np.set_printoptions(precision=2, suppress=True, linewidth=100)


# ---------------------------------------------------------------------------
# STAGE 0 -- a toy language with a rule worth learning.
#
# A "sentence" is 8 bracket pairs: an opener (a..h) followed by ITS closer
# (A..H). Which opener comes next is a coin toss. So:
#   - a closer is predictable from the token on its LEFT     "b ?"  -> B
#   - an opener is predictable from the token on its RIGHT   "? E"  -> e
# Half the tokens need right context. A causal model can never see it.
# ---------------------------------------------------------------------------
K = 8                        # pair types
P = 8                        # pairs per sentence
CLS, MASK = 0, 1             # special tokens
OPEN0, CLOSE0 = 2, 2 + K     # openers are ids 2..9 (a..h), closers 10..17 (A..H)
VOCAB = 2 + 2 * K            # 18
L = 1 + 2 * P                # 17 tokens: [CLS] + 16
NAMES = (["[CLS]", "_"] + [chr(97 + i) for i in range(K)]
         + [chr(65 + i) for i in range(K)])


def make_sentences(n, rng):
    """(n, L) int array. Column 0 is always [CLS]."""
    openers = rng.integers(0, K, (n, P))
    s = np.empty((n, L), dtype=np.int64)
    s[:, 0] = CLS
    s[:, 1::2] = OPEN0 + openers
    s[:, 2::2] = CLOSE0 + openers
    return s


def show(ids):
    return " ".join(NAMES[i] for i in ids)


def row(cells):
    """Align one string per position under show(): the [CLS] column is 5 wide."""
    return " ".join(c.rjust(5) if j == 0 else c for j, c in enumerate(cells))


# ---------------------------------------------------------------------------
# STAGE 1 -- THE MASKING RECIPE. Section 3.1, Task #1, of the paper.
#
# Pick 15% of the real tokens. Of the picked ones:
#   80% -> replaced by [MASK]
#   10% -> replaced by a random token   (so the model can't trust its input)
#   10% -> left exactly as they were    (so it learns to represent real tokens)
# The loss is computed ONLY at the picked positions. Everything else is
# context. labels == -1 means "no loss here".
# ---------------------------------------------------------------------------
def mask_batch(ids, rng, p_pick=0.15, p_mask=0.8, p_random=0.1):
    corrupted, labels = ids.copy(), np.full_like(ids, -1)
    kind = np.zeros_like(ids)          # 0 untouched, 1 [MASK], 2 random, 3 kept
    picked = (rng.random(ids.shape) < p_pick) & (ids != CLS)   # never pick [CLS]
    labels[picked] = ids[picked]           # the answer key
    roll = rng.random(ids.shape)
    to_mask = picked & (roll < p_mask)
    to_random = picked & (roll >= p_mask) & (roll < p_mask + p_random)
    to_keep = picked & ~to_mask & ~to_random
    corrupted[to_mask] = MASK
    corrupted[to_random] = rng.integers(OPEN0, VOCAB, int(to_random.sum()))
    kind[to_mask], kind[to_random], kind[to_keep] = 1, 2, 3
    return corrupted, labels, kind


# ---------------------------------------------------------------------------
# STAGE 2 -- attention, copied from the attention page. Same four steps.
# The ONLY thing BERT changes is step (3): it passes no mask.
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)   # overflow safety
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K_, V, mask=None):
    scores = Q @ K_.T / np.sqrt(Q.shape[-1])            # (1) score, (2) scale
    if mask is not None:                             # (3) mask: GPT yes, BERT no
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)                   # (4) normalise + blend
    return weights @ V, weights


def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))          # True = allowed to look


# ---------------------------------------------------------------------------
# STAGE 3 -- the encoder, in torch, so it can train. One block is the same
# "attend -> add & norm -> feedforward -> add & norm" as the attention page.
# BERT-base stacks 12 of these at d=768 with 12 heads: 110M parameters.
# We stack 2 at d=32 with 2 heads: ~27k parameters. Same shape, smaller numbers.
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                nn.Linear(4 * d, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, attn_mask):
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + a)                 # tokens talk to each other
        return self.norm2(x + self.ff(x))     # each token thinks alone


class TinyBERT(nn.Module):
    def __init__(self, causal=False, d=32, heads=2, layers=2):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(L, d)         # BERT learns positions, not sines
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.mlm_head = nn.Linear(d, VOCAB)   # "which token was here?"
        # THE SWITCH. None = every token sees both sides (BERT).
        # -inf above the diagonal = every token sees only its left (GPT).
        self.attn_mask = (torch.triu(torch.full((L, L), float("-inf")), 1)
                          if causal else None)

    def encode(self, ids):                    # (batch, L) -> (batch, L, d)
        x = self.tok(ids) + self.pos(torch.arange(L))
        for block in self.blocks:
            x = block(x, self.attn_mask)
        return x

    def forward(self, ids):                   # (batch, L) -> (batch, L, VOCAB)
        return self.mlm_head(self.encode(ids))


# ---------------------------------------------------------------------------
# STAGE 4 -- the objective. Cross-entropy at the picked positions, nothing
# else. A cloze test, marked only on the blanks.
# ---------------------------------------------------------------------------
def mlm_loss(logits, labels):
    labels = torch.from_numpy(labels)
    if int((labels >= 0).sum()) == 0:         # nothing picked -> nothing to learn
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1),
                           ignore_index=-1)


def train_mlm(model, steps, rng, p_pick=0.15, p_mask=0.8, p_random=0.1,
              eval_set=None, log_every=200, lr=3e-3, batch=64):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        inp, labels, _ = mask_batch(make_sentences(batch, rng), rng,
                                    p_pick, p_mask, p_random)
        loss = mlm_loss(model(torch.from_numpy(inp)), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if eval_set is not None and (step % log_every == 0 or step == 1):
            acc = masked_accuracy(model, eval_set["inp"], eval_set["labels"])
            print(f"  step {step:4d}   loss {loss.item():.3f}   "
                  f"masked-token accuracy {acc:6.1%}")
    return model


# ---------------------------------------------------------------------------
# STAGE 5 -- fine-tuning. Bolt a classification head on the [CLS] output.
# Same library, different main(). The paper fine-tunes EVERY parameter.
# ---------------------------------------------------------------------------
class Classifier(nn.Module):
    def __init__(self, encoder, n_classes=2):
        super().__init__()
        self.encoder = encoder
        self.cls_head = nn.Linear(encoder.tok.embedding_dim, n_classes)

    def forward(self, ids):
        h = self.encoder.encode(ids)          # (batch, L, d)
        return self.cls_head(h[:, 0, :])      # row 0 is [CLS]: (batch, n_classes)


# ---------------------------------------------------------------------------
# helpers for evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, inp):
    return model(torch.from_numpy(inp)).argmax(-1).numpy()


def masked_accuracy(model, inp, labels, where=None):
    sel = (labels >= 0) if where is None else where
    pred = predict(model, inp)
    return float((pred[sel] == labels[sel]).mean())


def make_eval(rng, n=2000, **mask_kwargs):
    original = make_sentences(n, rng)
    inp, labels, kind = mask_batch(original, rng, **mask_kwargs)
    return dict(original=original, inp=inp, labels=labels, kind=kind)


def partner_intact(ev):
    """For each position: is its partner still showing its true token?"""
    pos = np.arange(L)
    partner = np.where(pos % 2 == 1, pos + 1, pos - 1)  # opener right, closer left
    partner[0] = 0
    return ev["inp"][:, partner] == ev["original"][:, partner]


def make_cola(n, rng):
    """Half the sentences get ONE closer swapped for a wrong one. 1 = clean."""
    s = make_sentences(n, rng)
    broken = rng.random(n) < 0.5
    which = 2 + 2 * rng.integers(0, P, n)                 # a closer column
    wrong = (s[np.arange(n), which] - CLOSE0 + rng.integers(1, K, n)) % K + CLOSE0
    s[broken, which[broken]] = wrong[broken]
    return s, (~broken).astype(np.int64)


def train_classifier(clf, steps, rng, lr, eval_x, eval_y, batch=64):
    params = [p for p in clf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for _ in range(steps):
        x, y = make_cola(batch, rng)
        loss = F.cross_entropy(clf(torch.from_numpy(x)), torch.from_numpy(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (clf(torch.from_numpy(eval_x)).argmax(-1).numpy() == eval_y).mean()
    return float(acc), sum(p.numel() for p in params)


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_1_masking_recipe():
    line("DEMO 1: the masking recipe -- 15%, then 80/10/10")
    rng = np.random.default_rng(1)
    sents = make_sentences(8, rng)
    inp, labels, kind = mask_batch(sents, rng)
    i = int(np.argmax((labels >= 0).sum(1)))          # the busiest of the 8
    print("original :", show(sents[i]))
    print("corrupted:", show(inp[i]))
    print("labels   :", row([NAMES[l] if l >= 0 else "." for l in labels[i]]))
    print("kind     :", row([".MRK"[k] for k in kind[i]]))
    print("           (M = [MASK]  R = random token  K = kept as-is  . = no loss)")

    big = make_sentences(10000, rng)
    _, _, kind = mask_batch(big, rng)
    n = big.size - len(big)                   # real tokens, [CLS] excluded
    print(f"\nover {len(big)} sentences ({n} real tokens):")
    for k, name, want in ((1, "[MASK]", 0.12), (2, "random", 0.015),
                          (3, "kept", 0.015)):
        frac = (kind == k).sum() / n
        print(f"  {name:7s} {frac:6.1%}   (recipe says {want:.1%})")
    print(f"  picked  {(kind > 0).sum() / n:6.1%}   (recipe says 15.0%)")
    print("\nREAD THIS: only the picked 15% ever get a loss. The other 85% are")
    print("context. And of the picked ones, one in five is NOT a [MASK] token, so")
    print("the model can never fully trust a token just because it is visible.")


def demo_2_bidirectional_vs_causal():
    line("DEMO 2: the same attention, with and without the causal mask")
    rng = np.random.default_rng(2)
    tokens = ["b", "B", "e", "E", "a"]
    X = rng.normal(0, 1, (len(tokens), 4))     # stand-in embeddings, Q = K = V = X

    for title, mask in (("no mask (BERT)", None),
                        ("causal mask (GPT)", causal_mask(len(tokens)))):
        _, w = scaled_dot_product_attention(X, X, X, mask=mask)
        print(f"{title} -- each row = what that token can see:")
        for t, r in zip(tokens, w):
            cells = " ".join(f"{v:.2f}" for v in r)
            print(f"  {t}  [{cells}]   sees {int((r > 0).sum())} of {len(tokens)}")
        print()
    print("READ THIS: identical code, one argument. Without the mask every row")
    print("is full: 'e' reads 'E' to its right. With it, 'b' sees only itself and")
    print("'e' has no idea an 'E' is coming. That argument is the whole paper.")


def demo_3_train_bert(eval_set):
    line("DEMO 3: pre-train the tiny BERT (bidirectional, 15%, 80/10/10)")
    rng = np.random.default_rng(3)
    model = train_mlm(TinyBERT(causal=False), 600, rng, eval_set=eval_set)

    inp, labels = eval_set["inp"], eval_set["labels"]
    intact = partner_intact(eval_set) & (labels >= 0)
    lost = ~partner_intact(eval_set) & (labels >= 0)
    acc_intact = masked_accuracy(model, inp, labels, intact)
    acc_lost = masked_accuracy(model, inp, labels, lost)
    print(f"\n  partner visible and intact : {acc_intact:6.1%}"
          f"  ({intact.sum()} targets)")
    print(f"  partner hidden or corrupted: {acc_lost:6.1%}  ({lost.sum()} targets,"
          f" chance = {1 / K:.1%})")

    print("\nfill in the blanks (held-out sentences):")
    pred = predict(model, inp)
    hit = (pred == labels) | (labels < 0)
    busy = (labels >= 0).sum(1) >= 2
    picks = [int(np.flatnonzero(busy & hit.all(1))[0]),
             int(np.flatnonzero(busy & hit.all(1))[1]),
             int(np.flatnonzero(~hit.all(1))[0])]
    for i in picks:
        picked = labels[i] >= 0
        filled = np.where(picked, pred[i], inp[i])
        marks = "".join("✓" if p == t else "✗"
                        for p, t in zip(pred[i][picked], labels[i][picked]))
        print("  original :", show(eval_set["original"][i]))
        print("  input    :", show(inp[i]))
        print("  predicted:", show(filled), " ", marks)
    print("\nREAD THIS: the misses are the pairs where BOTH halves got hidden.")
    print("No context, no answer. Everything with one visible half is solved.")
    return model


def demo_4_break_causal(eval_set, bert):
    line("DEMO 4: BREAK IT -- same model, same data, keep the causal mask")
    rng = np.random.default_rng(3)                    # identical data stream
    gpt = train_mlm(TinyBERT(causal=True), 600, rng, eval_set=eval_set)

    inp, labels = eval_set["inp"], eval_set["labels"]
    openers = (labels >= 0) & (np.arange(L) % 2 == 1)
    closers = (labels >= 0) & (np.arange(L) % 2 == 0)
    print(f"\n  {'':30s}{'bidirectional':>15s}{'causal':>10s}")
    for name, sel in (("all masked tokens", None),
                      ("  closers (clue on the LEFT)", closers),
                      ("  openers (clue on the RIGHT)", openers)):
        print(f"  {name:30s}{masked_accuracy(bert, inp, labels, sel):>15.1%}"
              f"{masked_accuracy(gpt, inp, labels, sel):>10.1%}")
    print(f"  chance = {1 / K:.1%}")
    print("\nREAD THIS: the causal model closes brackets fine -- the opener is on")
    print("its left. It cannot guess an opener, because the only evidence is the")
    print("closer to its RIGHT, and the mask hides it. That one number is the")
    print("paper's argument for dropping the mask.")


def demo_5_break_masking(eval_set, bert):
    line("DEMO 5: BREAK IT -- three ways to get the masking recipe wrong")
    inp, labels, kind = eval_set["inp"], eval_set["labels"], eval_set["kind"]

    print("(a) always [MASK], never random/kept: p_mask=1.0, p_random=0.0")
    rng = np.random.default_rng(3)
    always = train_mlm(TinyBERT(), 600, rng, p_mask=1.0, p_random=0.0)
    print(f"  {'accuracy at positions that were...':36s}"
          f"{'80/10/10':>10s}{'100/0/0':>10s}")
    for k, name in ((1, "[MASK]"), (2, "a random token"), (3, "kept as-is")):
        sel = kind == k
        print(f"    {name:34s}{masked_accuracy(bert, inp, labels, sel):>10.1%}"
              f"{masked_accuracy(always, inp, labels, sel):>10.1%}")
    print("  the 100/0/0 model has never been marked on a visible token.")
    print("  Fine-tuning will show it nothing but visible tokens. That is the")
    print("  mismatch.")

    print("\n(b) pick 50% instead of 15%: p_pick=0.5")
    rng = np.random.default_rng(3)
    half = train_mlm(TinyBERT(), 600, rng, p_pick=0.5)
    ev50 = make_eval(np.random.default_rng(9), p_pick=0.5)
    intact = partner_intact(ev50) & (ev50["labels"] >= 0)
    acc15 = masked_accuracy(half, inp, labels)
    acc15_bert = masked_accuracy(bert, inp, labels)
    print(f"  accuracy on 15%-masked eval : {acc15:6.1%}"
          f"   (the 15% model: {acc15_bert:6.1%})")
    print(f"  accuracy on 50%-masked eval : "
          f"{masked_accuracy(half, ev50['inp'], ev50['labels']):6.1%}")
    print(f"  ...where the partner is intact: "
          f"{masked_accuracy(half, ev50['inp'], ev50['labels'], intact):6.1%}"
          f"  (only {intact.sum() / (ev50['labels'] >= 0).sum():.0%} of targets)")
    print("  the rule still gets learned. The TASK got worse: nearly half the")
    print("  blanks have had their only clue blanked out too.")

    print("\n(c) pick 0%: p_pick=0.0")
    rng = np.random.default_rng(3)
    nothing = train_mlm(TinyBERT(), 300, rng, p_pick=0.0, eval_set=eval_set)
    acc0 = masked_accuracy(nothing, inp, labels)
    print(f"  final masked-token accuracy: {acc0:6.1%}   (chance = {1 / K:.1%})")
    print("  loss 0.000 from step 1 and no gradient. Nothing was hidden, so")
    print("  nothing had to be inferred. The loss is only on the blanks. No")
    print("  blanks, no loss.")


def demo_6_cls_and_finetune(bert):
    line("DEMO 6: [CLS] and fine-tuning -- same library, different main()")
    rng = np.random.default_rng(6)
    x, y = make_cola(64, rng)
    with torch.no_grad():
        h = bert.encode(torch.from_numpy(x))
        logits = Classifier(bert)(torch.from_numpy(x))
    print(f"  encoder output       {tuple(h.shape)}   (batch, L, d)")
    print(f"  [CLS] row, h[:, 0]   {tuple(h[:, 0].shape)}       (batch, d)"
          "  <- one vector per sentence")
    print(f"  classifier logits    {tuple(logits.shape)}        (batch, classes)")
    print("\ntask: is the sentence grammatical? (one closer swapped -> label 0)")
    print("  ", show(x[0]), " label", y[0])
    print("  ", show(x[1]), " label", y[1])

    eval_x, eval_y = make_cola(2000, np.random.default_rng(7))
    frozen = Classifier(copy.deepcopy(bert))
    for p in frozen.encoder.parameters():
        p.requires_grad_(False)
    runs = (("head only, encoder frozen", frozen, 3e-3),
            ("everything (the paper)", Classifier(copy.deepcopy(bert)), 1e-3),
            ("everything, NO pre-training", Classifier(TinyBERT()), 1e-3))
    print(f"\n  {'300 steps of fine-tuning':30s}"
          f"{'trainable params':>18s}{'accuracy':>10s}")
    for name, clf, lr in runs:
        acc, n = train_classifier(clf, 300, np.random.default_rng(8), lr,
                                  eval_x, eval_y)
        print(f"  {name:30s}{n:>18d}{acc:>10.1%}")
    print("  chance = 50.0%")
    print("\nREAD THIS: [CLS] was never given a job during pre-training (no NSP")
    print("here), so a linear head on the frozen encoder finds nothing. Let the")
    print("gradients into the encoder and it learns the job in 300 steps -- but")
    print("only if it was pre-trained first. Random init, same 300 steps: chance.")
    print("Pre-training makes fine-tuning cheap; fine-tuning ALL of it makes")
    print("pre-training usable. The paper does both, on all 110M parameters.")


if __name__ == "__main__":
    import time
    t_start = time.time()
    EVAL = make_eval(np.random.default_rng(42))   # 2000 held-out sentences, 15%

    demo_1_masking_recipe()
    demo_2_bidirectional_vs_causal()
    bert = demo_3_train_bert(EVAL)
    demo_4_break_causal(EVAL, bert)
    demo_5_break_masking(EVAL, bert)
    demo_6_cls_and_finetune(bert)

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. encoder    = the attention page's encoder, causal mask REMOVED
  2. MLM        = hide 15% of tokens (80% [MASK], 10% random, 10% kept),
                  loss only on the hidden ones. A cloze test on the internet.
  3. NSP        = "did sentence B follow sentence A?", read off [CLS].
                  RoBERTa later showed you can drop it.
  4. [CLS]      = one extra token whose output vector stands for the sentence
  5. fine-tune  = new small head per task, then train EVERY parameter
  6. BERT-base  = 12 layers, 768 wide, 12 heads, 110M params
  7. the cost   = it can't generate text. That job went to GPT.
""")
    print(f"total wall time: {time.time() - t_start:.1f}s", file=sys.stderr)
