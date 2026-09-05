"""
GPT-1, GPT-2, GPT-3 -- for programmers, not researchers.

Run it:      python3 gpt_from_scratch.py
Debug it:    set a breakpoint in generate() and watch the sequence grow.

The mechanism -- decoder block, causal mask, the generation loop -- is plain
NumPy, forward pass only, so you can SEE the shapes. Torch appears for the one
thing NumPy can't do cheaply: the training loop. CPU, tiny model, ~30 seconds.
Every function is short.
"""

import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- the data. Plain English sentences I generate from four short word
# lists, so the model has a GRAMMAR to learn and cannot just memorise. Every
# sentence is subject + verb + object + adjunct, each drawn uniformly, so the
# best possible loss is known: 4 * ln(6) nats per sentence, spread over its
# characters. One "token" = one character, lowercase, ~30 symbols. Real GPTs
# use byte-pair encoding (~50k tokens); the loop is identical, the dict bigger.
# ---------------------------------------------------------------------------
SUBJECTS = ["the model", "the prompt", "the loop", "the mask", "a token", "the reader"]
VERBS = ["predicts", "reads", "writes", "copies", "hides", "feeds"]
OBJECTS = ["the next token", "the answer", "the future", "the prefix",
           "its own output", "the whole internet"]
ADJUNCTS = ["", " again", " slowly", " at scale", " for free", " by hand"]


def make_corpus(n_sentences=400, seed=0):
    rng = np.random.default_rng(seed)
    pick = lambda xs: xs[rng.integers(len(xs))]
    sents = [f"{pick(SUBJECTS)} {pick(VERBS)} {pick(OBJECTS)}{pick(ADJUNCTS)}."
             for _ in range(n_sentences)]
    paras = [" ".join(sents[i:i + 4]) for i in range(0, n_sentences, 4)]
    return "\n".join(paras) + "\n"


TEXT = make_corpus()
LOSS_FLOOR = 4 * math.log(6) / (len(TEXT) / 400)   # nats per char, roughly

# GPT-3's trick, as a string. The examples are IN the input. No weights move.
FEWSHOT = """\
english: cat -> french: chat
english: dog -> french: chien
english: house -> french: maison
english: cheese -> french:"""

chars = sorted(set(TEXT + FEWSHOT))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# STAGE 1 -- attention, copied from the attention page. Nothing new here.
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)   # overflow safety
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)   # -inf -> exactly 0 weight
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def causal_mask(n):
    # row i may look at columns <= i. The future is -inf. This one line is
    # what makes "predict the next token" an honest exam and not an open book.
    return np.tril(np.ones((n, n), dtype=bool))


def layer_norm(x, eps=1e-5):
    # numerical hygiene: each token's vector gets mean 0, variance 1
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def gelu(x):
    # GPT's activation. A ReLU with the corner sanded off. Not the idea.
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


# ---------------------------------------------------------------------------
# STAGE 2 -- the decoder block. Multi-head CAUSAL self-attention, then a
# per-token MLP, each wrapped in a skip link. GPT-2 puts LayerNorm on the way
# IN to each sub-layer (the 2017 transformer put it on the way out).
# ---------------------------------------------------------------------------
def init_params(vocab, block_size, d_model, n_heads, seed=0):
    rng = np.random.default_rng(seed)
    n = lambda *shape: rng.normal(0, 0.1, shape)   # real models LEARN these
    return dict(
        tok_emb=n(vocab, d_model), pos_emb=n(block_size, d_model),
        W_q=n(d_model, d_model), W_k=n(d_model, d_model),
        W_v=n(d_model, d_model), W_o=n(d_model, d_model),
        W_1=n(d_model, 4 * d_model), W_2=n(4 * d_model, d_model),
        W_out=n(d_model, vocab), n_heads=n_heads, block_size=block_size)


def multi_head_attention(x, p, causal=True):
    T, d = x.shape
    h, dk = p["n_heads"], d // p["n_heads"]
    Q, K, V = x @ p["W_q"], x @ p["W_k"], x @ p["W_v"]
    mask = causal_mask(T) if causal else None       # causal=False -> encoder
    outs, weights = [], []
    for i in range(h):                              # h heads, dk dims each
        s = slice(i * dk, (i + 1) * dk)
        o, w = scaled_dot_product_attention(Q[:, s], K[:, s], V[:, s], mask)
        outs.append(o)
        weights.append(w)
    return np.concatenate(outs, axis=-1) @ p["W_o"], weights


def decoder_block(x, p, causal=True):
    a, weights = multi_head_attention(layer_norm(x), p, causal)
    x = x + a                                        # tokens talk (backwards only)
    x = x + gelu(layer_norm(x) @ p["W_1"]) @ p["W_2"]  # each token thinks alone
    return x, weights                                # shape unchanged -> stackable


def gpt_forward(tokens, p, causal=True):
    T = len(tokens)
    assert T <= p["block_size"], f"context {T} > block_size {p['block_size']}"
    x = p["tok_emb"][tokens] + p["pos_emb"][:T]      # (T, d): what + where
    x, weights = decoder_block(x, p, causal)         # a real GPT stacks 12..96
    logits = layer_norm(x) @ p["W_out"]              # (T, vocab): a guess PER row
    return logits, weights


# ---------------------------------------------------------------------------
# STAGE 3 -- THE WHOLE PRODUCT. A while loop that eats its own output.
# ---------------------------------------------------------------------------
def generate(p, prefix, n_new, temperature=0.0, rng=None, trace=False):
    tokens = list(prefix)
    for step in range(n_new):
        ctx = tokens[-p["block_size"]:]              # crop: no memory past this
        logits, _ = gpt_forward(np.array(ctx), p)
        row = logits[-1]                             # only the LAST row matters
        if temperature == 0:
            nxt = int(np.argmax(row))                # greedy
        else:
            nxt = int(rng.choice(len(row), p=softmax(row / temperature)))
        tokens.append(nxt)                           # feed it back in
        if trace:
            print(f"  step {step}: T={len(ctx)} tokens -> logits{logits.shape}"
                  f" -> argmax of last row = {nxt:2d} = {decode([nxt])!r}")
    return tokens


# ---------------------------------------------------------------------------
# STAGE 4 -- the same model in torch, because it needs to be TRAINED and I am
# not hand-writing backprop through a transformer. Line for line the same
# forward pass as above, batched, with `causal` as a switch so we can cheat.
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, n_heads, block_size, causal):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * d, d))
        self.h, self.causal = n_heads, causal
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def attn(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (t.view(B, T, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        s = q @ k.transpose(-2, -1) / math.sqrt(d // self.h)
        if self.causal:
            s = s.masked_fill(~self.mask[:T, :T], float("-inf"))
        y = F.softmax(s, dim=-1) @ v
        return self.proj(y.transpose(1, 2).reshape(B, T, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, block_size, d=64, n_heads=4, n_layers=2, causal=True):
        super().__init__()
        self.tok_emb, self.pos_emb = nn.Embedding(vocab, d), nn.Embedding(block_size, d)
        self.blocks = nn.Sequential(*[Block(d, n_heads, block_size, causal)
                                      for _ in range(n_layers)])
        self.ln_f, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))   # IndexError if T > block
        return self.head(self.ln_f(self.blocks(x)))              # (B, T, vocab)


def get_batch(data, block_size, batch_size, gen, shift=True):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + block_size] for i in ix])
    if shift:
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # THE shift
    else:
        y = x.clone()                                # the break: label == input
    return x, y


def train(model, data, steps, lr=3e-3, batch_size=32, shift=True, log_every=150):
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        x, y = get_batch(data, model.block_size, batch_size, gen, shift)
        logits = model(x)
        # cross-entropy = -log p(correct next token), averaged over every position
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % log_every == 0:
            print(f"  step {step:5d}  loss {loss.item():.3f}")
    return loss.item()


@torch.no_grad()
def sample(model, prompt, n_new, temperature, seed=0):
    """Same loop as generate(), in torch. Also returns the mean entropy (nats)
    of the distribution each token was drawn from."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.tensor([encode(prompt)])
    entropies = []
    for _ in range(n_new):
        logits = model(idx[:, -model.block_size:])[:, -1, :]   # crop, last row
        if temperature == 0:
            probs = F.softmax(logits, dim=-1)
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)
        entropies.append(-(probs * torch.log(probs + 1e-12)).sum().item())
        idx = torch.cat([idx, nxt], dim=1)
    return decode(idx[0].tolist()), float(np.mean(entropies))


@torch.no_grad()
def loss_per_position(model, data, block_size, batch_size=64):
    """Cross-entropy at each of the T positions, averaged over a fixed batch."""
    gen = torch.Generator().manual_seed(123)
    x, y = get_batch(data, block_size, batch_size, gen)
    loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), reduction="none")
    return loss.view(batch_size, block_size).mean(0)


def show(text, indent="  ", width=72):
    """Print a sample in fixed-width chunks so nothing is wider than a page.
    Newlines are made visible as \\n. Concatenate the chunks to get the text."""
    flat = text.replace("\n", "\\n")
    for i in range(0, len(flat), width):
        print(indent + flat[i:i + width])


def find_loop(s):
    """Smallest period p such that the tail of s is the same p chars, 3 times."""
    for p in range(1, len(s) // 3):
        if s[-3 * p:-2 * p] == s[-2 * p:-p] == s[-p:]:
            return p
    return None


# ===========================================================================
# DEMOS
# ===========================================================================
BLOCK, D_MODEL, STEPS = 32, 64, 600
DATA = torch.tensor(encode(TEXT))
PROMPT = "the model "


def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_1_the_loop():
    """Random weights. Nothing learned. Just the shapes and the loop."""
    line("DEMO 1: the generation loop -- a while loop that eats its own output")

    p = init_params(VOCAB, block_size=8, d_model=16, n_heads=2)
    prefix = encode("the ")
    print(f"vocab = {VOCAB} characters, block_size = 8, d_model = 16, 2 heads")
    print(f"prefix {'the '!r} -> tokens {prefix}")

    logits, weights = gpt_forward(np.array(prefix), p)
    print(f"\ntok_emb[tokens] + pos_emb[:4] : {(p['tok_emb'][prefix] + p['pos_emb'][:4]).shape}")
    print(f"attention weights, per head   : {weights[0].shape} x {len(weights)} heads")
    print(f"logits                        : {logits.shape}  (one row PER position)")
    print(f"we keep only logits[-1]       : {logits[-1].shape}  (the next-token guess)")
    print("\nhead 0 weights (row = position asking, col = position answering):")
    print(weights[0])
    print("\nthe loop, 5 steps, greedy:")
    out = generate(p, prefix, 5, trace=True)
    print(f"result: {decode(out)!r}")

    # The causal mask has a second gift: earlier rows never change as the
    # sequence grows, so a real server caches them (the KV cache).
    l4, _ = gpt_forward(np.array(out[:4]), p)
    l8, _ = gpt_forward(np.array(out[:8]), p)
    print("\ncausal: logits for position 0 identical at ctx=4 and ctx=8?",
          np.allclose(l4[0], l8[0]))
    l4, _ = gpt_forward(np.array(out[:4]), p, causal=False)
    l8, _ = gpt_forward(np.array(out[:8]), p, causal=False)
    print("no mask: logits for position 0 identical at ctx=4 and ctx=8?",
          np.allclose(l4[0], l8[0]))
    print("\nREAD THIS: random weights, so the letters are noise. What matters is")
    print("the shape story: T tokens in, (T, vocab) out, keep the last row, argmax,")
    print("append, go again. That loop IS the product. The mask also freezes every")
    print("earlier row, which is why serving can cache them instead of recomputing.")


def demo_2_the_shift():
    """Inputs are tokens[:-1], targets are tokens[1:]. Break it: targets = inputs."""
    line("DEMO 2: the target shift -- the label is just the next character")

    toks = encode("autocomplete")
    inputs, targets = toks[:-1], toks[1:]
    print("text     :", "autocomplete")
    print("inputs   :", decode(inputs), "  <- tokens[:-1]")
    print("targets  :", " " + decode(targets), "  <- tokens[1:], shifted one left")
    print("pairs    :", " ".join(f"{decode([i])}->{decode([t])}" for i, t in zip(inputs, targets)))
    print("\nno human labelled anything. The text IS the label. That is why you")
    print("can train on the whole internet.\n")

    for shift, label in ((True, "honest  (targets = tokens[1:])"),
                         (False, "BROKEN  (targets = inputs)  ")):
        torch.manual_seed(0)
        model = TinyGPT(VOCAB, BLOCK, d=D_MODEL)
        print(f"{label}, 200 steps:")
        train(model, DATA, steps=200, log_every=50, shift=shift)
    print("\nREAD THIS: with the shift removed the loss falls to ~0 in a couple")
    print("hundred steps. Nothing about language was learned -- the model found")
    print("that output[i] = input[i] and stopped. Loss near zero is not success,")
    print("it is a smell. Check the shift first.")


def demo_3_train_and_sample():
    """Train the tiny GPT properly, then let it talk."""
    line(f"DEMO 3: train it ({STEPS} steps, CPU), then sample")

    torch.manual_seed(0)
    model = TinyGPT(VOCAB, BLOCK, d=D_MODEL)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"corpus: {len(TEXT):,} chars, 400 sentences, vocab {VOCAB}. first line:")
    show(TEXT.split("\n")[0])
    print(f"model: 2 layers, d_model={D_MODEL}, 4 heads, block_size={BLOCK}")
    print(f"  {n_params:,} params ({n_params / 175e9 * 100:.7f}% of GPT-3's 175B)")
    t0 = time.time()
    train(model, DATA, steps=STEPS)
    print(f"  a uniform guess scores ln({VOCAB}) = {math.log(VOCAB):.3f}; the corpus's own")
    print(f"  randomness puts the floor near {LOSS_FLOOR:.3f} (4 choices of 6 per sentence)")
    print(f"  trained in {time.time() - t0:.1f}s", file=sys.stderr)   # timing off stdout

    greedy, _ = sample(model, PROMPT, 100, temperature=0.0)
    warm, _ = sample(model, PROMPT, 100, temperature=0.8)
    print(f"\nprompt: {PROMPT!r}")
    print("\ngreedy (temperature 0), 100 chars:")
    show(greedy)
    print("\ntemperature 0.8, 100 chars:")
    show(warm)
    print("\nREAD THIS: the loss fell from a uniform guess toward the floor set by")
    print("my dice, and the same loop that printed noise in demo 1 now prints")
    print("sentences with the corpus's grammar. Nothing changed in the loop. Only")
    print("the weights moved. It learned the structure, not the meaning; there is")
    print("no meaning in the corpus to learn.")
    return model


def demo_4_temperature(model):
    """Divide the logits before softmax. Low = confident, high = random."""
    line("DEMO 4: temperature -- one number between the logits and the dice")

    print(f"prompt: {PROMPT!r}; entropy is of the distribution each char was drawn")
    print(f"from, mean over 100 draws, in nats. Max possible = ln({VOCAB}) = {math.log(VOCAB):.3f}\n")
    for temp in (0.2, 0.8, 2.0):
        text, ent = sample(model, PROMPT, 100, temperature=temp)
        print(f"temperature {temp}:  mean entropy {ent:.3f} nats")
        show(text[len(PROMPT):])
        print()

    long_greedy, _ = sample(model, PROMPT, 400, temperature=0.0)
    period = find_loop(long_greedy)
    print("temperature 0 for 400 chars -- does it loop?")
    if period:
        print(f"  yes: the tail repeats every {period} chars: {long_greedy[-period:]!r}")
        print("  last 120 chars:")
        show(long_greedy[-120:])
    else:
        print(f"  no cycle of length < {400 // 3} found. last 120 chars:")
        show(long_greedy[-120:])
    print("\nREAD THIS: temperature does not touch the model. It rescales the last")
    print("row of logits before softmax. 0.2 sharpens to near-argmax, 2.0 flattens")
    print("toward uniform. Greedy forever is deterministic, so once it revisits a")
    print("state it is stuck in a cycle. That is the 'repetition' bug, in one line.")


def demo_5_context_window(model):
    """The two ways the loop goes wrong at the edge of the context window."""
    line("DEMO 5: the context window -- the loop has no memory past block_size")

    p = init_params(VOCAB, block_size=8, d_model=16, n_heads=2)
    too_long = encode("the loop ")          # 9 tokens, block_size is 8
    print(f"numpy: {len(too_long)} tokens into a block_size={p['block_size']} model, no crop:")
    try:
        gpt_forward(np.array(too_long), p)
    except AssertionError as e:
        print(f"  AssertionError: {e}")
    print(f"torch: {BLOCK + 1} tokens into the block_size={BLOCK} trained model, no crop:")
    try:
        model(torch.tensor([encode(TEXT[:BLOCK + 1])]))
    except IndexError as e:
        print(f"  IndexError: {e}   (pos_emb has only {BLOCK} rows)")

    print("\ncrop from the wrong end -- idx[:, :block_size] instead of idx[:, -block_size:]:")
    idx = torch.tensor([encode(PROMPT)])
    with torch.no_grad():
        for _ in range(60):
            logits = model(idx[:, :model.block_size])[:, -1, :]      # WRONG END
            idx = torch.cat([idx, logits.argmax(dim=-1, keepdim=True)], dim=1)
    wrong = decode(idx[0].tolist())
    show(wrong)
    print("  every token after the window filled is the same one?",
          len(set(wrong[BLOCK:])) == 1)
    right, _ = sample(model, PROMPT, 60, temperature=0.0)
    print("  vs the right crop, same model, same prompt:")
    show(right)
    print("\nREAD THIS: the model sees at most block_size tokens, ever. GPT-3: 2048.")
    print("Forget to crop and the crash is the GOOD outcome. Crop from the head")
    print("and it re-reads the same stale prefix forever, never sees what it just")
    print("wrote, and prints the same character until you kill it.")


def demo_6_no_mask(causal_model):
    """Train the same model with the causal mask removed. Watch it cheat."""
    line("DEMO 6: remove the causal mask -- lower loss, worse text")

    torch.manual_seed(0)
    cheat = TinyGPT(VOCAB, BLOCK, d=D_MODEL, causal=False)
    print("same model, same data, same steps, mask removed:")
    train(cheat, DATA, steps=STEPS)

    pp_c = loss_per_position(causal_model, DATA, BLOCK)
    pp_n = loss_per_position(cheat, DATA, BLOCK)
    print(f"\nloss per position: mean over positions 0..{BLOCK - 2}, then position {BLOCK - 1},")
    print("the one row that has no next column to peek at:")
    print(f"  causal : {pp_c[:-1].mean():.3f}   last position: {pp_c[-1]:.3f}")
    print(f"  no mask: {pp_n[:-1].mean():.3f}   last position: {pp_n[-1]:.3f}")

    good, _ = sample(causal_model, PROMPT, 100, temperature=0.8)
    bad, _ = sample(cheat, PROMPT, 100, temperature=0.8)
    print(f"\nshort prompt {PROMPT!r}, temperature 0.8:")
    print("  causal :")
    show(good, indent="    ")
    print("  no mask:")
    show(bad, indent="    ")

    full = TEXT[:BLOCK]                     # a full window: last row is the honest one
    good, _ = sample(causal_model, full, 100, temperature=0.8)
    bad, _ = sample(cheat, full, 100, temperature=0.8)
    print(f"\nfull-window prompt {full!r}, continuation only:")
    print("  causal :")
    show(good[BLOCK:], indent="    ")
    print("  no mask:")
    show(bad[BLOCK:], indent="    ")
    print("\nREAD THIS: without the mask, position i can attend to position i+1,")
    print("which IS its target. So it copies, and the loss collapses at every row")
    print("but the last -- the only row with nothing to copy. At sampling time the")
    print("last row is the only row that ever matters, so 31 of 32 rows of")
    print("training were wasted on a trick that never fires in production. Short")
    print("prompt: the last row is a cheat row, output is one letter forever. Full")
    print("window: the honest row limps along on 1/32 of the gradient. Train/serve")
    print("skew, in one flag. Bidirectional attention is BERT, and BERT does not")
    print("generate.")


def demo_7_prompt_as_data(model):
    """Few-shot is a longer input string. The weights do not move."""
    line("DEMO 7: in-context learning -- the examples are in the input, not the weights")

    ids = encode(FEWSHOT)
    print(FEWSHOT)
    print(f"\nthat prompt is {len(ids)} tokens; the model sees one integer array:")
    print(f"  {ids[:12]} ... {ids[-8:]}")
    with torch.no_grad():
        logits = model(torch.tensor([ids[-BLOCK:]]))
    print(f"logits shape: {tuple(logits.shape)}   gradient updates: 0")
    print("\nREAD THIS: nothing in the code above knows what an 'example' is. The")
    print("three worked pairs are just more prefix. GPT-3's claim is that at 175B")
    print("parameters the next-token guess after that prefix is 'fromage'. My")
    print("model has 100k parameters and a six-word vocabulary of nouns; it will")
    print("not do this, and I am not going to pretend it does. The mechanism is")
    print("identical. The capability is what scale bought.")


if __name__ == "__main__":
    t_start = time.time()
    demo_1_the_loop()
    demo_2_the_shift()
    model = demo_3_train_and_sample()
    demo_4_temperature(model)
    demo_5_context_window(model)
    demo_6_no_mask(model)
    demo_7_prompt_as_data(model)

    line("THE THREE PAPERS, COMPRESSED")
    print("""
  the model   = tok_emb + pos_emb -> N x decoder_block -> layer_norm -> logits
  the block   = causal multi-head attention -> skip -> MLP -> skip
  the loss    = -log p(tokens[1:] | tokens[:-1]), every position, all at once
  the product = while True: append argmax(logits[-1]); crop to block_size

  GPT-1 (2018): 12 blocks, 117M params, BooksCorpus. Pretrain on the loss
                above, then fine-tune per task. Idea: pretraining transfers.
  GPT-2 (2019): 48 blocks, 1.5B, WebText. No fine-tuning at all. Idea: at
                scale, the prompt alone selects the task (zero-shot).
  GPT-3 (2020): 96 blocks, 175B, d_model 12288. Idea: put examples in the
                prompt and the frozen model uses them (few-shot, in-context).

  Same loop. Same loss. Same block. Three headlines.
""")
    print(f"total runtime {time.time() - t_start:.1f}s", file=sys.stderr)
