"""
Parameter-Efficient Fine-Tuning -- the family, for programmers, not researchers.

Run it:      python3 peft_from_scratch.py
Debug it:    set a breakpoint in block() and watch which hook fires where.

No torch. No autograd. No training. One frozen transformer block in NumPy,
four small trainable things you can bolt onto it (adapter, prefix, prompt,
(IA)^3), and the numbers that separate them: how many parameters each one
adds, whether step 0 is a no-op, and what each costs at inference.
Every function is <20 lines.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

D, D_FF, N_TOK = 64, 256, 16     # model width, FFN width, tokens in the input


# ---------------------------------------------------------------------------
# STAGE 0 -- the frozen base. One transformer block, one head, as a dict of
# weights. This is the framework you do NOT fork. Nothing in here trains.
# ---------------------------------------------------------------------------
def make_block(d=D, d_ff=D_FF, seed=0):
    rng = np.random.default_rng(seed)
    s = 1 / np.sqrt(d)
    return {
        "W_q": rng.normal(0, s, (d, d)), "W_k": rng.normal(0, s, (d, d)),
        "W_v": rng.normal(0, s, (d, d)), "W_o": rng.normal(0, s, (d, d)),
        "ln1_g": np.ones(d), "ln1_b": np.zeros(d),
        "W_1": rng.normal(0, s, (d, d_ff)), "b_1": np.zeros(d_ff),
        "W_2": rng.normal(0, 1 / np.sqrt(d_ff), (d_ff, d)), "b_2": np.zeros(d),
        "ln2_g": np.ones(d), "ln2_b": np.zeros(d),
    }


def n_params(tree):
    """Count every number in a nested dict of arrays. Works on the block AND
    on a hook dict, so 'trainable' is just n_params(hooks)."""
    if isinstance(tree, np.ndarray):
        return tree.size
    return sum(n_params(v) for v in tree.values())


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))   # overflow safety
    return e / e.sum(axis=axis, keepdims=True)


def relu(x):
    return np.maximum(x, 0)


def layernorm(x, g, b, eps=1e-5):
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


# ---------------------------------------------------------------------------
# STAGE 1 -- THE ANCHOR PAPER. Houlsby 2019, the adapter module:
#
#     Adapter(h) = h + W_up . ReLU(W_down . h)         h: (n, d)
#
# Down-project d -> m (m << d), nonlinearity, up-project m -> d, add back.
# The `h +` is a skip link: the adapter proposes a DIFF, not a replacement.
# With W_up = 0 the diff is 0 and the block is untouched. That is the init.
# ---------------------------------------------------------------------------
def adapter(h, A):
    z = relu(h @ A["W_down"] + A["b_down"])          # (n, d) -> (n, m)
    return h + z @ A["W_up"] + A["b_up"]             # (n, m) -> (n, d), + skip


# ---------------------------------------------------------------------------
# STAGE 2 -- the block, with every hook point marked. This is the whole page.
#
# hooks is a dict. Which keys are present tells you which method you are
# running. Nothing else changes. PEFT is choosing WHERE to register the hook:
#
#   prefix_k, prefix_v    prefix tuning  -- extra K/V rows attention can see
#   ia3_k, ia3_v, ia3_ff  (IA)^3         -- rescale K, V and the FFN hidden
#   adapter_attn,         Houlsby        -- bottleneck after each sublayer
#   adapter_ffn
#
# Prompt tuning has no hook in here: it prepends rows to X before the call.
# LoRA has no hook in here either: it adds a low-rank diff to W_q/W_v. See
# the LoRA page for that one.
# ---------------------------------------------------------------------------
def block(W, X, hooks=None, return_weights=False):
    h = hooks or {}
    Q, K, V = X @ W["W_q"], X @ W["W_k"], X @ W["W_v"]
    if "ia3_k" in h:
        K, V = K * h["ia3_k"], V * h["ia3_v"]          # (IA)^3: scale K, V
    if "prefix_k" in h:                                # prefix: p extra rows
        K = np.concatenate([h["prefix_k"], K])         #   the queries can
        V = np.concatenate([h["prefix_v"], V])         #   attend to
    weights = softmax(Q @ K.T / np.sqrt(Q.shape[-1]))  # (n, n + p)
    a = (weights @ V) @ W["W_o"]
    if "adapter_attn" in h:
        a = adapter(a, h["adapter_attn"])              # Houlsby hook 1
    X = layernorm(X + a, W["ln1_g"], W["ln1_b"])
    f = relu(X @ W["W_1"] + W["b_1"])
    if "ia3_ff" in h:
        f = f * h["ia3_ff"]                            # (IA)^3: scale hidden
    f = f @ W["W_2"] + W["b_2"]
    if "adapter_ffn" in h:
        f = adapter(f, h["adapter_ffn"])               # Houlsby hook 2
    X = layernorm(X + f, W["ln2_g"], W["ln2_b"])
    return (X, weights) if return_weights else X


# ---------------------------------------------------------------------------
# STAGE 3 -- the four methods, each as "build me the small trainable thing".
# In a framework these are the ONLY tensors with requires_grad=True.
# ---------------------------------------------------------------------------
def make_adapters(m, d=D, zero_up=True, seed=1):
    """Houlsby: two bottleneck adapters per block, width m. Up-proj zeroed so
    step 0 is the identity (the paper: 'near-identity initialisation')."""
    rng = np.random.default_rng(seed)
    def one():
        up = np.zeros((m, d)) if zero_up else rng.normal(0, 0.1, (m, d))
        return {"W_down": rng.normal(0, 0.1, (d, m)), "b_down": np.zeros(m),
                "W_up": up, "b_up": np.zeros(d)}
    return {"adapter_attn": one(), "adapter_ffn": one()}


def make_prefix(p, d=D, zero=False, seed=2):
    """Li & Liang: p learned key rows and p learned value rows, per layer.
    They never pass through W_k / W_v -- they ARE the keys and values."""
    rng = np.random.default_rng(seed)
    if zero:
        return {"prefix_k": np.zeros((p, d)), "prefix_v": np.zeros((p, d))}
    return {"prefix_k": rng.normal(0, 1, (p, d)),
            "prefix_v": rng.normal(0, 1, (p, d))}


def make_prompt(p, X, d=D, seed=3):
    """Lester: p learned INPUT tokens. Not a hook in the block -- new rows
    prepended to X. Init by sampling real token vectors (the paper's best)."""
    rng = np.random.default_rng(seed)
    return X[rng.integers(0, len(X), size=p)].copy()


def make_ia3(d=D, d_ff=D_FF):
    """Liu 2022: three scaling vectors, all ones at init -> identity."""
    return {"ia3_k": np.ones(d), "ia3_v": np.ones(d), "ia3_ff": np.ones(d_ff)}


def merge_ia3(W, h):
    """Fold the (IA)^3 vectors into the frozen weights. After this the hooks
    are gone and inference costs exactly what the base model costs."""
    M = dict(W)
    M["W_k"] = W["W_k"] * h["ia3_k"][None, :]   # scale the columns of W_k
    M["W_v"] = W["W_v"] * h["ia3_v"][None, :]
    M["W_2"] = W["W_2"] * h["ia3_ff"][:, None]  # scale the rows of W_2
    return M


# ---------------------------------------------------------------------------
# STAGE 4 -- inference cost. Count matmul FLOPs (2 per multiply-add) for one
# forward pass of n tokens. This is the column interviewers actually ask for.
# ---------------------------------------------------------------------------
def flops(n, d=D, d_ff=D_FF, p_prefix=0, m_adapter=0):
    proj = 4 * (2 * n * d * d)                    # Q, K, V, O projections
    attn = 2 * (2 * n * (n + p_prefix) * d)       # scores + blend, n+p keys
    ffn = 2 * (2 * n * d * d_ff)                  # up + down
    adap = 2 * 2 * (2 * n * d * m_adapter)        # two adapters, down + up
    return proj + attn + ffn + adap


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def pct(k, total):
    return f"{100 * k / total:6.2f}%"


def demo_1_frozen_block():
    line("DEMO 1: the frozen base -- one block, how many numbers")
    W = make_block()
    attn = sum(W[k].size for k in ("W_q", "W_k", "W_v", "W_o"))
    ffn = sum(W[k].size for k in ("W_1", "b_1", "W_2", "b_2"))
    ln = sum(W[k].size for k in ("ln1_g", "ln1_b", "ln2_g", "ln2_b"))
    print(f"d_model = {D}, d_ff = {D_FF}, one head")
    print(f"  attention (W_q, W_k, W_v, W_o):  {attn:>7,}")
    print(f"  feed-forward (W_1, b_1, W_2, b_2): {ffn:>7,}")
    print(f"  two layer norms:                  {ln:>7,}")
    print(f"  TOTAL frozen params:              {n_params(W):>7,}")
    X = np.random.default_rng(10).normal(0, 1, (N_TOK, D))
    print(f"\ninput X: {X.shape}  ->  block(W, X): {block(W, X).shape}")
    print("\nREAD THIS: in a real model this block repeats 12, 24, 96 times and")
    print("the widths are 768 to 12,288. Same dict, bigger arrays. Full")
    print("fine-tuning moves every one of these numbers, per task, and you")
    print("store a whole copy per task. PEFT asks: what is the smallest thing")
    print("I can bolt on and train instead?")
    return W, X


def demo_2_five_hooks(W):
    line("DEMO 2: five places to hook -- trainable params, as % of the block")
    total = n_params(W)
    L = 12   # a BERT-base-sized stack, to show what scales per layer
    rows = [
        ("full fine-tuning",        total,                   total),
        ("adapters, m=8 (x2)",      n_params(make_adapters(8)),  None),
        ("prefix tuning, p=8",      n_params(make_prefix(8)),    None),
        ("prompt tuning, p=8",      8 * D,                   8 * D),
        ("(IA)^3",                  n_params(make_ia3()),        None),
        ("LoRA r=8 on W_q, W_v",    2 * (2 * 8 * D),         None),
    ]
    print(f"{'method':<24}{'per block':>10}{'% block':>9}{'x12 blocks':>12}")
    for name, k, stack in rows:
        stack = k * L if stack is None else stack
        print(f"{name:<24}{k:>10,}{pct(k, total):>9}{stack:>12,}")
    print("\nREAD THIS: every method except prompt tuning pays PER LAYER.")
    print("Prompt tuning pays once, at the input, however deep the model is.")
    print("That is why its count stays flat in the last column.")

    print("\nadapter bottleneck sweep (two adapters per block):")
    for m in (2, 4, 8, 16, 64):
        k = n_params(make_adapters(m))
        note = "  <- m = d: no bottleneck" if m == D else ""
        print(f"  m = {m:>2}:  {k:>7,} params  {pct(k, total)}{note}")
    attn = sum(W[k].size for k in ("W_q", "W_k", "W_v", "W_o"))
    big = n_params(make_adapters(D))
    print(f"\n  at m = d the two adapters ({big:,}) outweigh")
    print(f"  the whole attention sublayer they were bolted onto ({attn:,}).")
    print("  The point of the method was the bottleneck. Remove it, no method.")

    print("\nforget requires_grad=False on the base (the classic mistake):")
    hooks = make_adapters(8)
    k, both = n_params(hooks), n_params(hooks) + n_params(W)
    print(f"  adapters only:      {k:>7,}  {pct(k, total)}")
    print(f"  adapters + base:    {both:>7,}  {pct(both, total)}  <- everything")


def demo_3_zero_init(W, X):
    line("DEMO 3: step 0 must be a no-op -- which methods start as identity?")
    base = block(W, X)

    def drift(hooks, X_in=X):
        out = block(W, X_in, hooks)
        return np.abs(out[-N_TOK:] - base).max()

    print("max |block_with_hook(X) - block(X)| before any training:")
    print(f"  adapters, W_up = 0:           {drift(make_adapters(8)):.6f}")
    rand = make_adapters(8, zero_up=False)
    print(f"  adapters, W_up random:        {drift(rand):.6f}  <- not a no-op")
    print(f"  (IA)^3, vectors = 1:          {drift(make_ia3()):.6f}")
    print(f"  prefix, K and V = 0:          {drift(make_prefix(8, zero=True)):.6f}"
          "  <- not a no-op")
    prompt = make_prompt(8, X)
    print(f"  prompt, 8 sampled tokens:     "
          f"{drift({}, np.concatenate([prompt, X])):.6f}  <- not a no-op")
    print("\nREAD THIS: a zeroed adapter and an all-ones (IA)^3 leave the block")
    print("byte-identical, so training starts FROM the pretrained model. A")
    print("zeroed prefix does not: a zero key still scores exp(0) = 1 in the")
    print("softmax, so it steals attention mass from real tokens on step 0.")
    print("Same for prompt tokens -- they are seen. LoRA with B = 0 is a no-op")
    print("for the same reason the adapter is: the diff is exactly zero.")


def demo_4_prefix_keys_vs_values(W, X):
    line("DEMO 4: prefix tuning -- keys decide, values deliver")
    p = 4
    hooks = make_prefix(p)
    _, w = block(W, X, hooks, return_weights=True)
    print(f"attention weights shape with p={p} prefix rows: {w.shape}  (n, p+n)")
    print("weight each query puts on the 4 prefix slots (first 3 queries):")
    print(w[:3, :p])
    print("mass on prefix per query (first 3):", w[:3, :p].sum(-1))

    print("\nBREAK IT: train V only, leave the prefix keys at zero")
    v_only = {"prefix_k": np.zeros((p, D)), "prefix_v": hooks["prefix_v"]}
    _, w0 = block(W, X, v_only, return_weights=True)
    print(w0[:3, :p])
    redraw = {"prefix_k": np.zeros((p, D)),
              "prefix_v": np.random.default_rng(77).normal(0, 1, (p, D))}
    _, w1 = block(W, X, redraw, return_weights=True)
    print("change every value row, max |weights diff|:", np.abs(w1 - w0).max())
    print("\nREAD THIS: with learned keys, each query reads the prefix slots")
    print("DIFFERENTLY (top block). Zero the keys and every slot scores the")
    print("same for every query: the four rows are identical and swapping the")
    print("values moves the weights by exactly 0. The prefix has collapsed")
    print("into one fixed vector added to everyone. Keys are the addressing.")


def demo_5_prompt_tuning(W, X):
    line("DEMO 5: prompt tuning -- the hook is in the input")
    p = 8
    prompt = make_prompt(p, X)
    Xp = np.concatenate([prompt, X])
    out, w = block(W, Xp, return_weights=True)
    mass = w[p:, :p].sum(-1)          # real queries, columns = prompt keys
    print(f"input: {p} prompt rows + {N_TOK} real rows = {Xp.shape}")
    print(f"attention mass real tokens put on the prompt:")
    print(f"  mean {mass.mean():.3f}   min {mass.min():.3f}   max {mass.max():.3f}"
          f"   (uniform would be {p / (p + N_TOK):.3f})")
    zero = np.concatenate([np.zeros((p, D)), X])
    _, wz = block(W, zero, return_weights=True)
    mz = wz[p:, :p].sum(-1)
    print(f"  same, prompt = zeros:  mean {mz.mean():.3f}")
    print("\nBREAK IT: read the output at row 0")
    print(f"  out.shape = {out.shape}   <- {p} of those rows are the prompt")
    print(f"  out[0]  belongs to prompt[0]      out[{p}] belongs to real token 0")
    print(f"  the answer is out[{p}:], shape {out[p:].shape}")
    print("\nREAD THIS: prompt tokens are ordinary positions. Attention sees")
    print("them, gives them weight, and their output rows come back with the")
    print("rest. Training moves the prompt so real tokens read something")
    print("useful there. Lester et al. found this only matches full fine-")
    print("tuning once the model is around 10B params; at this size, and at")
    print("BERT size, it lags. Not demonstrated here -- that needs training.")


def demo_6_inference_cost(W, X):
    line("DEMO 6: what each hook costs at inference (n=16 tokens, one block)")
    base = flops(N_TOK)
    print(f"base block FLOPs: {base:,}\n")
    rows = [
        ("adapters, m=8",     flops(N_TOK, m_adapter=8) - base, 0, "yes"),
        ("prefix, p=8",       flops(N_TOK, p_prefix=8) - base,  8, "no"),
        ("prompt, p=8",       flops(N_TOK + 8) - base,          8, "no"),
        ("(IA)^3, unmerged",  N_TOK * (2 * D + D_FF),           0, "no"),
        ("(IA)^3, merged",    0,                                0, "no"),
        ("LoRA, merged",      0,                                0, "no"),
    ]
    hdr = ("method", "extra FLOPs", "extra tokens", "adds a step?")
    print(f"{hdr[0]:<20}{hdr[1]:>12}{hdr[2]:>14}{hdr[3]:>14}")
    for name, f, t, seq in rows:
        print(f"{name:<20}{f:>12,}{t:>14}{seq:>14}")

    h = make_ia3()
    rng = np.random.default_rng(5)
    h["ia3_k"], h["ia3_v"] = rng.normal(1, 0.3, D), rng.normal(1, 0.3, D)
    h["ia3_ff"] = rng.normal(1, 0.3, D_FF)
    hooked, merged = block(W, X, h), block(merge_ia3(W, h), X)
    print(f"\n(IA)^3 merge check, max |hooked - merged|: "
          f"{np.abs(hooked - merged).max():.2e}   identical? "
          f"{np.allclose(hooked, merged)}")
    print("\nREAD THIS: the adapter's extra FLOPs are small, but they sit IN the")
    print("residual path: the block cannot finish until the adapter does.")
    print("That is latency, and it was Houlsby's cost. Prefix and prompt")
    print("tuning widen the attention instead and grow the KV cache by p per")
    print("layer. (IA)^3 and LoRA fold into the frozen weights: zero extra.")


if __name__ == "__main__":
    W, X = demo_1_frozen_block()
    demo_2_five_hooks(W)
    demo_3_zero_init(W, X)
    demo_4_prefix_keys_vs_values(W, X)
    demo_5_prompt_tuning(W, X)
    demo_6_inference_cost(W, X)

    line("THE WHOLE FAMILY, COMPRESSED")
    print("""
  Freeze the base. Register one small hook. Train only the hook.

  method         where the hook goes                 step 0    inference
  adapter        after each sublayer, in the path    no-op     + latency
  prefix         extra K/V rows per layer            changes   + p tokens/layer
  prompt         extra input rows, once              changes   + p tokens
  (IA)^3         scale K, V, FFN hidden              no-op     0 (merge)
  LoRA           low-rank diff on W_q, W_v           no-op     0 (merge)

  The rest of each paper is which task, which scale, which init worked.
  Real, but not the idea. The idea is the hook table.
""")
