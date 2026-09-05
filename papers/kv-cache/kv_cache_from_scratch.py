"""
KV caching and the generation loop -- for programmers, not researchers.

Run it:      python3 kv_cache_from_scratch.py
Debug it:    breakpoint in decode_step_cached and watch cache["K"] grow one row.

No torch. No training. Random weights, because this page is about COMPUTE and
MEMORY, not about learning. Every core function is short.

The claim: the causal mask makes token i's key and value immutable, so you can
memoise them. That single fact turns an O(n^3) generation loop into O(n^2).
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- a toy decoder. Random weights. Shapes are what matter here.
# ---------------------------------------------------------------------------
class Config:
    def __init__(self, vocab=64, d_model=32, n_heads=4, n_layers=2,
                 max_ctx=128):
        self.vocab = vocab
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_layers = n_layers
        self.max_ctx = max_ctx
        self.d_ff = 4 * d_model


def build_model(cfg, seed=0):
    """Random weights. We never train; we only measure the forward pass."""
    rng = np.random.default_rng(seed)
    n = lambda *s: rng.normal(0, 1.0 / np.sqrt(s[0]), s)
    return {
        "emb": n(cfg.vocab, cfg.d_model),
        "pos": positional_encoding(cfg.max_ctx, cfg.d_model),
        "layers": [{
            "Wq": n(cfg.d_model, cfg.d_model),
            "Wk": n(cfg.d_model, cfg.d_model),
            "Wv": n(cfg.d_model, cfg.d_model),
            "Wo": n(cfg.d_model, cfg.d_model),
            "W1": n(cfg.d_model, cfg.d_ff),
            "W2": n(cfg.d_ff, cfg.d_model),
        } for _ in range(cfg.n_layers)],
        "head": n(cfg.d_model, cfg.vocab),
    }


def positional_encoding(n_positions, d_model):
    """Same sinusoids as the transformer paper. Fixed table, max_ctx rows."""
    pos = np.arange(n_positions)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((n_positions, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def layernorm(x, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    return (x - m) / np.sqrt(x.var(-1, keepdims=True) + eps)


def split_heads(x, cfg):
    """(n, d_model) -> (n_heads, n, head_dim)"""
    return x.reshape(x.shape[0], cfg.n_heads, cfg.head_dim).transpose(1, 0, 2)


def merge_heads(x):
    """(n_heads, n, head_dim) -> (n, d_model)"""
    return x.transpose(1, 0, 2).reshape(x.shape[1], -1)


# ---------------------------------------------------------------------------
# STAGE 1 -- THE NAIVE LOOP. Every step, re-embed and re-project the WHOLE
# prefix, build the full (n, n) score matrix, mask it, and throw away every
# row except the last. This is what "generate the next token" means if you
# have never heard of a cache. Cost per step grows linearly with n.
# ---------------------------------------------------------------------------
def forward_full(model, cfg, ids, use_mask=True):
    """Run the whole prefix through every layer. Returns (n, vocab) logits."""
    n = len(ids)
    x = model["emb"][ids] + model["pos"][:n]
    mask = np.tril(np.ones((n, n), dtype=bool)) if use_mask else None
    for L in model["layers"]:
        h = layernorm(x)
        Q = split_heads(h @ L["Wq"], cfg)      # recomputed every step
        K = split_heads(h @ L["Wk"], cfg)      # recomputed every step -- waste
        V = split_heads(h @ L["Wv"], cfg)      # recomputed every step -- waste
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(cfg.head_dim)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        x = x + merge_heads(softmax(scores) @ V) @ L["Wo"]
        h = layernorm(x)
        x = x + np.maximum(h @ L["W1"], 0) @ L["W2"]
    return layernorm(x) @ model["head"]


# ---------------------------------------------------------------------------
# STAGE 2 -- THE CACHED LOOP. The causal mask says row i of the score matrix
# only reads columns <= i. So K[i] and V[i] can never be influenced by a token
# that arrives later: they are pure functions of the prefix up to i. Pure
# function + immutable inputs = safe to memoise. That is the whole trick.
#
# One new token in, one new row appended per layer, one (1, n) score row out.
# ---------------------------------------------------------------------------
def new_cache(cfg):
    """Per layer: K and V, each (n_heads, seq_so_far, head_dim). Never Q."""
    return [{"K": np.zeros((cfg.n_heads, 0, cfg.head_dim)),
             "V": np.zeros((cfg.n_heads, 0, cfg.head_dim))}
            for _ in range(cfg.n_layers)]


def decode_step_cached(model, cfg, token_id, pos, cache, append=True):
    """One token in, one row of logits out. `pos` indexes the position table."""
    if pos >= cfg.max_ctx:
        raise IndexError(f"position {pos} past context window {cfg.max_ctx}")
    x = model["emb"][[token_id]] + model["pos"][[pos]]     # (1, d_model)
    for L, c in zip(model["layers"], cache):
        h = layernorm(x)
        q = split_heads(h @ L["Wq"], cfg)      # (H, 1, hd) -- NOT cached
        k = split_heads(h @ L["Wk"], cfg)      # (H, 1, hd) -- appended
        v = split_heads(h @ L["Wv"], cfg)      # (H, 1, hd) -- appended
        if append:
            c["K"] = np.concatenate([c["K"], k], axis=1)
            c["V"] = np.concatenate([c["V"], v], axis=1)
        # No mask needed: the cache only ever holds the past. The mask has
        # become a data-structure invariant instead of a matrix of -inf.
        scores = q @ c["K"].transpose(0, 2, 1) / np.sqrt(cfg.head_dim)
        x = x + merge_heads(softmax(scores) @ c["V"]) @ L["Wo"]
        h = layernorm(x)
        x = x + np.maximum(h @ L["W1"], 0) @ L["W2"]
    return (layernorm(x) @ model["head"])[0]               # (vocab,)


def generate_naive(model, cfg, prompt, n_new):
    """Recompute everything, every step. Correct, and needlessly expensive."""
    ids = list(prompt)
    for _ in range(n_new):
        logits = forward_full(model, cfg, ids)[-1]
        ids.append(int(np.argmax(logits)))
    return ids


def generate_cached(model, cfg, prompt, n_new, cache=None, append=True):
    """Prefill the prompt, then one row per step. Same tokens, less work.

    append=False is the deliberate bug: the prompt is cached, but nothing the
    model generates is ever appended, so it re-reads the prompt forever.
    """
    cache = new_cache(cfg) if cache is None else cache
    ids = list(prompt)
    pos = cache[0]["K"].shape[1]
    for t in ids:                       # prefill (token-by-token here)
        logits = decode_step_cached(model, cfg, t, pos, cache, True)
        pos += 1
    for _ in range(n_new):              # decode
        nxt = int(np.argmax(logits))
        ids.append(nxt)
        logits = decode_step_cached(model, cfg, nxt, pos, cache, append)
        pos += 1
    return ids


# ---------------------------------------------------------------------------
# STAGE 3 -- the arithmetic. MACs = multiply-accumulates.
#
# Per token, per layer:  4*d*d  (Q,K,V,O projections)
#                      + 2*d*n  (scores, then the weighted sum over n cached)
#                      + 2*d*ff (the feed-forward, unchanged by any cache)
# The cache only ever touches the middle term. Say that out loud in interviews.
# ---------------------------------------------------------------------------
def macs_per_token(cfg, n_ctx):
    d, ff = cfg.d_model, cfg.d_ff
    return cfg.n_layers * (4 * d * d + 2 * d * n_ctx + 2 * d * ff)


def macs_cached_step(cfg, n_cached):
    """One new token attending over n_cached keys. Flat in the projections."""
    return macs_per_token(cfg, n_cached)


def macs_naive_step(cfg, n_ctx):
    """Recompute all n_ctx tokens from scratch. Linear in n_ctx, per step."""
    return n_ctx * macs_per_token(cfg, n_ctx)


# ---------------------------------------------------------------------------
# STAGE 4 -- cache memory. The formula that decides what you can serve:
#
#   bytes = 2 (K and V) * layers * kv_heads * head_dim * seq_len * dtype_bytes
#
# n_kv_heads == n_heads      -> multi-head attention (the 2017 paper)
# n_kv_heads == 1            -> multi-query attention (Shazeer 2019)
# 1 < n_kv_heads < n_heads   -> grouped-query attention (Ainslie 2023)
# Q always keeps all n_heads. Only K and V are shared, because only K and V
# are stored.
# ---------------------------------------------------------------------------
def kv_cache_bytes(layers, kv_heads, head_dim, seq_len, dtype_bytes=2,
                   batch=1):
    return 2 * layers * kv_heads * head_dim * seq_len * dtype_bytes * batch


GIB = 1024 ** 3


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


CFG = Config()
MODEL = build_model(CFG)
PROMPT = [3, 17, 42, 8]


def demo_1_the_two_loops():
    line("DEMO 1: the same loop, twice -- O(n^3) and O(n^2)")

    n = 64
    naive = sum(macs_naive_step(CFG, t) for t in range(1, n + 1))
    cached = sum(macs_cached_step(CFG, t) for t in range(1, n + 1))
    print(f"toy model: {CFG.n_layers} layers, {CFG.n_heads} heads, "
          f"d_model={CFG.d_model}, seq={n}\n")
    print(f"  naive  loop total MACs: {naive:>12,}")
    print(f"  cached loop total MACs: {cached:>12,}")
    print(f"  ratio                 : {naive / cached:>12.1f}x\n")

    print("  per-step cost (MACs), naive grows, cached is flat:")
    print("    step |        naive |      cached")
    for t in (1, 8, 16, 32, 48, 64):
        print(f"    {t:>4} | {macs_naive_step(CFG, t):>12,} |"
              f" {macs_cached_step(CFG, t):>11,}")

    print("\nREAD THIS: the naive step redoes the whole prefix, so its cost is")
    print("linear in n and the whole sequence costs ~n^3. The cached step does")
    print("one token, so the sequence costs ~n^2. The cached column is not")
    print("perfectly flat -- it creeps, because attention still scans n keys.")
    print("The projections and the feed-forward are what went flat.")


def demo_2_identical_outputs():
    line("DEMO 2: the cache is not an approximation")

    n_new = 12
    a = generate_naive(MODEL, CFG, PROMPT, n_new)
    b = generate_cached(MODEL, CFG, PROMPT, n_new)
    print("  naive  tokens:", a)
    print("  cached tokens:", b)
    print("  same token sequence?", a == b)

    ids = a[:-1]
    la = forward_full(MODEL, CFG, ids)[-1]
    cache = new_cache(CFG)
    for i, t in enumerate(ids):
        lb = decode_step_cached(MODEL, CFG, t, i, cache)
    print(f"  max |naive logits - cached logits| = {np.abs(la - lb).max():.3e}")
    print("\nREAD THIS: float noise, nothing else. This is the correctness")
    print("claim, and the causal mask is what buys it.")


def demo_3_the_mask_is_the_proof():
    line("DEMO 3: why the mask is the proof -- row i never moves")

    ids = PROMPT
    for use_mask, label in ((True, "WITH causal mask"), (False, "NO mask")):
        w4 = attention_weights(MODEL, CFG, ids, use_mask)
        w5 = attention_weights(MODEL, CFG, ids + [21], use_mask)
        print(f"\n  {label} -- layer 0, head 0 attention weights")
        print("    before appending a 5th token (4x4):")
        for r in w4:
            print("     ", r)
        print("    after  appending a 5th token, first 4 rows (4x5):")
        for r in w5[:4]:
            print("     ", r)
        drift = np.abs(w5[:4, :4] - w4).max()
        print(f"    max change in the first 4 rows: {drift:.3e}")

    print("\n  now the consequence for a cache:")
    ids2 = PROMPT + [21]
    ref = forward_full(MODEL, CFG, ids2, use_mask=False)[-1]
    cache = new_cache(CFG)
    for i, t in enumerate(ids2):
        stale = decode_step_cached(MODEL, CFG, t, i, cache)
    print(f"    unmasked model, cached vs recomputed logits, max abs diff:"
          f" {np.abs(ref - stale).max():.4f}")
    print(f"    argmax token: recomputed={int(np.argmax(ref))}  "
          f"cached={int(np.argmax(stale))}")

    print("\nREAD THIS: with the mask, appending a token changes NOTHING in the")
    print("earlier rows -- 0.0 drift -- so cached K and V stay valid forever.")
    print("Without the mask, every earlier row is renormalised over the new")
    print("column, the cache is stale, and the logits diverge. The mask is not")
    print("a performance trick. It is the proof that the function is pure.")


def attention_weights(model, cfg, ids, use_mask=True, layer=0, head=0):
    """Layer-0 attention weights for the given prefix. For demo 3 only."""
    n = len(ids)
    x = model["emb"][ids] + model["pos"][:n]
    L = model["layers"][layer]
    h = layernorm(x)
    Q = split_heads(h @ L["Wq"], cfg)
    K = split_heads(h @ L["Wk"], cfg)
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(cfg.head_dim)
    if use_mask:
        s = np.where(np.tril(np.ones((n, n), dtype=bool)), s, -np.inf)
    return softmax(s)[head]


# --- the real-model tables ------------------------------------------------
LLAMA7B = dict(layers=32, heads=32, head_dim=128, dtype_bytes=2)
LLAMA7B_PARAMS = 6_738_415_616          # the published parameter count
WEIGHT_BYTES = LLAMA7B_PARAMS * 2       # fp16


def demo_4_cache_memory():
    line("DEMO 4: the number that decides what you can serve")

    print("  bytes = 2 * layers * kv_heads * head_dim * seq_len * dtype_bytes")
    print(f"\n  Llama-2-7B shape: 32 layers, 32 heads, head_dim 128, fp16")
    print(f"  weights (fp16, {LLAMA7B_PARAMS/1e9:.2f}B params): "
          f"{WEIGHT_BYTES / GIB:6.2f} GiB")
    per_tok = kv_cache_bytes(32, 32, 128, 1)
    print(f"  cache per token: {per_tok:,} bytes = {per_tok/1024:.0f} KiB\n")

    print("      seq_len |   batch 1 |  batch 32 | batch 32 vs weights")
    for s in (1024, 4096, 32768, 131072):
        b1 = kv_cache_bytes(32, 32, 128, s) / GIB
        b32 = kv_cache_bytes(32, 32, 128, s, batch=32) / GIB
        print(f"      {s:>7,} | {b1:>6.2f} GiB | {b32:>6.1f} GiB |"
              f" {b32 / (WEIGHT_BYTES/GIB):>6.1f}x the weights")

    print("\nREAD THIS: one 128k-token conversation costs 64 GiB of cache -- five")
    print("times the model. Batch 32 at 4k already outweighs the weights. The")
    print("cache, not the checkpoint, is what your serving box runs out of.")


def demo_5_mqa_and_gqa():
    line("DEMO 5: MQA and GQA -- shrink the cache, keep the heads")

    print("  Q always keeps 32 heads. Only K and V get shared.\n")
    print("      variant           kv_heads |  4k, b32 | 32k, b32 | reduction")
    for name, kvh in (("multi-head (2017)", 32),
                      ("GQA, 8 groups", 8),
                      ("MQA, 1 kv head", 1)):
        a = kv_cache_bytes(32, kvh, 128, 4096, batch=32) / GIB
        b = kv_cache_bytes(32, kvh, 128, 32768, batch=32) / GIB
        print(f"      {name:<17} {kvh:>8} | {a:>6.1f}GiB | {b:>6.1f}GiB |"
              f" {32/kvh:>6.0f}x")

    print("\n  same table as bytes per token, batch 1:")
    for name, kvh in (("multi-head", 32), ("GQA-8", 8), ("MQA", 1)):
        print(f"      {name:<11} {kv_cache_bytes(32, kvh, 128, 1):>8,} bytes"
              f"/token")

    print("\nREAD THIS: the reduction is exactly n_heads / n_kv_heads. MQA takes")
    print("32x off and costs quality; GQA-8 takes 4x off and, in the GQA paper's")
    print("ablations, sits close to multi-head. That is why Llama-2 70B and most")
    print("models since use GQA-8. It is a memory decision, not a maths one.")


def demo_6_prefill_vs_decode():
    line("DEMO 6: prefill is compute-bound, decode is memory-bound")

    p = 512
    cfg = Config(d_model=4096, n_heads=32, n_layers=32, max_ctx=1)
    prefill = sum(macs_per_token(cfg, t) for t in range(1, p + 1))
    decode = prefill                       # same MACs, spread over 512 steps
    step = macs_per_token(cfg, p)
    print(f"  Llama-2-7B-shaped, {p}-token prompt (approximate, attn + FFN):")
    print(f"    prefill, one batched pass : {prefill:>16,} MACs, "
          f"weights read 1x")
    print(f"    {p} sequential decode steps: {decode:>16,} MACs, "
          f"weights read {p}x")
    print(f"    one decode step           : {step:>16,} MACs\n")

    print("  arithmetic intensity (MACs per byte moved off memory):")
    cache_b = kv_cache_bytes(32, 32, 128, p)
    print(f"    prefill : {prefill / (WEIGHT_BYTES + cache_b):>8.1f} MACs/byte")
    print(f"    decode  : {step / (WEIGHT_BYTES + cache_b):>8.2f} MACs/byte")
    print("    a modern accelerator needs roughly 100-300 MACs/byte to stay")
    print("    compute-bound. Prefill clears it. Decode misses it by more")
    print("    than two orders of magnitude, and no extra FLOPs will help.")

    print("\nREAD THIS: identical total arithmetic, opposite bottlenecks. Prefill")
    print("does 512 tokens' work per trip through the weights. Decode does one,")
    print("so the GPU spends its life waiting on memory. That is why batching")
    print("32 requests costs almost nothing in latency and multiplies throughput")
    print("-- the weights were already in flight. It is also why per-token")
    print("latency barely improves when you buy more FLOPs.")


def demo_7_break_the_cache():
    line("DEMO 7: four ways to break a KV cache")

    good = generate_cached(MODEL, CFG, PROMPT, 8)
    print("  correct cached output          :", good)

    # (a) forget to append the new K and V
    bad = generate_cached(MODEL, CFG, PROMPT, 8, append=False)
    print("  forgot to append K,V           :", bad)
    print(f"    distinct tokens generated: correct={len(set(good[4:]))}, "
          f"broken={len(set(bad[4:]))} -- it froze on {bad[-1]}")

    # (b) reuse another sequence's cache -- the classic serving bug
    other = [11, 2, 55, 30]
    warm = new_cache(CFG)
    for i, t in enumerate(other):
        decode_step_cached(MODEL, CFG, t, i, warm)
    dirty = generate_cached(MODEL, CFG, PROMPT, 8, cache=warm)
    print("  reused sequence B's cache      :", dirty[len(other):])
    print("    (prompt was", PROMPT, "but it attends to", other, "too)")

    # (c) run past the context window with no eviction
    try:
        generate_cached(MODEL, CFG, list(range(60)), CFG.max_ctx)
    except IndexError as e:
        print("  no eviction, past max_ctx      : IndexError:", e)

    # (d) cache the queries too
    q_norms = []
    cache = new_cache(CFG)
    for i, t in enumerate(PROMPT):
        x = MODEL["emb"][[t]] + MODEL["pos"][[i]]
        L = MODEL["layers"][0]
        h = layernorm(x)
        q_norms.append(float(np.linalg.norm(h @ L["Wq"])))
        decode_step_cached(MODEL, CFG, t, i, cache)
    print(f"  ||q|| at each step             : "
          f"{[round(v, 3) for v in q_norms]}")
    print("    each step has exactly one query, used once, then dead. Nothing")
    print("    to reuse. K and V are read again at every future step; Q is not.")

    print("\nREAD THIS: (a) freezes the model on the prompt -- it never sees what")
    print("it just wrote. (b) is the bug that leaks one user's context into")
    print("another's reply. (c) is why every serving stack needs eviction or a")
    print("sliding window. (d) is the interview question: you cache K and V")
    print("because they are read n times; you never cache Q, because it is read")
    print("once.")


if __name__ == "__main__":
    demo_1_the_two_loops()
    demo_2_identical_outputs()
    demo_3_the_mask_is_the_proof()
    demo_4_cache_memory()
    demo_5_mqa_and_gqa()
    demo_6_prefill_vs_decode()
    demo_7_break_the_cache()

    line("THE WHOLE THING, COMPRESSED")
    print("""
  1. naive decode  = recompute the prefix every step. O(n^3) over a sequence.
  2. causal mask   = K[i], V[i] can never change. The function is pure.
  3. so memoise it = append one row per step. O(n^2), one row not a matrix.
  4. cache K and V = never Q. Q is used once, K and V are read n more times.
  5. the cost      = 2*layers*kv_heads*head_dim*seq*bytes. It beats the weights.
  6. MQA / GQA     = shrink kv_heads. GQA-8 is 4x off, near multi-head quality.
  7. PagedAttention= stop storing each cache contiguously. Pages, like virtual
                     memory, so fragmentation stops wasting your GPU RAM.
  8. prefill       = compute-bound. decode = memory-bound. Batch accordingly.

  The feed-forward cost per token is unchanged by any of this. Caching is an
  attention optimisation, and the O(n^2) figure is the attention part only.
""")
