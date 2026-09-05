"""
Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
Layer (Shazeer et al., 2017) -- for programmers, not researchers.
With the one simplification from Switch Transformers (Fedus et al., 2021)
that every modern MoE LLM inherits: top-1 routing and a capacity factor.

Run it:      python3 moe_from_scratch.py
Debug it:    set a breakpoint in router() and watch one token pick experts.

No torch. No autograd. No training. The forward pass, the routing, the FLOP
count and the load-balancing losses as printed numbers. Every function is
under 20 lines.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- helpers. softmax turns a row of scores into percentages that sum
# to 1; the `- max` is overflow safety, same as in the attention script.
# softplus is a smooth relu, log(1 + e^x). The paper uses it to keep the
# noise scale positive.
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def softplus(x):
    return np.logaddexp(0, x)


# ---------------------------------------------------------------------------
# STAGE 1 -- one expert. A plain 2-layer MLP, d -> h -> d, relu in between.
# It is exactly the transformer's feed-forward block. Nothing is special
# about an expert: it is one backend service behind the load balancer.
# ---------------------------------------------------------------------------
def make_expert(d, h, rng):
    """Returns (W1, W2). Real models learn these; random here."""
    W1 = rng.normal(0, 1 / np.sqrt(d), (d, h))
    W2 = rng.normal(0, 1 / np.sqrt(h), (h, d))
    return W1, W2


def expert_forward(x, W1, W2):
    return np.maximum(0, x @ W1) @ W2        # (n, d) -> (n, h) -> (n, d)


# ---------------------------------------------------------------------------
# STAGE 2 -- THE ROUTER. Section 2.1 of the paper, "Noisy Top-K Gating":
#
#     H(x)_i = (x . W_g)_i + StandardNormal() * Softplus((x . W_noise)_i)
#     G(x)   = Softmax(KeepTopK(H(x), k))
#
# A load balancer: score every expert, keep the best k, give each of them a
# share that sums to 1. Every other expert gets exactly 0 -- and "exactly 0"
# is what makes the layer sparse. No call, no compute.
# ---------------------------------------------------------------------------
def router(x, W_g, k, W_noise=None, rng=None, bias=None, renormalise=True):
    # (1) SCORE: one logit per expert.  (n, d) @ (d, N) -> (n, N)
    logits = x @ W_g
    if bias is not None:                     # only used to fake a hot shard
        logits = logits + bias

    # (2) NOISE (training only): jitter the scores so the same k experts do
    #     not win every time. Learned scale, kept positive by softplus.
    if W_noise is not None:
        scale = softplus(x @ W_noise)
        logits = logits + rng.standard_normal(logits.shape) * scale

    # (3) PERCENTAGES over all N experts. The balance loss reads these.
    probs = softmax(logits)

    # (4) TOP-K: sort descending, keep the first k expert ids per token.
    top = np.argsort(-logits, axis=-1)[:, :k]            # (n, k), best first
    gates = np.zeros_like(probs)
    np.put_along_axis(gates, top, np.take_along_axis(probs, top, -1), -1)

    # (5) RENORMALISE the kept gates so they sum to 1 again. Identical to the
    #     paper's softmax-after-KeepTopK; this order is easier to debug.
    if renormalise:
        gates = gates / gates.sum(axis=-1, keepdims=True)
    return gates, top, probs


# ---------------------------------------------------------------------------
# STAGE 3 -- THE MoE LAYER. Equation 1 of the paper:
#
#     y = sum_i  G(x)_i * E_i(x)
#
# Only experts with G(x)_i > 0 are called. Loop over EXPERTS, not tokens:
# gather the tokens routed to expert i, run them as one batch, scatter the
# results back. That gather/scatter is the "dispatch" and "combine" the
# paper's section 3 is about.
#
# capacity (Switch, the method section): the most tokens one expert may take
# per batch. Overflow tokens are dropped -- they get 0 from this layer and
# ride the residual connection through unchanged.
# ---------------------------------------------------------------------------
def moe_forward(x, gates, experts, capacity=None):
    y = np.zeros_like(x)
    dropped = 0
    for i, (W1, W2) in enumerate(experts):
        sent = np.flatnonzero(gates[:, i] > 0)          # tokens routed to i
        if capacity is not None and len(sent) > capacity:
            dropped += len(sent) - capacity
            sent = sent[:capacity]                      # first come, first served
        if len(sent) == 0:
            continue                                    # idle expert: no FLOPs
        y[sent] += gates[sent, i][:, None] * expert_forward(x[sent], W1, W2)
    return y, dropped


# ---------------------------------------------------------------------------
# STAGE 4 -- WHY BOTHER. Parameters vs FLOPs.
# A dense FFN is 2 matrices of d*h. A MoE with N experts is N times that plus
# a tiny router. But a token only visits k experts, so its FLOPs are k times
# one FFN, not N. Decoupling "how much the model knows" from "how much work
# per token" is the entire reason the paper exists.
# ---------------------------------------------------------------------------
def ffn_params(d, h):
    return 2 * d * h


def ffn_flops_per_token(d, h):
    return 2 * ffn_params(d, h)             # each weight: one multiply, one add


def moe_params(d, h, N):
    return N * ffn_params(d, h) + d * N     # N experts + the router matrix


def moe_flops_per_token(d, h, N, k):
    return k * ffn_flops_per_token(d, h) + 2 * d * N


# ---------------------------------------------------------------------------
# STAGE 5 -- THE HOT-SHARD ALARM. Auxiliary losses, added to the real loss so
# gradient descent is punished for routing everything to one expert.
#
# Shazeer, section 4: importance_i = sum over the batch of gate_i.
#     L = CV(importance)^2, CV = std / mean. Zero when perfectly even.
#     (The paper's second loss, "load", does the same to a smooth estimate
#     of token COUNTS and needs the normal CDF. Skipped here.)
# Switch, the method section:  L = N * sum_i f_i * P_i
#     f_i = fraction of tokens dispatched to expert i
#     P_i = mean router probability for expert i
#     Both uniform (1/N each) gives N * N * 1/N^2 = 1, the minimum for top-1.
#     Everything on one expert gives f = 1, P ~ 1: the loss climbs to N.
# ---------------------------------------------------------------------------
def importance_loss(gates):
    importance = gates.sum(axis=0)                   # (N,) load in gate units
    cv = importance.std() / importance.mean()
    return cv ** 2


def switch_balance_loss(gates, probs):
    N = gates.shape[1]
    f = (gates > 0).mean(axis=0)                     # share of tokens per expert
    P = probs.mean(axis=0)                           # mean probability per expert
    return N * np.sum(f * P)


# ===========================================================================
# DEMOS
# ===========================================================================
D, H, N = 8, 16, 8       # toy sizes: d_model, expert hidden width, experts


def make_layer(d, h, n_experts, seed=0):
    """Random router + N random experts. Real models learn all of this."""
    rng = np.random.default_rng(seed)
    W_g = rng.normal(0, 0.5, (d, n_experts))         # router weights
    W_noise = rng.normal(0, 0.5, (d, n_experts))     # noise-scale weights
    experts = [make_expert(d, h, rng) for _ in range(n_experts)]
    return W_g, W_noise, experts


def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_1_router():
    line("DEMO 1: the router -- 16 tokens, 8 experts, top-2")
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (16, D))
    W_g, _, _ = make_layer(D, H, N)
    gates, top, probs = router(x, W_g, k=2)

    p = probs[0]
    print("token 0, router probabilities over all 8 experts:")
    print("   ", p, " sum", f"{p.sum():.3f}")
    print(f"  top-2 = experts {top[0].tolist()}, probs {p[top[0]]},"
          f" sum {p[top[0]].sum():.3f}")
    print(f"  renormalised gates {gates[0][top[0]]}, sum {gates[0].sum():.3f}")

    print("\nrouting table:")
    print("  token   1st   gate     2nd   gate")
    for t in range(16):
        a, b = top[t]
        print(f"  {t:5d}   E{a}    {gates[t, a]:.3f}    E{b}    {gates[t, b]:.3f}")
    counts = (gates > 0).sum(axis=0)
    print("\ntokens per expert: " +
          "  ".join(f"E{i}:{c}" for i, c in enumerate(counts)))
    print(f"total assignments: {counts.sum()} = 16 tokens x k=2")
    print("\nREAD THIS: every token got exactly two experts and the two gates")
    print("sum to 1. The other six experts got a gate of exactly 0 for that")
    print("token. Zero gate means the expert is never called. That is the")
    print("whole trick. The router is a load balancer whose rules are learned.")


def demo_2_layer_output():
    line("DEMO 2: the layer output -- gate x expert(x), summed over the top-2")
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (16, D))
    W_g, _, experts = make_layer(D, H, N)
    gates, top, _ = router(x, W_g, k=2)
    y, _ = moe_forward(x, gates, experts)

    dense = expert_forward(x, *experts[0])            # one FFN on its own
    print("moe output shape: ", y.shape)
    print("dense FFN shape:  ", dense.shape, " same shape: a drop-in for the FFN")

    a, b = top[0]
    by_hand = (gates[0, a] * expert_forward(x[:1], *experts[a])
               + gates[0, b] * expert_forward(x[:1], *experts[b]))
    print(f"\ntoken 0 by hand: {gates[0, a]:.3f} * E{a}(x) + "
          f"{gates[0, b]:.3f} * E{b}(x)")
    print("  equals moe output row 0?", np.allclose(by_hand[0], y[0]))
    print("  experts that ran for token 0: 2 of 8. The other 6 did nothing.")

    print("\nbreak it: renormalise=False. Experts are 8 identical copies here")
    print("(sparse-upcycling style init), so only the gates can move |y|.")
    same = [experts[0]] * N
    print("  k   gate sum   mean |y| renormalised   mean |y| not renormalised")
    for k in (1, 2, 4, 8):
        g_ok, _, _ = router(x, W_g, k)
        g_no, _, _ = router(x, W_g, k, renormalise=False)
        y_ok, _ = moe_forward(x, g_ok, same)
        y_no, _ = moe_forward(x, g_no, same)
        print(f"  {k}   {g_no.sum(1).mean():.3f}      "
              f"{np.linalg.norm(y_ok, axis=1).mean():.3f}"
              f"                   {np.linalg.norm(y_no, axis=1).mean():.3f}")
    print("\nREAD THIS: with renormalisation the output scale does not depend on")
    print("k. Without it, the layer silently multiplies its output by 'however")
    print("much probability the top-k happened to cover'. Change k after")
    print("training and every downstream layer sees a different scale.")


def demo_3_params_vs_flops():
    line("DEMO 3: parameters vs FLOPs -- the whole reason MoE exists")
    d, h, n_experts, k = 1024, 4096, 8, 2
    print(f"d_model={d}, ffn hidden={h}, experts={n_experts}, top-{k}\n")
    dp, mp = ffn_params(d, h), moe_params(d, h, n_experts)
    df, mf = ffn_flops_per_token(d, h), moe_flops_per_token(d, h, n_experts, k)
    print(f"dense FFN params:             {dp:>13,}")
    print(f"MoE params (8 experts):       {mp:>13,}   {mp / dp:.2f}x")
    print(f"dense FFN FLOPs per token:    {df:>13,}")
    print(f"MoE FLOPs per token (top-2):  {mf:>13,}   {mf / df:.2f}x")

    print("\nFLOPs per token as k grows (params stay at 8.00x):")
    for kk in (1, 2, 4, 8):
        f = moe_flops_per_token(d, h, n_experts, kk)
        note = ""
        if kk == n_experts:
            note = "   <- k = N: a dense layer, 8x the compute"
        print(f"  top-{kk}: {f:>13,}   {f / df:.2f}x{note}")

    print("\nMixtral 8x7B config, expert weights only (SwiGLU: 3 matrices each):")
    d, h, n_experts, k, layers = 4096, 14336, 8, 2, 32
    per_expert = 3 * d * h
    total = per_expert * n_experts * layers
    active = per_expert * k * layers
    print(f"  one expert:                {per_expert:>15,}")
    print(f"  all experts, all layers:   {total:>15,}   ~{total / 1e9:.1f}B")
    print(f"  active per token (top-2):  {active:>15,}   ~{active / 1e9:.1f}B")
    print("\nREAD THIS: 8x the parameters for 2x the per-token compute. The")
    print("router costs 2*d*N FLOPs, a rounding error next to one expert. The")
    print("price is memory: every expert must be loaded, whether or not it")
    print("runs. You pay for 8 in VRAM and use 2.")


def demo_4_load_balance():
    line("DEMO 4: hot shards -- 64 tokens, top-1 (Switch), and the balance loss")
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (64, D))
    W_g, _, experts = make_layer(D, H, N)

    def report(label, gates, probs):
        counts = (gates > 0).sum(axis=0)
        print(f"\n{label}")
        for i, c in enumerate(counts):
            print(f"  E{i} {'#' * c:<64} {c}")
        print("  tokens per expert: " +
              "  ".join(f"E{i}:{c}" for i, c in enumerate(counts)))
        print(f"  importance loss (Shazeer, CV^2):     "
              f"{importance_loss(gates):.3f}")
        print(f"  balance loss (Switch, N * sum f*P):  "
              f"{switch_balance_loss(gates, probs):.3f}")
        return counts

    g_ok, _, p_ok = router(x, W_g, k=1)
    report("random-init router (nobody touched it):", g_ok, p_ok)

    bias = np.zeros(N)
    bias[3] = 8.0                                     # a router that drifted
    g_hot, _, p_hot = router(x, W_g, k=1, bias=bias)
    counts = report("collapsed router (logit bias +8 on E3, nothing to stop it):",
                    g_hot, p_hot)

    # No tokens -> no gradient. Prove it without autograd: change E5's
    # weights and check whether the output notices.
    y_before, _ = moe_forward(x, g_hot, experts)
    W1, W2 = experts[5]
    perturbed = list(experts)
    perturbed[5] = (W1 + 1.0, W2)
    y_after, _ = moe_forward(x, g_hot, perturbed)
    print(f"\nE5 received {counts[5]} tokens. Add 1.0 to every weight in E5:")
    print("  output changed?", not np.allclose(y_before, y_after),
          " -> dLoss/dW for E5 is exactly 0. E5 never trains.")
    print("\nREAD THIS: a hot expert gets all the gradient, gets better, and")
    print("the router sends it more. The idle experts get no tokens, no")
    print("gradient, no improvement, and stay idle. The rich get richer. The")
    print("aux loss is the only thing in the objective pushing the other way.")


def demo_5_capacity():
    line("DEMO 5: capacity factor -- Switch's per-expert rate limit")
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (64, D))
    W_g, _, experts = make_layer(D, H, N)
    bias = np.zeros(N)
    bias[3] = 8.0
    print("64 tokens, 8 experts, top-1.  capacity = floor(factor * 64 / 8)\n")
    for label, b in (("random-init router:", None), ("collapsed router:", bias)):
        gates, _, _ = router(x, W_g, k=1, bias=b)
        for factor in (1.0, 1.25, 2.0):
            cap = int(factor * 64 / N)
            _, dropped = moe_forward(x, gates, experts, capacity=cap)
            print(f"  {label:<19} factor {factor:<5} capacity {cap:>2}"
                  f"  ->  dropped {dropped:>2} of 64")
        print()
    print("READ THIS: a dropped token gets y = 0 from this layer; the residual")
    print("connection carries x through unchanged. Nothing raises. Nothing")
    print("logs. A higher factor drops fewer tokens but pads every expert's")
    print("batch to the cap, so you pay compute for empty slots.")


def demo_6_noise():
    line("DEMO 6: noisy gating -- exploration in training, a bug at inference")
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (16, D))
    W_g, W_noise, experts = make_layer(D, H, N)

    def run(noise, seed):
        r = np.random.default_rng(seed)
        gates, top, _ = router(x, W_g, k=2,
                               W_noise=W_noise if noise else None, rng=r)
        y, _ = moe_forward(x, gates, experts)
        return y, np.sort(top, axis=1)

    y1, t1 = run(False, 10)
    y2, t2 = run(False, 11)
    print("noise off, two runs: outputs identical?", np.allclose(y1, y2))
    y1, t1 = run(True, 10)
    y2, t2 = run(True, 11)
    changed = int((t1 != t2).any(axis=1).sum())
    print("noise on,  two runs: outputs identical?", np.allclose(y1, y2))
    print(f"  max |y1 - y2| = {np.abs(y1 - y2).max():.3f};"
          f" {changed} of 16 tokens changed at least one expert")
    print("\nREAD THIS: in training the noise is deliberate. It stops the same")
    print("k experts winning every time, so the others get tokens and learn.")
    print("Leave it on at inference and the same prompt routes differently on")
    print("every call. Turn it off, like dropout, and the layer is a pure")
    print("function again.")


if __name__ == "__main__":
    demo_1_router()
    demo_2_layer_output()
    demo_3_params_vs_flops()
    demo_4_load_balance()
    demo_5_capacity()
    demo_6_noise()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. expert    = a plain FFN. N of them, same shape, different weights
  2. router    = softmax(x @ W_g): one score per expert, per token
  3. top-k     = keep the k best, renormalise, zero the rest. Sparse.
  4. output    = sum of gate_i * expert_i(x) over the k kept   (eq. 1)
  5. the deal  = params x N, FLOPs per token x k. That is the paper.
  6. noise     = jitter the router logits in training only  (section 2.1)
  7. aux loss  = punish uneven routing, or one expert eats everything
  8. Switch    = k=1, one balance loss, a capacity cap, drop the overflow

  Everything else is plumbing: dispatch/combine, all-to-all across devices,
  hierarchical routing for thousands of experts. Real, but not the idea.
""")
