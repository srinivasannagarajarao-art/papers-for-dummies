"""
DeepSeek-V3 -- for programmers, not researchers.

Run it:      python3 papers/deepseek-v3/deepseek-v3_from_scratch.py
Debug it:    breakpoint in bias_balance_step and watch the bias chase the load.

No torch. NumPy only. Nothing here is a benchmark: every "time" below is
bytes divided by a published bandwidth number, and every schedule is a
simulation of a schedule. The point is to make the constraint arithmetic
visible, the way you'd size a queue before writing the consumer.

All hardware figures are PUBLIC SPECIFICATIONS, not measurements.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- the machine, and the model, as constants.
#
# DeepSeek-V3 shapes are from the technical report (arXiv 2412.19437):
# 671B total parameters, 37B active per token, 61 layers, d_model 7168,
# 256 routed experts + 1 shared, top-8 routed, expert hidden 2048,
# node-limited routing to at most 4 nodes, 2048 H800 GPUs, 8 per node.
#
# The bandwidths are vendor specifications. H800 has the SAME arithmetic
# throughput as H100 and roughly HALF the NVLink. That one row is the paper.
# ---------------------------------------------------------------------------
D_MODEL = 7168
EXPERT_HIDDEN = 2048
TOPK_ROUTED = 8
N_ROUTED_EXPERTS = 256
GPUS_PER_NODE = 8
NODE_CAP_M = 4                 # node-limited routing: at most M nodes/token

BF16_PEAK_FLOPS = 989e12       # H100/H800 SXM, dense BF16, no sparsity
FP8_PEAK_FLOPS = 1979e12       # same chips, FP8 tensor cores -- twice the BF16
NVLINK_H100 = 900e9            # bytes/s, aggregate per GPU
NVLINK_H800 = 400e9            # bytes/s -- the export-control cut
IB_PER_GPU = 50e9              # 400 Gbps InfiniBand NIC = 50 GB/s


def moe_flops_per_token():
    """One token through top-8 SwiGLU experts. 3 matrices, 2 flops per MAC."""
    per_expert = 3 * 2 * D_MODEL * EXPERT_HIDDEN
    return TOPK_ROUTED * per_expert


def all_to_all_bytes_per_token(nodes_touched, dispatch_bytes, combine_bytes):
    """Bytes a single token pushes across each link type in one MoE layer.

    Dispatch: the hidden vector goes OUT to every node holding a chosen
    expert -- one InfiniBand copy per node, then NVLink fans it out to the
    GPUs inside that node. Combine: the expert outputs come back the same
    way. V3 dispatches in FP8 and combines in BF16, so the two directions
    are not the same size.
    """
    ib = nodes_touched * D_MODEL * (dispatch_bytes + combine_bytes)
    nvl = TOPK_ROUTED * D_MODEL * (dispatch_bytes + combine_bytes)
    return ib, nvl


def seconds(byte_count, bandwidth):
    """Bytes / bandwidth. This is the whole 'timing' model. No GPU was run."""
    return byte_count / bandwidth


# ---------------------------------------------------------------------------
# STAGE 1 -- routing, twice.
#
# The router is one matrix: scores = x @ W_g. Top-k wins. Same as the MoE
# page. The two ways of stopping one expert eating everything:
#
#   (a) auxiliary loss  -- add a penalty term to the LOSS. It has a gradient,
#       and that gradient pushes W_g away from the routing the model wanted.
#   (b) bias (V3)       -- add a per-expert bias to the score used for
#       SELECTION only. Nudge it up when the expert is starved, down when
#       it is hot. It never enters the loss, so its gradient is exactly zero.
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def route(X, W_g, bias=None, k=1):
    """Affinity scores, then top-k selection. Bias shifts SELECTION only."""
    scores = X @ W_g                       # (n_tokens, n_experts)
    probs = softmax(scores)                # what the model believes
    sel_scores = scores if bias is None else scores + bias
    chosen = np.argsort(-sel_scores, axis=1)[:, :k]
    return probs, chosen


def load_counts(chosen, n_experts):
    """Tokens per expert. The occupancy table an ops person would page on."""
    return np.bincount(chosen.ravel(), minlength=n_experts)


def aux_balance_loss_grad(probs, chosen, n_experts, alpha):
    """Switch-style auxiliary loss: alpha * N * sum_i f_i * P_i.

    f_i = fraction of tokens routed to expert i (no gradient, it's a count).
    P_i = mean router probability for expert i (this is where gradient flows).
    Returns the loss value and dL/dprobs, so you can see it is NOT zero.
    """
    n = chosen.shape[0]
    f = load_counts(chosen, n_experts) / n
    P = probs.mean(axis=0)
    loss = alpha * n_experts * float(np.sum(f * P))
    dprobs = alpha * n_experts * f[None, :] / n     # dL/dP_i spread over rows
    return loss, dprobs


def bias_balance_step(bias, chosen, n_experts, gamma):
    """V3's replacement, in three lines. Auxiliary-loss-free balancing.

    Starved expert -> bias up. Hot expert -> bias down. Fixed step size,
    sign only, like a thermostat. Never touches the loss, so it contributes
    NO gradient to W_g -- the router keeps learning what it wanted to learn.
    """
    load = load_counts(chosen, n_experts)
    err = load - load.mean()                 # + means overloaded
    return bias - gamma * np.sign(err)


def ce_loss_grad(probs, targets):
    """The gradient the model actually wants: pick the right expert.

    A stand-in for the real task gradient. Supervised here only so the
    'routing the model wanted' is something we can measure against.
    """
    n = probs.shape[0]
    loss = -np.mean(np.log(probs[np.arange(n), targets] + 1e-12))
    g = probs.copy()
    g[np.arange(n), targets] -= 1.0
    return loss, g / n


# ---------------------------------------------------------------------------
# STAGE 2 -- node-limited routing.
#
# A token wants its top-8 experts wherever they live. Left alone it can
# touch up to 8 different nodes, and every extra node is another InfiniBand
# copy of the same 7168-wide vector. The cap: score each NODE by the sum of
# its experts' affinities, keep the best M nodes, and pick the top-k experts
# only from those. Fewer nodes, fewer copies, smaller candidate pool.
# ---------------------------------------------------------------------------
def node_of_expert(n_experts, n_nodes):
    """Expert i lives on node i // (experts per node). Contiguous sharding."""
    per_node = n_experts // n_nodes
    return np.arange(n_experts) // per_node


def node_limited_topk(scores, n_nodes, k, m_cap):
    """Restrict candidates to the m_cap best nodes, then take top-k."""
    n_tokens, n_experts = scores.shape
    node_id = node_of_expert(n_experts, n_nodes)
    per_node = n_experts // n_nodes
    node_score = scores.reshape(n_tokens, n_nodes, per_node).sum(axis=2)
    keep = np.argsort(-node_score, axis=1)[:, :m_cap]      # (n_tokens, m_cap)
    allowed = np.zeros((n_tokens, n_nodes), dtype=bool)
    np.put_along_axis(allowed, keep, True, axis=1)
    masked = np.where(allowed[:, node_id], scores, -np.inf)
    return np.argsort(-masked, axis=1)[:, :k]


def nodes_touched(chosen, n_experts, n_nodes):
    """Distinct nodes per token. This is the InfiniBand copy count."""
    node_id = node_of_expert(n_experts, n_nodes)
    return np.array([len(set(node_id[row])) for row in chosen])


# ---------------------------------------------------------------------------
# STAGE 3 -- FP8, simulated in NumPy.
#
# e4m3: 1 sign, 4 exponent, 3 mantissa bits -> max 448, ~2 decimal digits.
# e5m2: 1 sign, 5 exponent, 2 mantissa bits -> max 57344, ~1 decimal digit.
# You trade range against precision, and there is not enough of either.
# The fix is not a better format, it is a smaller BLOCK sharing each scale.
# ---------------------------------------------------------------------------
FP8_FORMATS = {
    # name: (exponent bits, mantissa bits, exponent bias, max finite value)
    "e4m3": (4, 3, 7, 448.0),
    "e5m2": (5, 2, 15, 57344.0),
}


def fp8_cast(x, fmt):
    """Round-to-nearest into an FP8 grid. Overflow becomes inf, as it would.

    frexp splits x into mantissa in [0.5,1) and an exponent. Rounding the
    mantissa to `mbits` bits is exactly what the hardware does; anything
    past the format's max finite value has nowhere to go.
    """
    ebits, mbits, bias, maxval = FP8_FORMATS[fmt]
    m, e = np.frexp(x)
    m = np.round(m * 2 ** (mbits + 1)) / 2 ** (mbits + 1)
    y = np.ldexp(m, e)
    # Simplification: real FP8 has subnormals below min_normal; I flush them
    # to zero, so the "rounded to 0" counts below are a little pessimistic.
    min_normal = 2.0 ** (1 - bias)
    y = np.where(np.abs(y) < min_normal, 0.0, y)
    sign = np.where(y < 0, -1.0, 1.0)
    return np.where(np.abs(y) > maxval, sign * np.inf, y)


def fp8_blockwise(x, fmt, block):
    """Give every `block` values their own FP32 scale, then cast.

    This is V3's fine-grained quantisation: 1x128 tiles for activations,
    128x128 for weights. The scale is what stops a single outlier from
    deciding the resolution for its whole tensor.
    """
    _, _, _, maxval = FP8_FORMATS[fmt]
    flat = x.ravel().astype(np.float64)
    pad = (-flat.size) % block
    flat = np.concatenate([flat, np.zeros(pad)])
    tiles = flat.reshape(-1, block)
    scale = np.abs(tiles).max(axis=1, keepdims=True) / maxval
    scale = np.where(scale == 0, 1.0, scale)
    deq = fp8_cast(tiles / scale, fmt) * scale
    return deq.ravel()[: x.size].reshape(x.shape), scale.ravel()


def rel_error(a, b):
    """Relative L2 error. inf in, inf out -- which is the honest answer."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


# ---------------------------------------------------------------------------
# STAGE 4 -- the pipeline schedule, simulated.
#
# A 1F1B pipeline: P stages, m microbatches, each stage does compute then
# an all-to-all. Without overlap the device sits idle during the all-to-all.
# With overlap the all-to-all for microbatch i runs underneath the compute
# for microbatch i-1, so the device's cost per microbatch is max(c, a),
# not c + a. That is the idea DualPipe implements; this is only the idea.
# ---------------------------------------------------------------------------
def pipeline_bubble(n_stages, n_micro, compute, comm, overlap):
    """Simulate a 1F1B pipeline. Returns (makespan, busy fraction, bubble)."""
    step = max(compute, comm) if overlap else compute + comm
    busy_per_micro = compute        # only the maths counts as useful
    free = np.zeros(n_stages)      # when each stage is next free
    ready = np.zeros(n_micro)      # when each microbatch reaches this stage
    for s in range(n_stages):
        for i in range(n_micro):
            start = max(free[s], ready[i])
            free[s] = start + step
            ready[i] = free[s]
    makespan = float(free.max())
    busy = n_stages * n_micro * busy_per_micro
    return makespan, busy / (n_stages * makespan), 1 - busy / (n_stages * makespan)


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def demo_1_the_constraint():
    """Count the bytes. Divide by the spec sheet. That is the whole story."""
    line("DEMO 1: the constraint, in bytes -- one MoE layer, one GPU")

    tokens = 4096                       # tokens on this GPU, one micro-batch
    flops = tokens * moe_flops_per_token()
    t_compute = flops / BF16_PEAK_FLOPS

    ib_b, nvl_b = all_to_all_bytes_per_token(NODE_CAP_M, 1, 2)  # FP8 out, BF16 back
    ib_total, nvl_total = ib_b * tokens, nvl_b * tokens

    print(f"tokens on this GPU        : {tokens}")
    print(f"d_model                   : {D_MODEL}, top-{TOPK_ROUTED} routed"
          f" experts, capped at {NODE_CAP_M} nodes")
    print(f"expert FLOPs per token    : {moe_flops_per_token()/1e6:,.1f} MFLOP")
    print(f"expert FLOPs, this batch  : {flops/1e12:,.2f} TFLOP")
    print(f"InfiniBand bytes  (out+in): {ib_total/1e6:,.1f} MB "
          f"({NODE_CAP_M} node copies x (1B dispatch + 2B combine))")
    print(f"NVLink bytes      (out+in): {nvl_total/1e6:,.1f} MB "
          f"({TOPK_ROUTED} expert copies inside the nodes)")

    print("\narithmetic intensity of this layer:")
    print(f"  {flops/(ib_total+nvl_total):,.0f} FLOP per byte moved")
    print("machine balance (FLOP per byte the link can feed, FP8 compute):")
    for nm, bw in (("H100 NVLink 900 GB/s", NVLINK_H100),
                   ("H800 NVLink 400 GB/s", NVLINK_H800),
                   ("InfiniBand   50 GB/s", IB_PER_GPU)):
        print(f"  {nm}: {FP8_PEAK_FLOPS/bw:,.0f}")

    t_fp8 = flops / FP8_PEAK_FLOPS
    print("\nTHE SPINE. Intra-node leg only, the one export controls cut.")
    print("Compute is FP8, because V3's GEMMs are:")
    print("\n  machine          compute   NVLink leg   comm/compute   verdict")
    for nm, bw in (("H100  900 GB/s", NVLINK_H100), ("H800  400 GB/s", NVLINK_H800)):
        t_nvl = seconds(nvl_total, bw)
        r = t_nvl / t_fp8
        verdict = "COMPUTE-bound" if r < 1 else "COMM-bound"
        print(f"  {nm}  {t_fp8*1e3:6.2f}ms   {t_nvl*1e3:7.2f}ms   "
              f"{r:11.2f}x   {verdict}")
    print("\n  same layer, same arithmetic, one number on a spec sheet halved,")
    print("  and the bottleneck moves from the tensor cores to the wire.")

    print("\nthe cross-node leg, which the export cut did NOT touch:")
    for nm, bw in (("H100", NVLINK_H100), ("H800", NVLINK_H800)):
        t_nvl = seconds(nvl_total, bw)
        t_ib = seconds(ib_total, IB_PER_GPU)
        tot = (t_nvl + t_ib) / t_fp8
        print(f"  {nm}: IB {t_ib*1e3:5.2f}ms + NVLink {t_nvl*1e3:5.2f}ms "
              f"= {tot:.2f}x compute   ({t_ib/t_nvl:.1f}x the NVLink leg)")

    print("\nhalving NVLink alone, holding everything else fixed:")
    a = seconds(nvl_total, NVLINK_H100)
    b = seconds(nvl_total, NVLINK_H800)
    print(f"  NVLink leg  {a*1e3:.2f}ms -> {b*1e3:.2f}ms   ({b/a:.2f}x)")

    print("\nnow the same batch with NO node cap (worst case, 8 distinct nodes)")
    ib_b8, _ = all_to_all_bytes_per_token(TOPK_ROUTED, 1, 2)
    t_ib8 = seconds(ib_b8 * tokens, IB_PER_GPU)
    print(f"  InfiniBand bytes: {ib_b8*tokens/1e6:,.1f} MB, "
          f"{t_ib8*1e3:.2f}ms  ({t_ib8/seconds(ib_total, IB_PER_GPU):.2f}x)")

    print("\nand what BF16 dispatch instead of FP8 would cost:")
    ib16, nvl16 = all_to_all_bytes_per_token(NODE_CAP_M, 2, 2)
    print(f"  bytes: {(ib16+nvl16)*tokens/1e6:,.1f} MB vs "
          f"{(ib_b+nvl_b)*tokens/1e6:,.1f} MB   "
          f"({(ib16+nvl16)/(ib_b+nvl_b):.2f}x)")

    print("\nREAD THIS: bandwidths above are published specifications, and the")
    print("times are bytes/bandwidth, not measurements. The report itself says")
    print("it tuned the node cap so NVLink and IB traffic can be overlapped,")
    print("citing an effective NVLink:IB ratio of about 3.2:1 -- so treat the")
    print("900 and 400 as ceilings nobody reaches. The shape is what matters:")
    print("this layer moves so many bytes per FLOP that the links, not the")
    print("tensor cores, set the clock. Cut one link in half and you are")
    print("further into the region where the arithmetic units wait.")


def demo_2_balancing():
    """Auxiliary loss vs V3's bias. Both balance. Only one costs you routing."""
    line("DEMO 2: auxiliary-loss-free load balancing")

    rng = np.random.default_rng(0)
    n_experts, d, n_tokens, steps = 8, 16, 512, 60

    # Toy world: each token has a TRUE best expert. The types are Zipf-skewed,
    # so the routing the model wants is genuinely imbalanced. That is the
    # whole tension: balance and quality are not the same objective.
    p_type = 1.0 / np.arange(1, n_experts + 1)
    p_type /= p_type.sum()
    targets = rng.choice(n_experts, size=n_tokens, p=p_type)
    proto = rng.normal(0, 1, (n_experts, d))
    X = proto[targets] + rng.normal(0, 1.10, (n_tokens, d))

    print("true expert demand (Zipf, the routing the model actually wants):")
    print(" ", np.bincount(targets, minlength=n_experts))

    def train(mode, alpha=0.0, gamma=0.0, lr=0.5):
        W = rng2.normal(0, 0.3, (d, n_experts))
        bias = np.zeros(n_experts)
        history = []
        for t in range(steps):
            probs, chosen = route(X, W, bias if mode == "bias" else None, k=1)
            counts = load_counts(chosen, n_experts)
            history.append(counts)
            _, g_task = ce_loss_grad(probs, targets)
            g_bal = np.zeros_like(g_task)
            if mode == "aux":
                _, dP = aux_balance_loss_grad(probs, chosen, n_experts, alpha)
                # dL/dscores for a softmax: p * (dP - sum(p*dP))
                g_bal = probs * (dP - (probs * dP).sum(axis=1, keepdims=True))
            W -= lr * (X.T @ (g_task + g_bal))
            if mode == "bias":
                bias = bias_balance_step(bias, chosen, n_experts, gamma)
        probs, _ = route(X, W, None, k=1)
        clean = probs.argmax(axis=1)          # what the router believes, unbiased
        acc = float((clean == targets).mean())
        gbn = float(np.linalg.norm(X.T @ g_bal))
        return np.array(history), acc, gbn, bias

    rng2 = np.random.default_rng(7)
    h_aux, acc_aux, gn_aux, _ = train("aux", alpha=1.50)
    rng2 = np.random.default_rng(7)
    h_bias, acc_bias, gn_bias, bias_final = train("bias", gamma=0.6)
    rng2 = np.random.default_rng(7)
    h_none, acc_none, gn_none, _ = train("none")

    def spread(h):
        return h.max(axis=1) / np.maximum(h.min(axis=1), 1)

    print("\nmax/min expert occupancy, by step (1.0 = perfectly balanced):")
    print("  step      none      aux-loss      bias (V3)")
    for t in (0, 5, 10, 20, 40, 59):
        print(f"  {t:4d}   {spread(h_none)[t]:7.1f}   {spread(h_aux)[t]:9.1f}"
              f"   {spread(h_bias)[t]:12.1f}")

    print("\nfinal per-expert token counts (512 tokens, top-1):")
    print("  none    :", h_none[-1])
    print("  aux-loss:", h_aux[-1])
    print("  bias    :", h_bias[-1])
    print("  V3 bias vector:", np.round(bias_final, 2))

    print("\nnow the part that motivates the whole idea --")
    print("  gradient norm contributed by the balancing term:")
    print(f"    aux-loss : {gn_aux:.6f}   <- fights the routing the model wanted")
    print(f"    bias (V3): {gn_bias:.6f}   <- exactly zero, by construction")
    print("  router accuracy vs the true best expert (bias removed):")
    print(f"    no balancing : {acc_none:.4f}")
    print(f"    aux-loss     : {acc_aux:.4f}")
    print(f"    bias (V3)    : {acc_bias:.4f}")

    print("\nBREAK IT: crank the auxiliary loss weight to alpha=6.0")
    rng2 = np.random.default_rng(7)
    h_hi, acc_hi, gn_hi, _ = train("aux", alpha=6.0)
    print(f"  balance  max/min : {spread(h_hi)[-1]:.1f}  (aux 1.50: "
          f"{spread(h_aux)[-1]:.1f}, bias: {spread(h_bias)[-1]:.1f})")
    print(f"  router accuracy  : {acc_hi:.4f}  <- balanced, and worse")
    print(f"  balancing grad   : {gn_hi:.6f}")

    print("\nBREAK IT: bias update rate gamma=8.0 instead of 0.6")
    rng2 = np.random.default_rng(7)
    h_osc, acc_osc, _, bias_osc = train("bias", gamma=8.0)
    print("  max/min occupancy, last 8 steps:")
    print("   ", np.round(spread(h_osc)[-8:], 1))
    print("   vs gamma=0.6:", np.round(spread(h_bias)[-8:], 1))
    print(f"  final bias vector: {np.round(bias_osc, 1)}")

    print("\nREAD THIS: both mechanisms reach balance. The auxiliary loss buys")
    print("it with a gradient that is, by definition, not the task's gradient.")
    print("The bias buys it with a number that never appears in the loss at all")
    print("-- it moves the SELECTION, and the router keeps learning what it")
    print("wanted. Same alert, no side effect on the service behind it.")


def demo_3_node_limited_routing():
    """Cap the fan-out. Count the InfiniBand copies before and after."""
    line("DEMO 3: node-limited routing -- capping the fan-out")

    rng = np.random.default_rng(3)
    n_tokens, n_nodes = 4096, 32
    n_experts, k = N_ROUTED_EXPERTS, TOPK_ROUTED
    # Real routing is learned and correlated: a token has a home node it
    # mostly likes, plus scattered preferences elsewhere. Uniform random
    # scores would make the node cap look free, which it is not.
    home = rng.integers(0, n_nodes, n_tokens)
    node_id = node_of_expert(n_experts, n_nodes)
    scores = rng.normal(0, 1, (n_tokens, n_experts))
    scores += 1.2 * (node_id[None, :] == home[:, None])

    free = np.argsort(-scores, axis=1)[:, :k]
    capped = node_limited_topk(scores, n_nodes, k, NODE_CAP_M)

    nt_free = nodes_touched(free, n_experts, n_nodes)
    nt_cap = nodes_touched(capped, n_experts, n_nodes)

    print(f"{n_experts} experts over {n_nodes} nodes "
          f"({n_experts//n_nodes} experts per node), top-{k}, "
          f"{n_tokens} tokens")
    print(f"\n  uncapped: mean {nt_free.mean():.3f} nodes/token, "
          f"max {nt_free.max()}")
    print(f"  capped  : mean {nt_cap.mean():.3f} nodes/token, "
          f"max {nt_cap.max()}  (M={NODE_CAP_M})")

    def ib_bytes(nt):
        return float(nt.sum()) * D_MODEL * (1 + 2)

    b_free, b_cap = ib_bytes(nt_free), ib_bytes(nt_cap)
    print(f"\n  InfiniBand bytes, uncapped: {b_free/1e6:,.1f} MB  "
          f"({seconds(b_free, IB_PER_GPU)*1e3:.2f} ms at 50 GB/s)")
    print(f"  InfiniBand bytes, capped  : {b_cap/1e6:,.1f} MB  "
          f"({seconds(b_cap, IB_PER_GPU)*1e3:.2f} ms at 50 GB/s)")
    print(f"  saved: {(1-b_cap/b_free)*100:.1f}%")

    print("\nwhat the cap costs -- experts still reachable per token:")
    print(f"  uncapped : {n_experts} of {n_experts} (100.0%)")
    reach = NODE_CAP_M * (n_experts // n_nodes)
    print(f"  capped   : {reach} of {n_experts} ({reach/n_experts*100:.1f}%)")
    same = float(np.mean([len(set(a) & set(b)) for a, b in zip(free, capped)]))
    print(f"  mean overlap with the uncapped choice: {same:.2f} of {k} experts")

    print("\nBREAK IT: raise the cap. M=32 is every node, i.e. no cap at all")
    for m in (2, 4, 8, 16, 32):
        c = node_limited_topk(scores, n_nodes, k, m)
        nt = nodes_touched(c, n_experts, n_nodes)
        ov = float(np.mean([len(set(a) & set(b)) for a, b in zip(free, c)]))
        print(f"  M={m}: {nt.mean():.3f} nodes/token, "
              f"{ib_bytes(nt)/1e6:8,.1f} MB IB, overlap {ov:.2f}/{k}")

    print("\nREAD THIS: the cap is a fan-out limit, exactly like capping the")
    print("number of shards a query may hit. It is not free -- a token gives")
    print("up some of its preferred experts -- but the traffic it removes is")
    print("on the slowest link in the machine. The report pairs the cap with")
    print("overlapping the IB and NVLink legs, which only works if the two")
    print("legs are roughly the same size. That is what M is tuned for.")


def demo_4_fp8():
    """Two 8-bit formats, neither of them good enough on its own."""
    line("DEMO 4: FP8 -- the other half of the byte reduction")

    print("format   exp  man     max finite   min normal   steps per octave")
    for name, (eb, mb, bias, mx) in FP8_FORMATS.items():
        print(f"  {name}    {eb}    {mb}   {mx:12,.0f}   "
              f"{2.0**(1-bias):10.3e}   {2**mb:>4d}")

    rng = np.random.default_rng(11)
    tensors = {
        "weights   (N(0,0.02))": rng.normal(0, 0.02, 4096),
        "activation (N(0,1))  ": rng.normal(0, 1.0, 4096),
        "act+outlier (x1500)  ": None,
        "grads     (N(0,1e-5))": rng.normal(0, 1e-5, 4096),
    }
    a = rng.normal(0, 1.0, 4096)
    a[7] = 1500.0                 # the one activation outlier that kills you
    tensors["act+outlier (x1500)  "] = a

    # What real activations look like: mostly ordinary, with the occasional
    # enormous value. One scale for the lot is set by the loudest number in
    # the lot, and every quiet value underneath it rounds to nothing.
    layered = rng.normal(0, 1.0, 4096)
    quiet = np.ones(4096, dtype=bool)
    for j in range(0, 4096, 1024):
        layered[j] = 3000.0
        quiet[j] = False

    print("\ndirect cast, one tensor, no scaling -- relative L2 error:")
    print("  tensor                     e4m3        e5m2     e4m3 finite?")
    for name, t in tensors.items():
        q4, q5 = fp8_cast(t, "e4m3"), fp8_cast(t, "e5m2")
        fin = bool(np.isfinite(q4).all())
        e4 = rel_error(t, q4)
        e5 = rel_error(t, q5)
        print(f"  {name}  {e4:10.6f}  {e5:10.6f}     {fin}")

    print("\nwhich tensor dies first: the one with the outlier.")
    print(f"  max |activation| = {np.abs(a).max():,.1f}  vs e4m3 max 448")
    q = fp8_cast(a, "e4m3")
    print(f"  values that became inf: {int((~np.isfinite(q)).sum())}")
    g = tensors["grads     (N(0,1e-5))"]
    print(f"  gradients underflow e4m3's min normal: "
          f"{int((fp8_cast(g, 'e4m3') == 0).sum())} of 4096 -> 0")

    print("\nnow fine-grained scaling. First the outlier tensor, one scale:")
    deq, sc = fp8_blockwise(a, "e4m3", 4096)
    print(f"  direct cast, no scale : {rel_error(a, fp8_cast(a, 'e4m3')):.6f}"
          f"   <- overflowed, the tensor is gone")
    print(f"  one FP32 scale        : {rel_error(a, deq):.6f}   finite again")

    print("\nbut one scale is set by the loudest value, and the quiet ones pay.")
    print("4096 activations, N(0,1), with four values of 3000 dropped in.")
    print("Error measured on the 4092 ORDINARY values only:")
    print("  block   err on quiet values   dead (rounded to 0)   bits/value")
    for block in (4096, 512, 128, 32):
        deq, sc = fp8_blockwise(layered, "e4m3", block)
        dead = int((deq[quiet] == 0).sum())
        bits = 8 + 32 / block
        print(f"  {block:5d}   {rel_error(layered[quiet], deq[quiet]):18.6f}"
              f"   {dead:19d}   {bits:10.3f}")

    print("\nmemory and traffic, DeepSeek-V3 shapes:")
    act = 4096 * D_MODEL
    print(f"  one activation tile, {4096} tokens x {D_MODEL}:")
    for nm, b in (("BF16", 2), ("FP8 (1x128 scales)", 1 + 4 / 128)):
        print(f"    {nm:20s} {act*b/1e6:8,.2f} MB")
    print(f"    saving: {(1 - (1+4/128)/2)*100:.1f}%")
    ib_8, nvl_8 = all_to_all_bytes_per_token(NODE_CAP_M, 1, 2)
    ib_16, nvl_16 = all_to_all_bytes_per_token(NODE_CAP_M, 2, 2)
    print(f"  all-to-all bytes/token: BF16 dispatch {ib_16+nvl_16:,} B ->")
    print(f"                          FP8  dispatch {ib_8+nvl_8:,} B "
          f"({(1-(ib_8+nvl_8)/(ib_16+nvl_16))*100:.1f}% less)")

    print("\nREAD THIS: e4m3 has the precision and not the range; e5m2 has the")
    print("range and not the precision. V3 uses e4m3 nearly everywhere and")
    print("wins the range back with per-block scales -- 1x128 for activations,")
    print("128x128 for weights -- plus master weights and optimiser state kept")
    print("wider, and accumulation promoted to FP32. The format did not get")
    print("better. The blocks got smaller. Same argument as the quantisation")
    print("page, one octave further down.")


def demo_5_multi_token_prediction():
    """Two heads, two targets per position, one forward pass."""
    line("DEMO 5: multi-token prediction -- a denser training signal")

    n_tokens = 4096
    print(f"micro-batch: {n_tokens} positions")
    for depth in (0, 1, 2):
        heads = depth + 1
        print(f"  MTP depth {depth} ({heads} head{'s' if heads>1 else ''}): "
              f"{n_tokens*heads:,} prediction targets per forward pass  "
              f"({heads:.1f}x)")

    print("\nwhat each head predicts, for the sentence positions 0..4:")
    toks = ["the", "cat", "ate", "the", "food"]
    for i in range(len(toks) - 2):
        print(f"  pos {i} ({toks[i]:5s}) -> head 1: {toks[i+1]:5s}"
              f"   head 2: {toks[i+2]:5s}")

    print("\nreuse at inference -- head 2 becomes a draft model.")
    print("expected tokens accepted per decoding step = 1 + p(second accepted):")
    for p in (0.0, 0.5, 0.85, 0.90, 1.0):
        print(f"  acceptance {p:.2f}  ->  {1+p:.2f} tokens/step "
              f"({1+p:.2f}x, before the extra head's own cost)")

    print("\nREAD THIS: the report states the second-token acceptance rate is")
    print("85-90% and that using MTP for speculative decoding improved decoding")
    print("throughput by about 1.8x. The 1+p arithmetic above is mine, not")
    print("theirs, and it ignores verification cost. What the report is clear")
    print("about is that MTP is a TRAINING objective -- the extra head is")
    print("dropped for normal inference -- and that reusing it for speculation")
    print("is a bonus, not the reason it exists.")


def demo_6_overlap():
    """Simulate the schedule. Not a measurement of theirs."""
    line("DEMO 6: overlapping communication with compute")

    stages, micro = 8, 32
    compute, comm = 1.0, 0.8       # per microbatch, arbitrary units

    print(f"pipeline: {stages} stages, {micro} microbatches, "
          f"compute {compute}, all-to-all {comm} per microbatch")
    print("\n  schedule            makespan   device busy   bubble")
    for label, ov in (("no overlap", False), ("comm hidden", True)):
        mk, busy, bub = pipeline_bubble(stages, micro, compute, comm, ov)
        print(f"  {label:18s}  {mk:8.2f}   {busy*100:9.1f}%   {bub*100:6.1f}%")

    base = pipeline_bubble(stages, micro, compute, comm, False)[0]
    best = pipeline_bubble(stages, micro, compute, comm, True)[0]
    print(f"\n  wall-clock penalty for not overlapping: {base/best:.2f}x")
    print(f"  illustration only: a 56.7-day run with this shape becomes "
          f"{56.7*base/best:.1f} days")

    print("\nsame pipeline as the all-to-all gets more expensive:")
    print("  comm    bubble (no overlap)   bubble (overlapped)   speedup")
    for c in (0.2, 0.5, 0.8, 1.2, 2.0):
        m0, _, b0 = pipeline_bubble(stages, micro, compute, c, False)
        m1, _, b1 = pipeline_bubble(stages, micro, compute, c, True)
        print(f"  {c:4.1f}    {b0*100:16.1f}%   {b1*100:18.1f}%   "
              f"{m0/m1:8.2f}x")

    print("\nmore microbatches, the classic bubble fix, no overlap:")
    for m in (8, 16, 32, 64):
        _, _, b = pipeline_bubble(stages, m, compute, comm, False)
        print(f"  {m:3d} microbatches: bubble {b*100:.1f}%")

    print("\nREAD THIS: this is a SIMULATION of a schedule, not a measurement")
    print("of DeepSeek's. Two separate things shrink the bubble: more")
    print("microbatches, which is free and old, and hiding the all-to-all")
    print("underneath compute, which is what DualPipe does. Note the second")
    print("column: once comm is hidden, making it more expensive stops")
    print("mattering until it exceeds compute. That is the whole design goal")
    print("-- get the communication cost under the compute cost and it")
    print("disappears from the wall clock.")


def demo_7_the_cost_number():
    """The most misreported number in the field. Do the division yourself."""
    line("DEMO 7: the $5.6M number, divided out")

    hours = 2.788e6
    rate = 2.0
    print(f"  final pre-training GPU-hours (report) : {hours:,.0f} H800-hours")
    print(f"  assumed rental price (report's own)   : ${rate:.2f} per GPU-hour")
    print(f"  product                               : ${hours*rate/1e6:.3f}M")
    print(f"  on {2048} GPUs that is "
          f"{hours/2048/24:.1f} days of wall clock")
    print(f"  tokens: 14.8e12 over 671e9 params, 37e9 active per token")
    print(f"  tokens per active parameter           : {14.8e12/37e9:,.0f}")
    print(f"  approx forward+backward FLOPs (6*N*D) : "
          f"{6*37e9*14.8e12/1e21:.1f} ZFLOP")
    eff = (6 * 37e9 * 14.8e12) / (hours * 3600 * 2048 / 2048 * BF16_PEAK_FLOPS)
    print(f"  implied MFU against BF16 peak         : {eff*100:.1f}%")
    print(f"  against the FP8 peak they actually used: {eff*50:.1f}%")

    print("\nwhat that number does NOT include: research, ablations, failed")
    print("runs, the earlier DeepSeek models this one builds on, salaries, or")
    print("the cluster itself -- the report says so directly. It is the cost")
    print("of the final training run at a rental price they assumed. Quoting")
    print("it as 'the cost of building DeepSeek-V3' is wrong, and it is the")
    print("single most misreported figure in the field.")


if __name__ == "__main__":
    demo_1_the_constraint()
    demo_2_balancing()
    demo_3_node_limited_routing()
    demo_4_fp8()
    demo_5_multi_token_prediction()
    demo_6_overlap()
    demo_7_the_cost_number()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  0. the constraint = H800: H100 arithmetic, roughly half the NVLink.
     An MoE layer is an all-to-all. All-to-all is what a narrow link hates.
  1. MLA            = compress the KV cache to one latent vector per token.
                      Theirs, from V2. Its own subject; see the KV-cache page.
  2. balance        = a per-expert BIAS on the selection score, nudged by
                      recent load. No auxiliary loss, so no rogue gradient.
  3. node cap       = a token may reach at most 4 nodes. A fan-out limit.
  4. FP8            = e4m3 with per-block scales, FP32 accumulation, wider
                      master weights. Halves the dispatch bytes.
  5. MTP            = a second head predicts token t+2. Denser signal in
                      training, a free draft model at inference.
  6. DualPipe       = overlap the all-to-all with compute so the bubble
                      stops growing when comm does.
  7. below CUDA     = hand-written PTX for the comm kernels, and a slice of
                      each GPU's SMs reserved for communication.
  8. no tensor par. = the most communication-hungry parallelism, avoided.

  The order matters. Every line after 0 exists because of line 0.
""")
