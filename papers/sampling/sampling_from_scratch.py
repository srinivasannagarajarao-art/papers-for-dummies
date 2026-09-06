"""
Temperature, top-k and top-p -- for programmers, not researchers.

Run it:      python3 papers/sampling/sampling_from_scratch.py
Debug it:    breakpoint() inside top_p and print `keep` for a few p values.

NumPy only. No model, no training, no torch. Sampling is arithmetic over a
probability vector, so the whole paper fits in a handful of short functions
and you can watch every number move.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- softmax. Turns a row of logits into percentages summing to 1.
#
# The `- max` is not maths, it's overflow protection: exp(1000) is inf,
# exp(1000-1000) is 1, and the ratio is unchanged.
# ---------------------------------------------------------------------------
def softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# STAGE 1 -- TEMPERATURE. It divides the LOGITS, before the softmax.
#
# T < 1 stretches the gaps apart  -> sharper, closer to argmax.
# T > 1 squashes the gaps together -> flatter, closer to uniform.
# It is monotonic, so it can never reorder the vocabulary: the most likely
# token before is the most likely token after. It is a contrast slider.
#
# T = 0 is division by zero. Every real implementation special-cases it to
# mean "greedy" instead of computing it.
# ---------------------------------------------------------------------------
def apply_temperature(logits, T):
    if T == 0:
        raise ZeroDivisionError("T=0: real code branches to argmax instead")
    return logits / T


def entropy_nats(p):
    """How undecided the distribution is. 0 = certain, ln(V) = no idea."""
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())


# ---------------------------------------------------------------------------
# STAGE 2 -- TOP-K (Fan et al., 2018). Keep the k highest-probability tokens,
# throw the rest away, renormalise what is left back to sum 1.
#
# k is a fixed COUNT. It has no idea what shape the distribution is, so the
# probability mass it happens to capture swings wildly step to step.
# ---------------------------------------------------------------------------
def top_k(p, k):
    keep = np.argsort(-p)[:k]              # indices of the k largest
    out = np.zeros_like(p)
    out[keep] = p[keep]
    return out / out.sum(), np.sort(keep)  # RENORMALISE AFTER truncating


# ---------------------------------------------------------------------------
# STAGE 3 -- TOP-P / nucleus sampling (Holtzman et al., 2019). Keep the
# SMALLEST set whose cumulative probability reaches p, then renormalise.
#
# p is a fixed MASS, so the size of the set is decided by the distribution,
# not by you. Confident step -> a couple of tokens. Flat step -> dozens.
# That adaptivity is the whole paper.
# ---------------------------------------------------------------------------
def top_p(p, p_thresh):
    order = np.argsort(-p)                 # most likely first
    cum = np.cumsum(p[order])
    n = int(np.searchsorted(cum, p_thresh) + 1)  # include the crossing token
    n = min(n, len(p))
    keep = order[:n]
    out = np.zeros_like(p)
    out[keep] = p[keep]
    return out / out.sum(), np.sort(keep)


# ---------------------------------------------------------------------------
# STAGE 4 -- the pipeline every inference server runs, in this order:
#     logits -> temperature -> softmax -> top-k -> top-p -> renormalise -> draw
# Temperature acts on logits. Truncation acts on probabilities. Swap those
# two and you get a different distribution, not a rounding difference.
# ---------------------------------------------------------------------------
def sample_once(logits, T=1.0, k=None, p_thresh=None, rng=None):
    rng = rng or np.random.default_rng(0)
    probs = softmax(logits) if T == 0 else softmax(apply_temperature(logits, T))
    if T == 0:
        return int(np.argmax(logits))      # the special case, not a division
    if k is not None:
        probs, _ = top_k(probs, k)
    if p_thresh is not None:
        probs, _ = top_p(probs, p_thresh)
    return int(rng.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# STAGE 5 -- a toy first-order Markov chain, so "greedy loops" is something
# you watch happen rather than something you are told.
# ---------------------------------------------------------------------------
CHAIN_TOKENS = ["the", "model", "is", "a", "learns", "from", "data", "good"]
CHAIN = np.array([
    # the   model   is      a     learns  from   data   good
    [0.05,  0.55,  0.03,  0.05,  0.02,  0.02,  0.18,  0.10],  # the
    [0.06,  0.02,  0.50,  0.03,  0.24,  0.05,  0.06,  0.04],  # model
    [0.20,  0.04,  0.02,  0.45,  0.03,  0.06,  0.05,  0.15],  # is
    [0.05,  0.48,  0.02,  0.02,  0.03,  0.05,  0.25,  0.10],  # a
    [0.10,  0.05,  0.05,  0.05,  0.02,  0.55,  0.13,  0.05],  # learns
    [0.25,  0.05,  0.03,  0.07,  0.02,  0.03,  0.50,  0.05],  # from
    [0.30,  0.06,  0.25,  0.08,  0.05,  0.06,  0.10,  0.10],  # data
    [0.35,  0.20,  0.05,  0.15,  0.05,  0.05,  0.10,  0.05],  # good
])


def walk(start, n_steps, mode, T=1.0, k=None, p_thresh=None, rng=None):
    """Walk the chain. mode='greedy' takes the argmax, 'sample' draws."""
    rng = rng or np.random.default_rng(0)
    state, seq, surp = start, [start], []
    for _ in range(n_steps):
        row = CHAIN[state]
        if mode == "greedy":
            nxt = int(np.argmax(row))
        else:
            logits = np.log(row)           # probabilities back to logits
            nxt = sample_once(logits, T=T, k=k, p_thresh=p_thresh, rng=rng)
        surp.append(-np.log2(row[nxt]))    # bits of surprise, under the model
        seq.append(nxt)
        state = nxt
    return seq, float(np.mean(surp))


def find_cycle(seq):
    """A real loop: the shortest period the tail actually repeats at."""
    for period in range(1, len(seq) // 3 + 1):
        tail = seq[-3 * period:]
        if tail[:period] * 3 == tail:
            return tail[:period]
    return []


def show(tokens, seq):
    return " ".join(tokens[i] for i in seq)


# ===========================================================================
# THE ONE FIXED DISTRIBUTION. Everything below is measured against this.
# Context: "the meeting is scheduled for ___". A realistic shape: one clear
# favourite, three plausible alternatives, and a long junk tail.
# ===========================================================================
VOCAB = ["Monday", "tomorrow", "next", "the", "March", "later", "noon",
         "sometime", "whenever", "banana", "purple", "17", "zzz", "qux",
         "wombat", "asdf"]
LOGITS = np.array([6.0, 5.2, 4.6, 4.1, 3.2, 2.8, 2.4, 1.9,
                   1.2, 0.6, 0.3, 0.0, -0.4, -0.8, -1.3, -2.0])
BASE = softmax(LOGITS)

# A SECOND step, same vocabulary, much flatter: the model genuinely does not
# know which word comes next. This is where fixed k falls apart.
FLAT_LOGITS = np.array([1.30, 1.22, 1.15, 1.10, 1.05, 1.00, 0.95, 0.90,
                        0.85, 0.80, 0.75, 0.70, 0.62, 0.55, 0.45, 0.30])
FLAT = softmax(FLAT_LOGITS)


def table(p, idx=None, n=8):
    idx = np.argsort(-p)[:n] if idx is None else idx
    for i in idx:
        bar = "#" * int(round(p[i] * 60))
        print(f"    {VOCAB[i]:>9}  {p[i]:.4f}  {bar}")


def demo_1_the_distribution():
    print("\n=== 1. the one distribution, printed once ==========================")
    print("  context: 'the meeting is scheduled for ___'   (16-token vocab)")
    table(BASE, idx=np.arange(16))
    print(f"  sums to {BASE.sum():.6f}   entropy {entropy_nats(BASE):.4f} nats")
    print(f"  tail mass below rank 8: {BASE[np.argsort(-BASE)[8:]].sum():.4f}")
    print("\n  the SECOND step, same vocab, flat -- the model is unsure:")
    table(FLAT, idx=np.arange(16))
    print(f"  sums to {FLAT.sum():.6f}   entropy {entropy_nats(FLAT):.4f} nats")


def demo_2_temperature():
    print("\n=== 2. temperature: the contrast slider ============================")
    print("  applied to LOGITS, before softmax.  top 4 tokens each time.")
    for T in (0.2, 0.5, 0.8, 1.0, 1.5, 2.0):
        p = softmax(apply_temperature(LOGITS, T))
        top = np.argsort(-p)[:4]
        row = "  ".join(f"{VOCAB[i]}={p[i]:.4f}" for i in top)
        print(f"  T={T:<4} H={entropy_nats(p):.4f}  {row}")
    print("\n  ranking is IDENTICAL at every temperature:")
    ranks = [tuple(np.argsort(-softmax(apply_temperature(LOGITS, T))))
             for T in (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)]
    print(f"    all six orderings equal? {len(set(ranks)) == 1}")
    print("    only the GAPS changed. temperature cannot promote a token.")
    print("\n  T = 0 exactly:")
    try:
        apply_temperature(LOGITS, 0.0)
    except ZeroDivisionError as e:
        print(f"    raises: {e}")
    print(f"    raw numpy would give: {LOGITS[:3] / 1e-12} ... then inf/nan")
    print(f"    what real code does instead: argmax -> '{VOCAB[int(np.argmax(LOGITS))]}'")


def demo_3_top_k():
    print("\n=== 3. top-k: a fixed count, blind to shape =======================")
    for k in (1, 3, 5, 10):
        q, keep = top_k(BASE, k)
        mass = BASE[keep].sum()
        show_n = min(k, 3)
        kept = " ".join(f"{VOCAB[i]}={q[i]:.4f}" for i in np.argsort(-q)[:show_n])
        more = "" if k <= show_n else " ..."
        print(f"  k={k:<3} keeps {mass:.4f} of the mass | {kept}{more}")
    print("\n  now the SAME k on the two different steps:")
    for k in (3, 5, 10):
        m1 = BASE[top_k(BASE, k)[1]].sum()
        m2 = FLAT[top_k(FLAT, k)[1]].sum()
        print(f"  k={k:<3} confident step keeps {m1:.4f} | flat step keeps {m2:.4f}"
              f"  (gap {m1 - m2:.4f})")
    print("  one number cannot be right for both. that is Holtzman's complaint.")


def demo_4_top_p():
    print("\n=== 4. top-p: a fixed mass, set size adapts ======================")
    print("  p     confident step        flat step")
    for pt in (0.5, 0.8, 0.9, 0.95, 1.0):
        _, k1 = top_p(BASE, pt)
        _, k2 = top_p(FLAT, pt)
        print(f"  {pt:<5} {len(k1):>2} tokens, mass {BASE[k1].sum():.4f}   "
              f"{len(k2):>2} tokens, mass {FLAT[k2].sum():.4f}")
    _, k1 = top_p(BASE, 0.9)
    _, k2 = top_p(FLAT, 0.9)
    print(f"\n  at p=0.9 the nucleus is {len(k1)} tokens on the confident step "
          f"and {len(k2)} on the flat one.")
    print(f"    confident nucleus: {' '.join(VOCAB[i] for i in k1)}")
    print(f"    flat nucleus:      {' '.join(VOCAB[i] for i in k2[:8])} ...")
    print(f"  p=1.0 truncates nothing. tail mass still reachable: "
          f"{BASE[np.argsort(-BASE)[8:]].sum():.4f}, "
          f"'{VOCAB[-1]}' at {BASE[-1]:.6f}")


def demo_5_order_of_operations():
    print("\n=== 5. order matters: temperature then truncate, or the reverse ===")
    T, pt = 1.5, 0.9
    # A: the real order -- temperature on the logits, THEN nucleus on softmax.
    a, keep_a = top_p(softmax(apply_temperature(LOGITS, T)), pt)
    # B: the wrong order -- nucleus at T=1, THEN temperature the survivors.
    trunc, keep_b = top_p(BASE, pt)
    b = softmax(np.log(np.where(trunc > 0, trunc, 1e-30)) / T)
    b = np.where(trunc > 0, b, 0.0)
    b = b / b.sum()
    print(f"  T={T}, p={pt}")
    print(f"    temp-then-p keeps {len(keep_a)} tokens: "
          f"{' '.join(VOCAB[i] for i in keep_a)}")
    print(f"    p-then-temp keeps {len(keep_b)} tokens: "
          f"{' '.join(VOCAB[i] for i in keep_b)}")
    print(f"    same surviving set? {np.array_equal(keep_a, keep_b)}")
    for i in np.argsort(-a)[:5]:
        print(f"    {VOCAB[i]:>9}  temp-then-p {a[i]:.4f}   p-then-temp {b[i]:.4f}"
              f"   diff {a[i] - b[i]:+.4f}")
    print(f"  max abs difference: {np.abs(a - b).max():.4f}")
    print("  real servers do temperature -> top-k -> top-p. the reverse is a bug.")
    print("\n  and renormalising BEFORE truncating instead of after:")
    wrong = np.zeros_like(BASE)
    wrong[keep_a] = (BASE / BASE.sum())[keep_a]   # normalised, THEN cut
    print(f"    sum after cutting an already-normalised vector: {wrong.sum():.4f}")
    print("    np.random.choice raises: probabilities do not sum to 1")


def demo_6_greedy_loops():
    print("\n=== 6. why greedy decoding fails =================================")
    g_seq, g_surp = walk(0, 60, "greedy")
    print(f"  greedy from 'the': {show(CHAIN_TOKENS, g_seq[:12])} ...")
    cyc = find_cycle(g_seq)
    print(f"  loop detected, period {len(cyc)}: {show(CHAIN_TOKENS, cyc)}")
    print(f"  greedy avg surprisal: {g_surp:.4f} bits/token")
    rng = np.random.default_rng(0)
    s_seq, _ = walk(0, 60, "sample", T=1.0, p_thresh=0.9, rng=rng)
    print(f"\n  sampled (T=1.0, p=0.9): {show(CHAIN_TOKENS, s_seq[:11])} ...")
    print(f"  loop detected? {'no' if not find_cycle(s_seq) else 'yes'}"
          f" -- it revisits states without getting stuck")
    rng = np.random.default_rng(1)
    surps = [walk(0, 60, "sample", T=1.0, p_thresh=0.9, rng=rng)[1]
             for _ in range(200)]
    print(f"  sampled avg surprisal over 200 runs: {np.mean(surps):.4f} bits/token")
    print(f"  greedy is {np.mean(surps) - g_surp:.4f} bits/token LESS surprising")
    print("  -- and unreadable. maximising likelihood is not the goal.")
    rng = np.random.default_rng(2)
    k1 = [walk(0, 60, "sample", k=1, rng=rng)[0] for _ in range(3)]
    same = all(s == g_seq for s in k1)
    print(f"\n  top-k with k=1 is greedy wearing a hat: "
          f"{'identical to the greedy walk' if same else 'differs'}")
    print(f"    k=1 walk: {show(CHAIN_TOKENS, k1[0][:12])} ...")


if __name__ == "__main__":
    demo_1_the_distribution()
    demo_2_temperature()
    demo_3_top_k()
    demo_4_top_p()
    demo_5_order_of_operations()
    demo_6_greedy_loops()
    print()
