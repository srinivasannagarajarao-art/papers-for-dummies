"""
Fast Inference from Transformers via Speculative Decoding -- for programmers.

Run it:      python3 speculative_decoding_from_scratch.py
Debug it:    breakpoint in verify_round() and step one drafted token at a time.

No torch. No neural net at all -- the "models" here are explicit probability
tables, because the paper's claim is about DISTRIBUTIONS, and a table is the
only model whose true distribution you can print and compare against.

Every core function is under 20 lines.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- the "models".
#
# A model here is a table: row = current token, row contents = the next-token
# distribution. That is an order-1 Markov chain, and it is enough. Everything
# the paper proves is about one call producing one distribution; the fact that
# a real target model is 70B parameters and this one is a 6x6 array changes
# nothing about the acceptance rule.
# ---------------------------------------------------------------------------
VOCAB = 6
TOKENS = ["a", "b", "c", "d", "e", "f"]


def random_table(seed, sharpness=1.0):
    """A row-stochastic (VOCAB, VOCAB) table. Higher sharpness = peakier."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, sharpness, (VOCAB, VOCAB))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def blend(target, other, alpha):
    """Draft = alpha*target + (1-alpha)*other. alpha=1 -> perfect draft."""
    return alpha * target + (1.0 - alpha) * other


def temper(table, temperature):
    """Re-sharpen or flatten a table. T<1 peakier, T>1 flatter."""
    logits = np.log(np.clip(table, 1e-12, None)) / temperature
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# STAGE 1 -- sampling. One uniform in [0,1) -> one token, by inverse CDF.
# Doing it this way (instead of rng.choice) keeps every demo reproducible with
# a countable number of random draws, so two runs print identical numbers.
# ---------------------------------------------------------------------------
def sample_from(probs, u):
    return int(np.searchsorted(np.cumsum(probs), u, side="right"))


# ---------------------------------------------------------------------------
# STAGE 2 -- THE WHOLE PAPER, part 1: the acceptance rule.
#
# The draft model proposed token x, and it proposed it with probability q[x].
# The target model would have said p[x]. Then:
#
#     accept x with probability   min(1, p[x] / q[x])
#
# q[x] > p[x] means the draft over-proposes x, so we keep only p/q of those.
# q[x] <= p[x] means the draft under-proposes x, so we keep every one.
# ---------------------------------------------------------------------------
def accept_prob(p, q, x):
    return min(1.0, p[x] / q[x])


# ---------------------------------------------------------------------------
# STAGE 3 -- THE WHOLE PAPER, part 2: the residual distribution.
#
# Accepting alone under-samples the tokens the draft was too shy about. On a
# rejection we do NOT resample from p -- that would double-count. We resample
# from the positive part of (p - q), normalised. That is exactly the mass the
# accept step failed to deliver, so the two steps sum back to p. This one line
# is the difference between "provably identical output" and "usually fine".
# ---------------------------------------------------------------------------
def residual(p, q):
    r = np.clip(p - q, 0.0, None)
    s = r.sum()
    if s <= 0:                     # degenerate: p is fully covered by q
        return p / p.sum()
    return r / s


# ---------------------------------------------------------------------------
# STAGE 4 -- draft k tokens, autoregressively, with the cheap model.
# k cheap sequential calls. This is the part you are betting will be thrown
# away when you guess wrong.
# ---------------------------------------------------------------------------
def draft_tokens(Q, ctx, k, rng):
    cur, drafted = ctx, []
    for _ in range(k):
        cur = sample_from(Q[cur], rng.random())
        drafted.append(cur)
    return drafted


# ---------------------------------------------------------------------------
# STAGE 5 -- verify all k drafts in ONE target call, then emit.
#
# The target sees the prefix plus all k drafted tokens at once, so it returns
# k+1 next-token distributions for the price of one forward pass (the same
# trick as training-time teacher forcing). Then walk them left to right:
#
#   accept  -> emit the drafted token, move on
#   reject  -> emit one token from residual(p, q) and STOP. Everything after
#              the rejection was conditioned on a token that no longer exists.
#   all k accepted -> you also get a free bonus token from the last
#              distribution, because the target already computed it.
#
# `rule` selects deliberate breakages; "correct" is the paper.
# ---------------------------------------------------------------------------
def verify_round(P, Q, ctx, drafted, rng, rule="correct"):
    prefixes = [ctx] + drafted          # k+1 positions, one target call
    out, n_accepted = [], 0
    for i, x in enumerate(drafted):
        p, q = P[prefixes[i]], Q[prefixes[i]]
        if rule == "greedy":
            ok = (x == int(np.argmax(p)))          # BREAK: argmax agreement
        else:
            ok = rng.random() < accept_prob(p, q, x)
        if ok:
            out.append(x)
            n_accepted += 1
            continue
        if rule == "no_resample":
            out.append(sample_from(p, rng.random()))          # BREAK
        else:
            out.append(sample_from(residual(p, q), rng.random()))
        if rule == "continue":
            out.extend(drafted[i + 1:])                       # BREAK
        return out, n_accepted
    out.append(sample_from(P[prefixes[-1]], rng.random()))     # bonus token
    return out, n_accepted


def spec_generate(P, Q, ctx, n_tokens, k, rng, rule="correct"):
    """Emit at least n_tokens. Returns (tokens, target_calls, accepted)."""
    out, calls, accepted = [], 0, 0
    while len(out) < n_tokens:
        drafted = draft_tokens(Q, ctx, k, rng)
        new, n_acc = verify_round(P, Q, ctx, drafted, rng, rule)
        calls += 1
        accepted += n_acc
        out.extend(new)
        ctx = out[-1]
    return out[:n_tokens], calls, accepted


# ---------------------------------------------------------------------------
# STAGE 6 -- the cost model.
#
#   speedup = E[tokens per round] / (1 + k*c)
#
# c = one draft-model token as a fraction of one target forward pass. The
# numerator is what you gained; the denominator is one target call plus k
# sequential draft calls. Break-even: c < (E[tokens] - 1) / k.
# ---------------------------------------------------------------------------
def speedup(tokens_per_round, k, c):
    return tokens_per_round / (1.0 + k * c)


def break_even_c(tokens_per_round, k):
    return (tokens_per_round - 1.0) / k


# ---------------------------------------------------------------------------
# STAGE 7 -- total variation distance. Half the sum of absolute differences
# between two distributions: "the largest probability either one can disagree
# about on any event". 0.00 means identical.
# ---------------------------------------------------------------------------
def tv_distance(a, b):
    return 0.5 * np.abs(a - b).sum()


# ---------------------------------------------------------------------------
# STAGE 8 -- MAC counting, borrowed wholesale from the KV-cache page, because
# the free compute speculative decoding spends is the idle compute that page
# measured. Verifying k tokens reads the weights once, same as decoding one.
# ---------------------------------------------------------------------------
def macs_per_token(d_model, d_ff, layers, n_ctx):
    return layers * (4 * d_model * d_model + 2 * d_model * n_ctx
                     + 2 * d_model * d_ff)


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


TARGET = random_table(seed=1, sharpness=2.5)
UNIFORM = np.full((VOCAB, VOCAB), 1.0 / VOCAB)
GOOD_DRAFT = blend(TARGET, random_table(seed=2, sharpness=2.5), 0.85)
MED_DRAFT = blend(TARGET, random_table(seed=2, sharpness=2.5), 0.55)
POOR_DRAFT = blend(TARGET, random_table(seed=3, sharpness=2.5), 0.20)
CTX0 = 0


def joint_two(P, ctx):
    """True target distribution over the first TWO emitted tokens."""
    return np.outer(P[ctx], np.ones(VOCAB)) * P


def empirical_joint(P, Q, ctx, k, trials, rule, seed):
    rng = np.random.default_rng(seed)
    counts = np.zeros((VOCAB, VOCAB))
    for _ in range(trials):
        toks, _, _ = spec_generate(P, Q, ctx, 2, k, rng, rule)
        counts[toks[0], toks[1]] += 1
    return counts / counts.sum()


def demo_1_the_rule_by_hand():
    line("DEMO 1: the acceptance rule, one token, arithmetic you can follow")
    p, q = TARGET[CTX0], GOOD_DRAFT[CTX0]
    print("  context token          : '%s'" % TOKENS[CTX0])
    print("  target  p(.|ctx)       :", p)
    print("  draft   q(.|ctx)       :", q)
    for x in (1, 4):
        a = accept_prob(p, q, x)
        print("\n  draft proposes '%s'" % TOKENS[x])
        print("    q[%s] = %.4f, p[%s] = %.4f" % (TOKENS[x], q[x],
                                                  TOKENS[x], p[x]))
        print("    accept with min(1, p/q) = min(1, %.4f/%.4f) = %.4f"
              % (p[x], q[x], a))
        print("    %s" % ("draft under-proposes it -> always accept"
                          if a >= 1.0 else
                          "draft over-proposes it -> reject %.1f%% of the time"
                          % (100 * (1 - a))))
    print("\n  on rejection, resample from residual = norm(max(0, p - q)):")
    print("    p - q          :", p - GOOD_DRAFT[CTX0])
    print("    residual       :", residual(p, GOOD_DRAFT[CTX0]))
    print("\nREAD THIS: the residual is the mass the accept step failed to")
    print("deliver. Accept + residual add back up to exactly p. That is the")
    print("whole proof, and it is two lines of arithmetic.")


def demo_2_correctness():
    line("DEMO 2: the output distribution IS the target's (the whole point)")
    trials, k = 60000, 4
    truth = joint_two(TARGET, CTX0)
    print("  %d rounds, k=%d drafted tokens per round, first two tokens kept\n"
          % (trials, k))

    emp = empirical_joint(TARGET, GOOD_DRAFT, CTX0, k, trials, "correct", 10)
    print("  first-token marginal, target vs speculative decoding:")
    print("    token      target   speculative    diff")
    for t in range(VOCAB):
        print("      %s       %.4f      %.4f      %+.4f"
              % (TOKENS[t], TARGET[CTX0][t], emp.sum(axis=1)[t],
                 emp.sum(axis=1)[t] - TARGET[CTX0][t]))
    print("\n  total variation distance on the two-token joint:")
    print("    correct rule                     : %.4f" % tv_distance(emp,
                                                                      truth))

    for rule, label in [("greedy", "accept on argmax agreement    "),
                        ("no_resample", "resample from p, not residual "),
                        ("continue", "keep drafts after a rejection  ")]:
        b = empirical_joint(TARGET, GOOD_DRAFT, CTX0, k, trials, rule, 10)
        print("    BROKEN: %s : %.4f" % (label, tv_distance(b, truth)))

    ref = empirical_joint(TARGET, TARGET, CTX0, 1, trials, "correct", 11)
    print("\n  sampling-error floor (target model sampled directly): %.4f"
          % tv_distance(ref, truth))
    print("\nREAD THIS: the correct rule sits at the sampling-error floor. The")
    print("broken ones sit an order of magnitude above it. That gap is not")
    print("noise -- it is a different model answering your users.")


def measure(P, Q, k, trials, seed):
    """Empirical acceptance rate and tokens emitted per target call."""
    rng = np.random.default_rng(seed)
    toks, calls, acc = 0, 0, 0
    ctx = CTX0
    for _ in range(trials):
        drafted = draft_tokens(Q, ctx, k, rng)
        new, n_acc = verify_round(P, Q, ctx, drafted, rng)
        toks += len(new)
        calls += 1
        acc += n_acc
        ctx = new[-1]
    return acc / (calls * k), toks / calls


def demo_3_acceptance_and_speedup():
    line("DEMO 3: how close the draft is decides everything")
    k, trials, c = 4, 40000, 0.10
    print("  k=%d drafted tokens per round, draft cost c=%.2f of a target"
          " call\n" % (k, c))
    print("  draft quality      accept rate   tokens/target call   speedup")
    others = random_table(seed=2, sharpness=2.5)
    for alpha in (1.0, 0.95, 0.85, 0.6, 0.3, 0.0):
        Q = blend(TARGET, others, alpha)
        rate, tpr = measure(TARGET, Q, k, trials, 20)
        print("  alpha=%.2f %s   %.4f        %.4f            %.3fx"
              % (alpha,
                 "(perfect)" if alpha == 1.0 else
                 "(junk)   " if alpha == 0.0 else "         ",
                 rate, tpr, speedup(tpr, k, c)))
    print("\n  temperature mismatch (same model, wrong sampling temperature):")
    for T in (1.0, 1.3, 2.0, 0.5):
        Q = temper(TARGET, T)
        rate, tpr = measure(TARGET, Q, k, trials, 21)
        print("    draft T=%.1f vs target T=1.0 : accept %.4f, "
              "tokens/call %.4f, %.3fx"
              % (T, rate, tpr, speedup(tpr, k, c)))
    print("\nREAD THIS: acceptance rate is the only lever that matters, and it")
    print("is a property of the PAIR of models, not of either one. A draft")
    print("that is merely mistuned -- right weights, wrong temperature --")
    print("gives most of the speedup back.")


def demo_4_draft_length():
    line("DEMO 4: how many tokens to draft -- there is an optimum")
    trials, c = 40000, 0.10
    print("  middling draft (alpha=0.55), draft cost c=%.2f\n" % c)
    print("   k   accept rate   tokens/target call   cost 1+k*c   speedup")
    best = (0, 0.0)
    for k in (1, 2, 3, 4, 5, 6, 8, 12, 16):
        rate, tpr = measure(TARGET, MED_DRAFT, k, trials, 30)
        s = speedup(tpr, k, c)
        best = max(best, (s, k), key=lambda z: z[0])
        print("  %2d      %.4f        %.4f           %.2f        %.3fx"
              % (k, rate, tpr, 1 + k * c, s))
    print("\n  best k here: %d at %.3fx" % (best[1], best[0]))
    print("\nREAD THIS: tokens per round saturates -- after the first")
    print("rejection every remaining draft is thrown away, so drafting more")
    print("of them buys nothing while still costing c each. Too short wastes")
    print("the free compute; too long pays for work you will discard.")


def demo_5_cost_model():
    line("DEMO 5: break-even -- when a draft model makes you SLOWER")
    k, trials = 4, 40000
    _, tpr = measure(TARGET, MED_DRAFT, k, trials, 40)
    be = break_even_c(tpr, k)
    print("  middling draft, k=%d, tokens per target call = %.4f" % (k, tpr))
    print("  break-even draft cost c* = (tokens - 1)/k = %.4f" % be)
    print("  i.e. the draft model must cost less than %.1f%% of the target"
          " per token.\n" % (100 * be))
    print("  draft cost c    1 + k*c    speedup")
    for c in sorted([0.01, 0.05, 0.10, be, 0.60, 1.00]):
        s = speedup(tpr, k, c)
        tag = "  <-- break-even" if abs(c - be) < 1e-9 else (
            "  SLOWER than no speculation" if s < 1.0 else "")
        print("      %.4f      %.4f     %.3fx%s" % (c, 1 + k * c, s, tag))
    print("\n  a 'draft' that is a quarter of the target (c=0.25) and only")
    print("  as good as our poor draft:")
    _, tpr_poor = measure(TARGET, POOR_DRAFT, k, trials, 41)
    print("    tokens/call %.4f, speedup %.3fx"
          % (tpr_poor, speedup(tpr_poor, k, 0.25)))
    print("\nREAD THIS: two numbers decide the whole thing -- how often the")
    print("draft is accepted, and how cheap it is. A 7B drafting for a 70B is")
    print("c ~ 0.1. A 70B drafting for a 70B is c = 1 and always loses.")


def demo_6_where_the_free_compute_comes_from():
    line("DEMO 6: verifying k tokens costs about the same as decoding one")
    d, ff, layers, n = 4096, 11008, 32, 512
    weight_bytes = int(6.74e9 * 2)          # 7B params, fp16
    per_token = macs_per_token(d, ff, layers, n)
    print("  Llama-2-7B-shaped, %d tokens of context, fp16 weights" % n)
    print("  weights dragged off memory per forward pass: %s bytes (%.2f GiB)"
          % (f"{weight_bytes:,}", weight_bytes / 1024 ** 3))
    print("\n   tokens in the pass        MACs        MACs per byte moved")
    for k in (1, 2, 4, 8, 16):
        print("        %2d          %16s          %8.2f"
              % (k, f"{k * per_token:,}", k * per_token / weight_bytes))
    print("\n  a modern accelerator needs roughly 100-300 MACs/byte to stay")
    print("  compute-bound. Decoding one token misses that by ~1000x, so the")
    print("  arithmetic units idle while the weights stream past. Verifying")
    print("  16 tokens in the same pass moves the same bytes and does 16x the")
    print("  arithmetic -- and still does not saturate the chip.")
    print("\nREAD THIS: the extra tokens are close to free. That is the whole")
    print("reason this trick exists, and it is the KV-cache page's finding")
    print("spent rather than merely observed.")


if __name__ == "__main__":
    demo_1_the_rule_by_hand()
    demo_2_correctness()
    demo_3_acceptance_and_speedup()
    demo_4_draft_length()
    demo_5_cost_model()
    demo_6_where_the_free_compute_comes_from()
    print("\n" + "=" * 72)
    print("Now go break it. Start with STAGE 3: return p instead of the")
    print("residual, rerun demo 2, and watch the distribution drift.")
    print("=" * 72)
