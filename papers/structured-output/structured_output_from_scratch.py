"""
Constrained decoding -- for programmers, not researchers.

Run it:      python3 structured_output_from_scratch.py
Debug it:    breakpoint in allowed_tokens() and watch the mask shrink.

Anchored on Willard & Louf, 2023, "Efficient Guided Generation for Large
Language Models" (arXiv 2307.09702, the Outlines paper), with the earlier
grammar-constrained decoding of Geng et al., 2023 (arXiv 2305.13971).

There is NO language model in this script. A model contributes exactly one
thing to constrained decoding -- a vector of logits over the vocabulary -- so
I generate that vector with numpy and spend the rest of the file on the part
that actually does the work: the automaton, the mask, and the index that maps
parser states to allowed token sets.

NumPy and plain Python. Every core function is under 20 lines.
"""

import json
import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- the target grammar, as a character automaton.
#
# The schema we want back is exactly this, no whitespace allowed:
#
#     {"name":"<letters>","age":<digits>}
#
# That language is REGULAR, so a finite-state machine is enough. Real JSON with
# arbitrary nesting is not regular -- you need a stack to match brackets -- but
# a fixed schema of this shape is, and most application schemas are.
#
# `add_literal` walks a fixed string, minting one state per character. Same
# thing a hand-written recursive-descent parser does, just written down as data.
# ---------------------------------------------------------------------------
LETTERS = "abcdefghijklmnopqrstuvwxyz "
DIGITS = "0123456789"


def add_literal(delta, src, text, dst):
    """Chain states src -> ... -> dst, one transition per character of text."""
    state = src
    for ch in text[:-1]:
        nxt = max(delta) + 1                 # next free state number
        delta[state][ch] = nxt
        delta[nxt] = {}
        state = nxt
    delta[state][text[-1]] = dst
    return delta


# The five states that mean something. Everything else is a step inside one of
# the fixed literals, minted by add_literal.
START, NAME, DIGIT1, DIGITS_N, END = 0, 1, 2, 3, 4
NAMED = {START: "start", NAME: "in name string", DIGIT1: "need a digit",
         DIGITS_N: "in age", END: "done"}


def build_json_dfa():
    """The whole grammar. Returns (transitions, accepting states)."""
    delta = {s: {} for s in range(5)}
    add_literal(delta, START, '{"name":"', NAME)   # opening + the first key
    for ch in LETTERS:
        delta[NAME][ch] = NAME                     # string body, loops
    add_literal(delta, NAME, '","age":', DIGIT1)   # close string, second key
    for ch in DIGITS:
        if ch != "0":
            delta[DIGIT1][ch] = DIGITS_N           # at least one digit, and
        delta[DIGITS_N][ch] = DIGITS_N             # JSON forbids a leading 0

    delta[DIGITS_N]["}"] = END                     # done
    return delta, {END}


DELTA, ACCEPT = build_json_dfa()


def step_char(state, ch):
    """One character. None means 'this character cannot continue the doc'."""
    return DELTA.get(state, {}).get(ch)


def step_token(state, token):
    """Feed a whole token through, character by character.

    Returns the state after the token, or None if ANY character inside it
    fell off the automaton. A token is allowed or it is not; there is no
    partial credit, because you cannot emit half a token."""
    for ch in token:
        state = step_char(state, ch)
        if state is None:
            return None
    return state


# ---------------------------------------------------------------------------
# STAGE 1 -- the vocabulary. Real tokenisers ship ~50k of these; the point is
# that they are multi-character strings chosen by a compression algorithm that
# never heard of your grammar. See ../tokenization/.
#
# Note the deliberately awkward ones: '":"' and '","' straddle grammar
# boundaries, and '}' arrives glued to other characters in '0}' and '9}'.
# ---------------------------------------------------------------------------
VOCAB = [
    "{", "}", '"', ":", ",", "{\"", '":"', '","', '":', '"}',
    "name", "age", "nam", "e", "a", "g",
    "ada", "lovelace", " ", "ace", "lo", "ve",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "36", "180", "0}", "9}", "36}",
    "null", "true", "```", "json", "Sure", "here", "!", "\n",
]
TOK_ID = {t: i for i, t in enumerate(VOCAB)}
V = len(VOCAB)


# ---------------------------------------------------------------------------
# STAGE 2 -- THE WHOLE PAPER, in two functions.
#
# (1) which tokens could continue a valid document from here
# (2) set every other logit to -inf, then sample as usual
#
# That is the same trick as the causal mask on the attention page: -inf before
# the softmax, so the forbidden entries come out at exactly 0. Only here the
# mask runs over the VOCABULARY instead of over positions, and it is driven by
# a parser instead of by causality. See ../attention/.
# ---------------------------------------------------------------------------
def allowed_tokens(state):
    """Naive version: walk every token in the vocabulary through the DFA.

    Cost is O(vocab x token length) PER GENERATED TOKEN. This is the thing
    the Outlines paper removes."""
    out = {}
    for i, tok in enumerate(VOCAB):
        nxt = step_token(state, tok)
        if nxt is not None:
            out[i] = nxt
    return out


def apply_mask(logits, allowed):
    """Set every disallowed logit to -inf. Not 0 -- 0 is a perfectly ordinary
    logit that softmax will happily hand probability to."""
    masked = np.full_like(logits, -np.inf)
    idx = np.fromiter(allowed, dtype=int, count=len(allowed))
    masked[idx] = logits[idx]
    return masked


def softmax(x):
    x = x - np.max(x)                    # numerical hygiene, as ever
    e = np.exp(x)
    return e / e.sum()


def sample(logits, rng):
    """Plain categorical sampling. Identical to an unconstrained decoder --
    the constraint lives entirely in the logits handed to it."""
    p = softmax(logits)
    return int(rng.choice(len(p), p=p))


# ---------------------------------------------------------------------------
# STAGE 3 -- the index. This is the Outlines paper's contribution.
#
# The set of allowed tokens depends only on the parser state, and there are
# five of those. So compute it once, at load time, and the per-step cost drops
# from "scan the vocabulary" to "dict lookup". Precompute, then it is O(1) in
# the size of the grammar and the vocabulary for the rest of time.
# ---------------------------------------------------------------------------
def build_index():
    """state -> {token id: state after that token}. Built once, reused forever."""
    return {s: allowed_tokens(s) for s in DELTA}


# ---------------------------------------------------------------------------
# STAGE 4 -- the decode loop. This is a normal sampling loop from the GPT page
# with two lines added: look up what is allowed, mask before you sample.
# See ../gpt/.
# ---------------------------------------------------------------------------
def generate(rng, index, max_steps=400, mask_with_zero=False, trace=False):
    state, pieces = 0, []
    for _ in range(max_steps):
        logits = rng.normal(0, 2.0, V)                 # stand-in for a model
        allowed = index[state]
        if mask_with_zero:                             # the classic bug
            masked = np.where(np.isin(np.arange(V), list(allowed)), logits, 0.0)
        else:
            masked = apply_mask(logits, allowed)
        tok = sample(masked, rng)
        pieces.append(VOCAB[tok])
        if trace:
            print(f"  state {state} -> allowed {len(allowed):2d}/{V} "
                  f"-> emitted {VOCAB[tok]!r}")
        state = allowed.get(tok, step_token(state, VOCAB[tok]))
        if state is None:                              # only reachable when broken
            return "".join(pieces), None
        if state in ACCEPT:
            break
    return "".join(pieces), state


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_1_the_retry_baseline():
    """Ask nicely, parse, retry on failure. Everyone's first answer."""
    line("DEMO 1: the retry loop, priced")

    print("A request costs 400 tokens. You retry until the JSON parses.")
    print("p = probability one attempt returns valid JSON.\n")
    print(f"{'p':>6} {'E[attempts]':>12} {'p95 attempts':>13} "
          f"{'p99.9':>7} {'tokens/1k reqs':>15} {'worst case':>11}")

    rng = np.random.default_rng(0)
    for p in (0.70, 0.90, 0.95, 0.99):
        # Geometric distribution. Mean 1/p; the k-th quantile is the smallest
        # n with 1-(1-p)^n >= k, i.e. ceil(log(1-k)/log(1-p)).
        mean = 1 / p
        q95 = int(np.ceil(np.log(1 - 0.95) / np.log(1 - p)))
        q999 = int(np.ceil(np.log(1 - 0.999) / np.log(1 - p)))
        draws = rng.geometric(p, size=20000)
        print(f"{p:6.2f} {mean:12.2f} {q95:13d} {q999:7d} "
              f"{int(1000 * mean * 400):15,d} {'unbounded':>11}")
        assert abs(draws.mean() - mean) < 0.05, "simulation disagrees with theory"

    print("\nsimulated 20,000 requests at p=0.95: mean attempts "
          f"{rng.geometric(0.95, size=20000).mean():.3f}, "
          f"max {rng.geometric(0.95, size=20000).max()}")
    print("\nREAD THIS: 95% sounds like a passing grade. It means 1 request in")
    print("20 pays double, 1 in 400 pays triple, and there is no n at which you")
    print("can say 'this will have succeeded'. The distribution has no worst")
    print("case -- only a tail you have not seen yet. Constrained decoding")
    print("replaces the whole table with 1.00 attempts and no tail.")


def demo_2_the_mechanism():
    """One concrete step: the distribution before and after the mask."""
    line("DEMO 2: mask the vocabulary, then sample")

    rng = np.random.default_rng(7)
    state = 0                                   # nothing emitted yet
    logits = rng.normal(0, 2.0, V)
    allowed = allowed_tokens(state)
    masked = apply_mask(logits, allowed)

    p_before, p_after = softmax(logits), softmax(masked)
    order = np.argsort(-p_before)[:8]
    print(f"parser state {state}: the document is empty, only '{{' can start it")
    print(f"allowed tokens: {sorted(VOCAB[i] for i in allowed)}\n")
    print(f"{'token':>10} {'logit':>8} {'p before':>10} {'p after':>10}")
    for i in list(order) + [j for j in allowed if j not in order]:
        flag = "  <- the only survivors" if i in allowed else "  <- forbidden"
        print(f"{VOCAB[i]!r:>10} {logits[i]:8.3f} {p_before[i]:10.4f} "
              f"{p_after[i]:10.4f}{flag}")

    p_bad_before = sum(p_before[i] for i in range(V) if i not in allowed)
    print("\ntotal probability on invalid tokens before masking: "
          f"{p_bad_before:.4f}")
    print("total probability on invalid tokens after  masking: "
          f"{sum(p_after[i] for i in range(V) if i not in allowed):.4f}")
    print(f"probabilities still sum to 1: {p_after.sum():.6f}")

    zeroed = np.where(np.isin(np.arange(V), list(allowed)), logits, 0.0)
    p_zero = softmax(zeroed)
    leak = sum(p_zero[i] for i in range(V) if i not in allowed)
    print(f"\nsame mask written with 0.0 instead of -inf: {leak:.4f} of the")
    print("probability mass is STILL on invalid tokens.")
    print("\nREAD THIS: this is the causal mask from the attention page, pointed")
    print("at the vocabulary instead of at positions. -inf, not 0.")


def demo_3_state_machine():
    """Run the automaton and prove the output parses by construction."""
    line("DEMO 3: the automaton, step by step")

    index = build_index()
    print(f"{len(DELTA)} states in all, {len(DELTA) - len(NAMED)} of them single")
    print("steps inside a fixed literal. The five that mean something:\n")
    for s, name in NAMED.items():
        chars = sorted(DELTA[s])
        shown = "".join(chars[:12]) + ("..." if len(chars) > 12 else "")
        print(f"  state {s} {name:15s} {len(chars):2d} chars {shown!r:20s}"
              f" -> {len(index[s]):2d}/{V} tokens")

    print("\none generation, traced:")
    rng = np.random.default_rng(3)
    text, state = generate(rng, index, trace=True)
    print(f"\n  emitted: {text}")
    print(f"  final state {state}, accepting: {state in ACCEPT}")
    print(f"  json.loads says: {json.loads(text)}")

    n, fails, truncated, lengths = 500, 0, 0, []
    rng = np.random.default_rng(11)
    for _ in range(n):
        text, state = generate(rng, index)
        lengths.append(len(text))
        if state not in ACCEPT:
            truncated += 1                 # hit the step cap mid-document
            continue
        try:
            obj = json.loads(text)
            assert set(obj) == {"name", "age"}
        except Exception:
            fails += 1
    print(f"\n{n} constrained generations: {n - fails} parsed, {fails} failed, "
          f"0 retries")
    print(f"lengths {min(lengths)}-{max(lengths)} chars, all schema-conformant, "
          f"{truncated} cut off by the step cap")
    print("\nREAD THIS: not 'usually parses'. The parser cannot leave the set of")
    print("valid prefixes, because the tokens that would take it out were never")
    print("sampleable. Validation after the fact is a test. This is a type.")


def demo_4_the_tokeniser_problem():
    """The genuinely hard part: grammars are characters, models emit tokens."""
    line("DEMO 4: the grammar speaks characters, the model speaks tokens")

    state = NAME                                # inside the name string
    print("parser state 1 = inside the name string. As CHARACTERS the grammar")
    print(f"allows {len(DELTA[NAME])} of them here: the letters, space, and the")
    print("closing quote. Now try whole tokens.\n")
    print(f"{'token':>10} {'first char ok?':>15} {'whole token ok?':>16}  why")
    for tok in ('"}', '","', "lovelace", "ace", "}", "age", '":"'):
        first = step_char(state, tok[0]) is not None
        ok = step_token(state, tok)
        why = f"ends in state {ok}" if ok is not None else "walks off the DFA"
        print(f"{tok!r:>10} {str(first):>15} {str(ok is not None):>16}  {why}")

    print("\nThe interesting one is '\"}': the quote is legal here, it closes")
    print("the name string. The '}' glued to it is not -- the schema still owes")
    print("us an age field. A character-level check on the first character says")
    print("yes. The token as a unit is a no. And you cannot emit half a token.")

    print("\nThe same token flips legality with the state. Token '0}':")
    for s, name in NAMED.items():
        print(f"  from state {s} ({name:15s}): {step_token(s, '0}')}")

    index = build_index()
    print("\nSo the answer is an index -- parser state -> allowed token ids:")
    for s in NAMED:
        toks = sorted(VOCAB[i] for i in index[s])
        head = toks[:6] + (["..."] if len(toks) > 6 else [])
        print(f"  state {s} ({NAMED[s]:15s}) {len(index[s]):2d} tokens  {head}")
    print(f"  ... and the other {len(DELTA) - len(NAMED)} literal states, "
          "1 or 2 tokens each")
    print("\nREAD THIS: that table IS the Outlines paper's contribution. Not the")
    print("masking -- people were masking already. The claim is that you can")
    print("precompute it once per (grammar, tokeniser) pair. See ../tokenization/")
    print("for why a vocabulary is full of strings like '\"}' to begin with.")


def demo_5_cost():
    """Naive scan versus the precomputed index, counted rather than timed.

    I count DFA character-steps instead of milliseconds so two runs of this
    script print the same numbers. It is also the more honest unit: it is the
    work, not my laptop's mood."""
    line("DEMO 5: what the index buys you")

    def scan_work(state):
        """Character-steps to find the allowed set by scanning the vocabulary."""
        steps = 0
        for tok in VOCAB:
            s = state
            for ch in tok:
                steps += 1
                s = step_char(s, ch)
                if s is None:
                    break
        return steps

    index = build_index()
    build_cost = sum(scan_work(s) for s in DELTA)

    print(f"vocabulary {V} tokens, grammar {len(DELTA)} states\n")
    print(f"{'parser state':>14} {'naive scan':>12} {'index lookup':>14}")
    for s, name in NAMED.items():
        print(f"{name:>14} {scan_work(s):9d} steps {1:9d} lookup")

    print(f"\nindex built once, at load time: {build_cost} character-steps, "
          f"{sum(len(v) for v in index.values())} entries")

    print("\nand the scan grows with the vocabulary while the lookup does not:")
    global VOCAB
    original = VOCAB
    for mult in (1, 4, 16, 64):
        VOCAB = original + [f"zz{i}" for i in range(len(original) * (mult - 1))]
        print(f"  vocab {len(VOCAB):5d}: naive {scan_work(NAME):7d} steps "
              f"per generated token, index 1")
    VOCAB = original

    print("\nREAD THIS: a real vocabulary is 50k-200k tokens and a real grammar")
    print("has far more states than five, so that first column is the difference")
    print("between shipping this and not. The paper's claim is exactly this")
    print("shape: move the scan to load time and the per-token cost of finding")
    print("the allowed set stops depending on the size of the vocabulary or the")
    print("grammar. That is the whole word 'Efficient' in the title.")


def demo_6_valid_and_wrong():
    """Syntax is not semantics."""
    line("DEMO 6: guaranteed well-formed, guaranteed nothing else")

    truth = {"name": "ada lovelace", "age": 36}
    index = build_index()
    rng = np.random.default_rng(5)

    print(f"ground truth: {truth}\n")
    print(f"{'output':>34} {'parses':>7} {'schema':>7} {'correct':>8}")
    seen = set()
    for _ in range(400):
        text, state = generate(rng, index)
        if state not in ACCEPT or text in seen or len(text) > 34:
            continue          # short ones only, so the table stays readable
        seen.add(text)
        obj = json.loads(text)
        print(f"{text:>34} {'yes':>7} {'yes':>7} "
              f"{('yes' if obj == truth else 'NO'):>8}")
        if len(seen) == 6:
            break

    print(f"\n{len(seen)} distinct outputs, all valid JSON, all matching the")
    print("schema. The number that were right about Ada Lovelace: "
          f"{sum(1 for t in seen if json.loads(t) == truth)}")
    print("\nREAD THIS: the constraint is over the language of the output, not")
    print("over the world. Schema validity is a syntax property. Whether the age")
    print("is 36 is a different question, and you measure it with an eval set --")
    print("see ../evals/. Constraining does not make the model more correct, and")
    print("it can shift the output distribution away from what the unconstrained")
    print("model would have said, which is a real and sometimes-argued cost.")


def demo_7_the_breaks():
    """The failures, side by side with the working version."""
    line("DEMO 7: four ways to get this wrong")

    index = build_index()

    print("(a) mask with 0.0 instead of -inf, 200 generations:")
    rng = np.random.default_rng(21)
    bad = 0
    for _ in range(200):
        text, state = generate(rng, index, mask_with_zero=True)
        if state not in ACCEPT:
            bad += 1
    print(f"    invalid outputs: {bad}/200")
    rng = np.random.default_rng(21)
    bad = sum(1 for _ in range(200)
              if generate(rng, index)[1] not in ACCEPT)
    print(f"    with -inf:       {bad}/200")

    print("\n(b) apply the grammar to tokens character-by-character instead of")
    print("    running the whole token through the DFA:")
    state = 1
    for tok in ('"}', "ace"):
        naive_ok = all(step_char(state, ch) is not None or ch in DELTA[state]
                       for ch in tok[:1])
        print(f"    token {tok!r:>6}: first-char check says {naive_ok}, "
              f"whole-token check says {step_token(state, tok) is not None}")
    print("    '\"}' is accepted by the sloppy check and produces "
          '{"name":"ada"} --')
    print("    valid JSON, wrong schema, no age field.")

    print("\n(c) reject after sampling instead of masking before it:")
    rng = np.random.default_rng(31)
    counts = []
    for _ in range(300):
        tries, state = 0, START
        while True:
            tries += 1
            tok = sample(rng.normal(0, 2.0, V), rng)   # unconstrained sample
            if step_token(state, VOCAB[tok]) is not None:
                break
        counts.append(tries)
    counts = np.array(counts)
    print(f"    samples needed for ONE legal first token: mean "
          f"{counts.mean():.1f}, max {counts.max()}")
    print("    with the mask applied first:              1, every time")
    print("    Rejection sampling is what 'validate and retry' is, one token")
    print("    at a time, and it pays the model for every rejected draw.")

    print("\n(d) no attempt cap on the retry loop:")
    rng = np.random.default_rng(41)
    draws = rng.geometric(0.9, size=100000)
    print(f"    p=0.90, 100,000 requests: mean {draws.mean():.3f} attempts, "
          f"max {draws.max()}")
    print(f"    that longest request cost {int(draws.max()) * 400:,} tokens "
          "and 1 timeout")


if __name__ == "__main__":
    demo_1_the_retry_baseline()
    demo_2_the_mechanism()
    demo_3_state_machine()
    demo_4_the_tokeniser_problem()
    demo_5_cost()
    demo_6_valid_and_wrong()
    demo_7_the_breaks()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. the model gives you logits over the vocabulary. that is all it gives you
  2. a parser state says which tokens could continue a valid document
  3. set the rest to -inf and sample. same mask as attention, over the vocab
  4. the parser is a finite-state machine when the schema is regular; JSON
     with arbitrary nesting is context-free, so that wants a pushdown stack
  5. the hard part is that grammars are over characters and models emit
     tokens, and one token can straddle a grammar boundary
  6. so precompute state -> allowed token ids once per (grammar, tokeniser).
     that index is the Outlines paper. per-token cost becomes a dict lookup
  7. what you get is a syntax guarantee. not a correctness one

  Everything else -- schema-to-regex compilation, the union of many grammars,
  pushdown automata for nesting -- is engineering on top of those two lines.
""")
