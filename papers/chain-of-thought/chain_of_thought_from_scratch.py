"""
Chain-of-Thought (Wei et al., 2022) and ReAct (Yao et al., 2022)
-- for programmers, not researchers.

Run it:      python3 chain_of_thought_from_scratch.py
Debug it:    breakpoint in react_loop() and watch the scratchpad grow.

HONESTY NOTE, read this before anything else:

    THERE IS NO LANGUAGE MODEL IN THIS FILE.

Nothing here is evidence about what GPT-4 or Claude does. What this script
demonstrates is the MECHANISM the two papers exploit:

  * a fixed computation budget per emitted token, and what that forbids;
  * the shape of a chain-of-thought prompt versus a standard one;
  * why a plurality vote over independent samples beats a single sample
    (this one IS a real, correct statistical demonstration);
  * the thought -> action -> observation -> repeat state machine, with a
    real calculator and a real table lookup that the script actually calls;
  * what breaks when you drop the observation, parse the answer loosely,
    correlate the samples, or forget the step limit.

Where a component stands in for a model, it says so on the line above it.

No torch. NumPy and plain Python.
"""

import re
import numpy as np

np.random.seed(0)


def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ---------------------------------------------------------------------------
# PART 1 -- the fixed serial budget.
#
# A transformer does the SAME amount of work for every token it emits: one
# forward pass, a fixed number of layers. It cannot loop internally. So the
# only way to get more serial steps is to emit more tokens, because each
# emitted token is a fresh forward pass that can read the previous ones.
#
# The evaluator below is an ANALOGY for that, not a measurement of it. It is
# allowed exactly one operation per token it emits. Nothing more.
# ---------------------------------------------------------------------------
# A "problem": start from a number, then apply ops strictly in order.
# Each op needs the result of the one before it. That is what serial means.
PROBLEM = [("start", 7), ("mul", 3), ("add", 11), ("sub", 5),
           ("mul", 2), ("add", 9)]


def apply_op(state, op):
    """One operation. This is the 'one unit of work per token' budget."""
    kind, arg = op
    if kind == "start":
        return arg
    if kind == "mul":
        return state * arg
    if kind == "add":
        return state + arg
    if kind == "sub":
        return state - arg
    raise ValueError(kind)


def solve_with_token_budget(problem, budget):
    """Emit at most `budget` tokens; one op each; answer with what you have.

    Returns (answer, tokens_emitted, steps_needed, scratchpad).
    The scratchpad IS the emitted tokens -- it is the only memory there is.
    """
    steps_needed = len(problem)
    scratch = []
    state = None
    for op in problem[:budget]:
        state = apply_op(state, op)
        scratch.append(f"{op[0]} {op[1]} -> {state}")
    return state, min(budget, steps_needed), steps_needed, scratch


# ---------------------------------------------------------------------------
# PART 2 -- the two prompt formats.
#
# This is the entire difference the chain-of-thought paper introduces. Not a
# new architecture, not new weights, not a new loss. The exemplars in the
# prompt show their working, so the continuation shows its working too.
# ---------------------------------------------------------------------------
EXEMPLARS = [
    {
        "q": "A shelf holds 4 boxes. Each box holds 6 mugs. 5 mugs break. "
             "How many are left?",
        "working": "4 boxes * 6 mugs = 24 mugs. 24 - 5 broken = 19.",
        "a": "19",
    },
    {
        "q": "A van does 3 trips a day carrying 12 crates. It runs 4 days. "
             "How many crates?",
        "working": "3 trips * 12 crates = 36 a day. 36 * 4 days = 144.",
        "a": "144",
    },
]

QUESTION = ("A cafe brews 9 pots a day, 8 cups per pot. It spills 14 cups. "
            "How many cups are served?")


def build_prompt(exemplars, question, show_working):
    """Standard few-shot when show_working=False; chain-of-thought when True."""
    parts = []
    for ex in exemplars:
        if show_working:
            parts.append(f"Q: {ex['q']}\nA: {ex['working']} "
                         f"The answer is {ex['a']}.")
        else:
            parts.append(f"Q: {ex['q']}\nA: The answer is {ex['a']}.")
    parts.append(f"Q: {question}\nA:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# PART 3 -- answer parsing. The most common real bug in a CoT pipeline.
#
# Once the model shows its working, the working is full of numbers. A loose
# regex grabs the first one it sees, which is an intermediate result.
# ---------------------------------------------------------------------------
def parse_loose(text):
    """First number anywhere in the text. Looks fine. Is wrong."""
    m = re.search(r"(-?\d+)", text)
    return m.group(1) if m else None


def parse_strict(text):
    """Anchored on the answer phrase, last occurrence. Boring. Correct."""
    hits = re.findall(r"[Tt]he answer is\s*(-?\d+)", text)
    return hits[-1] if hits else None


# ---------------------------------------------------------------------------
# PART 4 -- self-consistency (Wang et al., 2022).
#
# This part is a REAL statistical demonstration and the numbers mean what
# they say. The solver below is a stochastic multi-step process: each step is
# right with probability p, and when a step goes wrong it goes wrong in a
# scattered way. That is the only assumption self-consistency needs.
#
# Correct paths all land on the same answer. Wrong paths disagree with each
# other. So the mode of many samples beats one sample.
# ---------------------------------------------------------------------------
STEP_CORRECT_P = 0.80        # each step is right 4 times out of 5
ERROR_OFFSETS = np.array([-7, -3, -2, -1, 1, 2, 3, 4, 6, 8, 11, 13])


def sample_path(problem, rng):
    """One independent reasoning path. Returns its final answer."""
    state = None
    for op in problem:
        state = apply_op(state, op)
        if rng.random() > STEP_CORRECT_P:
            # a slip: the step is carried out wrong, and the error carries on
            state = state + int(rng.choice(ERROR_OFFSETS))
    return state


def plurality_vote(answers):
    """Majority vote over final answers -- ties broken by first seen."""
    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], -answers.index(kv[0])))[0]


def truth(problem):
    """The answer with no slips at all."""
    state = None
    for op in problem:
        state = apply_op(state, op)
    return state


# ---------------------------------------------------------------------------
# PART 5 -- the tools ReAct calls. These are real. The script runs them.
# ---------------------------------------------------------------------------
FLEET = {           # a tiny "external environment", written here in the file
    "northwind": {"vans": 34, "depot": "Leeds"},
    "contoso":   {"vans": 21, "depot": "Bristol"},
    "fabrikam":  {"vans": 58, "depot": "Hull"},
}


def tool_lookup(company, field):
    """Real lookup over a real table. Returns None when it does not know."""
    row = FLEET.get(company.lower())
    if row is None or field not in row:
        return None
    return row[field]


def tool_calc(expression):
    """Real calculator. Digits and + - * / ( ) only, so it stays a calculator."""
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return None
    return eval(expression, {"__builtins__": {}}, {})


# ---------------------------------------------------------------------------
# PART 6 -- the ReAct loop as a state machine.
#
# thought -> action -> observation -> thought -> ... -> answer
#
# `policy` stands in for the model. It is a hand-written controller: it reads
# the scratchpad and returns the next action. A real system would sample this
# from a language model. The LOOP is the paper's contribution and the loop
# here is exactly the real thing; the policy is not.
# ---------------------------------------------------------------------------
def react_loop(policy, question, max_steps=8, feed_observations=True,
               verbose=True):
    """Run thought/action/observation until the policy answers or we cut it off.

    max_steps=None means no guard. Nothing stops it. That is the point of
    demo 7, and it is why the caller there passes a watchdog instead.
    """
    scratch = []          # the ONLY thing the policy gets to condition on
    steps = 0
    while max_steps is None or steps < max_steps:
        steps += 1
        thought, action = policy(question, scratch)
        if verbose:
            print(f"  Thought {steps}: {thought}")
        if action[0] == "finish":
            if verbose:
                print(f"  Answer: {action[1]}")
            return action[1], steps, scratch

        if verbose:
            args = ", ".join(repr(x) for x in action[1:])
            print(f"  Action  {steps}: {action[0]}({args})")
        obs = run_action(action)
        if verbose:
            print(f"  Obs     {steps}: {obs}")

        # THE line that matters. With feed_observations=False the fact the
        # tool just fetched never reaches the scratchpad, so the next thought
        # is conditioned on a tool result the policy never saw. (Arithmetic
        # results still come back -- it is the environment fact we are
        # withholding, which is the one ReAct is about.)
        if feed_observations or action[0] == "calc":
            scratch.append((action, obs))
        else:
            scratch.append((action, None))
    return None, steps, scratch


def run_action(action):
    kind = action[0]
    if kind == "lookup":
        return tool_lookup(action[1], action[2])
    if kind == "calc":
        return tool_calc(action[1])
    return None


# The policy: a lookup table over "what have I observed so far".
# Stands in for a model. Deterministic, so the transcript is reproducible.
GUESSES = {"northwind": 40, "contoso": 25}   # plausible, memorised, wrong


def fleet_policy(question, scratch):
    """Answer 'how many more vans does Northwind have than Contoso?'"""
    seen = {a[1]: obs for a, obs in scratch if a[0] == "lookup"}
    calcs = [obs for a, obs in scratch if a[0] == "calc"]

    if "northwind" not in seen:
        return ("I need Northwind's van count. I should look it up, not "
                "recall it.", ("lookup", "northwind", "vans"))
    if "contoso" not in seen:
        return ("Got Northwind. Now Contoso.",
                ("lookup", "contoso", "vans"))
    if not calcs:
        # If the observation was not fed back, `seen[...]` is None and the
        # policy falls back to what it "remembers". That fallback is a
        # hardcoded number standing in for a guess -- it is not a model.
        n = seen["northwind"] if seen["northwind"] is not None \
            else GUESSES["northwind"]
        c = seen["contoso"] if seen["contoso"] is not None \
            else GUESSES["contoso"]
        return (f"Now subtract: {n} minus {c}.", ("calc", f"{n} - {c}"))
    return (f"The difference is {calcs[-1]}.", ("finish", calcs[-1]))


def missing_key_policy(question, scratch):
    """Asks for a field the table does not have, over and over."""
    return ("The table must have the driver count somewhere. Try again.",
            ("lookup", "northwind", "drivers"))


# ===========================================================================
# DEMOS
# ===========================================================================
def demo_1_serial_budget():
    line("DEMO 1: a fixed budget per token is a hard ceiling (an ANALOGY)")

    print("This evaluator may do ONE operation per token it emits. That is a")
    print("stand-in for a fixed-depth network doing one forward pass per")
    print("token. It is an analogy for the depth limit, NOT a measurement of")
    print("any real model.\n")

    correct = truth(PROBLEM)
    for budget in (1, 3, 6):
        ans, used, needed, scratch = solve_with_token_budget(PROBLEM, budget)
        tag = "correct" if ans == correct else "WRONG"
        print(f"token budget {budget}:  steps needed {needed}, "
              f"steps available {used}  ->  answer {ans}  ({tag})")

    print("\nthe working, when it is allowed to exist:")
    _, _, _, scratch = solve_with_token_budget(PROBLEM, 6)
    for i, s in enumerate(scratch, 1):
        print(f"   token {i}: {s}")

    print(f"\ntrue answer: {correct}")
    print("\nREAD THIS: the problem needs 6 serial steps. Asked to answer")
    print("immediately, the evaluator has 1 step of budget and returns a")
    print("partial result. It is not lazy and it is not under-informed -- it")
    print("ran out of serial depth. Chain-of-thought is not a pep talk. It is")
    print("giving the computation somewhere to happen.")


def demo_2_prompt_formats():
    line("DEMO 2: the only difference between the two prompts")

    print("--- STANDARD few-shot -------------------------------------------")
    print(build_prompt(EXEMPLARS, QUESTION, show_working=False))
    print("\n--- CHAIN-OF-THOUGHT few-shot -----------------------------------")
    print(build_prompt(EXEMPLARS, QUESTION, show_working=True))

    a = build_prompt(EXEMPLARS, QUESTION, show_working=False)
    b = build_prompt(EXEMPLARS, QUESTION, show_working=True)
    print(f"\nprompt lengths: standard {len(a)} chars, CoT {len(b)} chars")
    print("identical except the exemplar answers show their working?",
          all(ln in b for ln in a.split("\n") if ln.startswith("Q:")))

    print("\nREAD THIS: no new weights, no new loss, no new architecture. The")
    print("exemplars show their working, so the continuation does too. Wei et")
    print("al. report that this only helps at sufficient model scale and can")
    print("HURT smaller models -- that is their emergence result, and nothing")
    print("in this script demonstrates it. Kojima et al. (2022) later showed")
    print("you can often skip the exemplars entirely and append 'Let's think")
    print("step by step'.")


def demo_3_answer_parsing():
    line("DEMO 3: the loose regex bug that eats CoT pipelines")

    trace = ("9 pots * 8 cups = 72 cups. 72 - 14 spilled = 58. "
             "The answer is 58.")
    print("model output:\n  " + trace + "\n")
    print(f"parse_loose  -> {parse_loose(trace)}   <-- first number in the "
          "working")
    print(f"parse_strict -> {parse_strict(trace)}   <-- anchored on 'the "
          "answer is'")

    print("\nREAD THIS: the standard prompt's answer contains exactly one")
    print("number, so a loose regex works and nobody notices. Turn on chain-")
    print("of-thought and the same regex silently starts returning an")
    print("intermediate result. Your accuracy drops and the prompt gets the")
    print("blame.")


def demo_4_self_consistency():
    line("DEMO 4: self-consistency -- vote over independent paths")

    correct = truth(PROBLEM)
    trials = 4000
    rng = np.random.default_rng(0)

    print(f"per-step accuracy {STEP_CORRECT_P}, {len(PROBLEM)} steps, "
          f"{trials} trials")
    print(f"true answer {correct}\n")
    print("  paths (k)   accuracy of the vote")
    for k in (1, 3, 5, 11, 21, 41):
        wins = 0
        for _ in range(trials):
            paths = [sample_path(PROBLEM, rng) for _ in range(k)]
            if plurality_vote(paths) == correct:
                wins += 1
        print(f"  {k:9d}   {wins / trials:.3f}")

    print("\none sample of 11 paths, so you can see the shape of it:")
    rng2 = np.random.default_rng(7)
    paths = [sample_path(PROBLEM, rng2) for _ in range(11)]
    print("  answers:", paths)
    print("  vote   :", plurality_vote(paths), " truth:", correct)

    print("\nREAD THIS: a single path is right about a quarter of the time")
    print("-- it has to get all 6 steps right. But the right paths all agree on one")
    print("number and the wrong paths scatter, so the mode is right far more")
    print("often than any single path. Gains flatten because you are")
    print("converging on the distribution's mode, not adding new information.")


def demo_5_correlated_votes():
    line("DEMO 5: the break -- vote over paths that are not independent")

    correct = truth(PROBLEM)
    trials = 4000
    k = 21

    indep = np.random.default_rng(0)
    wins = 0
    for _ in range(trials):
        paths = [sample_path(PROBLEM, indep) for _ in range(k)]
        wins += plurality_vote(paths) == correct
    print(f"k={k}, independent paths        : {wins / trials:.3f}")

    # The break: every path in a trial is drawn from the SAME seed, which is
    # what temperature 0 or a shared cached prefix gets you.
    corr = np.random.default_rng(0)
    wins = 0
    for _ in range(trials):
        seed = int(corr.integers(1 << 30))
        paths = [sample_path(PROBLEM, np.random.default_rng(seed))
                 for _ in range(k)]
        wins += plurality_vote(paths) == correct
    print(f"k={k}, 21 copies of ONE path    : {wins / trials:.3f}")

    print("\nREAD THIS: same vote, same k, no benefit whatsoever. Voting buys")
    print("you nothing unless the errors are independent. Sampling at")
    print("temperature 0, or reusing one cached trace, gets you this row.")


def demo_6_react_loop():
    line("DEMO 6: the ReAct loop -- two lookups and one calculation")

    q = "How many more vans does Northwind have than Contoso?"
    print("Question:", q, "\n")
    ans, steps, _ = react_loop(fleet_policy, q, max_steps=8)
    print(f"\nfinished in {steps} loop iterations, answer = {ans}")
    print(f"ground truth = {FLEET['northwind']['vans'] - FLEET['contoso']['vans']}")

    print("\nREAD THIS: thought, action, observation, repeat. The observations")
    print("are real -- the script called tool_lookup and tool_calc and those")
    print("touched the actual table and the actual calculator. The policy is a")
    print("hand-written state machine standing in for a model; the loop is the")
    print("real thing.")


def demo_7_grounding():
    line("DEMO 7: the break -- run the loop without feeding observations back")

    q = "How many more vans does Northwind have than Contoso?"
    truth_ = FLEET["northwind"]["vans"] - FLEET["contoso"]["vans"]

    grounded, _, _ = react_loop(fleet_policy, q, verbose=False)
    ungrounded, _, _ = react_loop(fleet_policy, q, feed_observations=False,
                                  verbose=False)

    print("with observations fed back   :", grounded, " <- checked")
    print("with observations discarded  :", ungrounded, " <- guessed")
    print("ground truth                 :", truth_)
    print("\nthe ungrounded transcript, so you can see where it goes wrong:")
    react_loop(fleet_policy, q, feed_observations=False, verbose=True)

    print("\nREAD THIS: the tool was called both times. Both transcripts")
    print("contain a real observation line. The only difference is whether")
    print("that line went back into the scratchpad the policy reads. Drop it")
    print("and the loop falls through to a hardcoded plausible number -- which")
    print("here stands in for a memorised guess. This is the hallucination")
    print("ReAct is arranged to prevent: not by knowing more, but by making")
    print("the next step read a fact it just fetched.")


def demo_8_loop_control():
    line("DEMO 8: no step limit, and the guard that saves you")

    q = "How many drivers does Northwind employ?"
    print("The table has no 'drivers' field, so the tool returns None forever.")
    print("The policy never gives up. Watch.\n")

    # No guard. We run it under an external watchdog so this demo terminates;
    # in your service there is no watchdog, there is just a bill.
    watchdog = 200
    steps = 0
    scratch = []
    while steps < watchdog:
        steps += 1
        _, action = missing_key_policy(q, scratch)
        scratch.append((action, run_action(action)))
    print(f"max_steps=None: still looping after {steps} iterations, "
          "no answer, no end")
    print(f"                tool calls made: {len(scratch)}, "
          f"distinct observations: {len(set(o for _, o in scratch))}")

    ans, steps, _ = react_loop(missing_key_policy, q, max_steps=5,
                               verbose=False)
    print(f"max_steps=5   : guard fired at step {steps}, returned {ans}")

    print("\nREAD THIS: an agent loop with no step limit is a while-True with a")
    print("credit card attached. The guard is three lines and it is the")
    print("difference between a bad answer and a bad invoice. Return None and")
    print("let the caller decide -- do not let the loop decide for itself.")


if __name__ == "__main__":
    demo_1_serial_budget()
    demo_2_prompt_formats()
    demo_3_answer_parsing()
    demo_4_self_consistency()
    demo_5_correlated_votes()
    demo_6_react_loop()
    demo_7_grounding()
    demo_8_loop_control()

    line("THE WHOLE THING, COMPRESSED")
    print("""
  1. a transformer spends a FIXED amount of compute per emitted token
  2. so a problem needing more serial steps than that cannot fit in one token
  3. chain-of-thought = emit the intermediate steps, giving the computation
     somewhere to happen. Each token is another forward pass that reads the
     last one. A pure function versus one with a scratch buffer.
  4. Wei et al.: the gain shows up only at sufficient scale, and can hurt
     small models. That is the paper's headline, and it is about emergence.
  5. Kojima et al.: often you can drop the exemplars and say
     "Let's think step by step".
  6. self-consistency = sample several independent paths, take the majority
     answer. Works because right answers agree and wrong ones scatter.
  7. ReAct = let the scratch buffer contain the result of a real tool call.
     thought -> action -> observation -> repeat. Fewer invented facts,
     because the next step reads something that was fetched, not recalled.
  8. always put a step limit on the loop.
""")
