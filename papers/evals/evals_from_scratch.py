"""
Evaluation -- for programmers, not researchers.

Run it:      python3 evals_from_scratch.py
Debug it:    put a breakpoint in judged_accuracy and watch the gap shrink.

NumPy only. No torch, no autograd, no model.

THERE IS NO LANGUAGE MODEL IN HERE. Every "system", every "judge", every
"grader" is an explicit probability model written out in about five lines.
That is not a shortcut, it is the whole point: you cannot measure the
statistical behaviour of an evaluation unless you already know the truth it
is trying to recover. Here we set the truth, then watch the eval get it
wrong. Nothing printed below is evidence about any real model.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# STAGE 1 -- a test case is a coin flip.
#
# A system with true accuracy p, run on n test cases, produces n Bernoulli
# draws. That is the entire statistical model of your golden set. Everything
# else in this file is a consequence of it.
# ---------------------------------------------------------------------------
def run_eval(p, n, rng):
    """Return the pass/fail vector for one system on n cases."""
    return (rng.random(n) < p).astype(int)


# ---------------------------------------------------------------------------
# STAGE 2 -- the confidence interval on a DIFFERENCE of two accuracies.
#
# Standard error of one proportion: sqrt(p(1-p)/n).
# Two independent systems: add the variances, then sqrt.
# The 95% interval is +/- 1.96 of that. If it straddles 0, your eval cannot
# tell the two systems apart, no matter how confident the dashboard looks.
# ---------------------------------------------------------------------------
def diff_ci(a, b):
    """95% CI on (mean(a) - mean(b)) for two independent pass/fail vectors."""
    pa, pb = a.mean(), b.mean()
    se = np.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))
    d = pa - pb
    return d, d - 1.96 * se, d + 1.96 * se


# ---------------------------------------------------------------------------
# STAGE 3 -- a judge that is right some fraction of the time.
#
# The judge sees the true verdict and repeats it with probability `agree`.
# Otherwise it flips it. That is the simplest honest model of an imperfect
# grader, and it has a closed form:
#
#     measured = agree * p + (1 - agree) * (1 - p)
#
# Rearranged: measured_gap = (2 * agree - 1) * true_gap.
# A judge that agrees 80% of the time multiplies every gap you measure by
# 0.6. Your real 4-point improvement is reported as 2.4.
# ---------------------------------------------------------------------------
def judged_accuracy(truth, agree, rng):
    """Grade a pass/fail vector through a judge with agreement rate `agree`."""
    keep = rng.random(len(truth)) < agree
    return np.where(keep, truth, 1 - truth).mean()


# ---------------------------------------------------------------------------
# STAGE 4 -- a pairwise judge with three biases, each a separate knob.
#
# quality  : the two answers' true quality, in [0, 1]
# pos      : how much extra the FIRST-shown answer wins by
# length   : how much extra the LONGER answer wins by
# self_pref: how much extra an answer from the judge's own family wins by
#
# The judge adds the biases to the quality difference and flips a coin
# weighted by the result. Bias here is an additive nudge on a probability,
# which is the crudest possible model and still enough to wreck a leaderboard.
# ---------------------------------------------------------------------------
def pairwise_judge(qa, qb, len_a, len_b, own_a, own_b,
                   pos=0.0, length=0.0, self_pref=0.0, rng=RNG):
    """P(judge picks A) -> a 0/1 vote. A is always shown first."""
    p = 0.5 + 0.5 * (qa - qb)          # quality: a fair judge stops here
    p += pos                            # position: A is on top, A gets a bump
    p += length * np.sign(len_a - len_b)
    p += self_pref * (own_a - own_b)
    return (rng.random() < np.clip(p, 0.0, 1.0)) * 1


# ---------------------------------------------------------------------------
# STAGE 5 -- the swap mitigation. Ask twice, positions reversed, average.
#
# If position bias is a constant +b for whoever is first, then asking in both
# orders and averaging cancels it exactly. This is the same trick as running
# an A/B test in both directions to cancel a rendering-order effect.
# ---------------------------------------------------------------------------
def swapped_win_rate(n, qa, qb, len_a, len_b, own_a, own_b,
                     pos, length, self_pref, rng):
    """A's win rate, averaged over both presentation orders."""
    wins = 0.0
    for _ in range(n):
        v1 = pairwise_judge(qa, qb, len_a, len_b, own_a, own_b,
                            pos, length, self_pref, rng)          # A first
        v2 = pairwise_judge(qb, qa, len_b, len_a, own_b, own_a,
                            pos, length, self_pref, rng)          # B first
        wins += 0.5 * (v1 + (1 - v2))
    return wins / n


# ---------------------------------------------------------------------------
# STAGE 6 -- contamination. A leaked item is not a test, it is a lookup.
#
# `leak` is the fraction of benchmark items that appeared in training. On
# those the model scores `memorised` (near 1.0, it has seen the answer). On
# the rest it scores its true held-out ability. The reported number is a
# blend; the truth is only the second term.
# ---------------------------------------------------------------------------
def contaminated_score(true_ability, leak, memorised=0.98):
    """The score a contaminated benchmark reports for a given true ability."""
    return leak * memorised + (1 - leak) * true_ability


# ---------------------------------------------------------------------------
# STAGE 7 -- absolute rating vs pairwise comparison, on the same quality.
#
# An absolute grader needs a SCALE, and a scale needs an anchor it does not
# have. So it carries a per-session offset: today everything is generous,
# next week everything is stern. You rate system A in one session and system
# B in another, so that offset does NOT cancel, and it does not average away
# with more items either -- it is shared by every item in the session.
#
# A pairwise grader sees both answers in the SAME session and only has to say
# which is better. The offset is on both sides of the comparison and cancels
# exactly. That, not "less noise", is why comparisons are more stable.
# ---------------------------------------------------------------------------
def absolute_rating(q, offset, noise, rng):
    """Quality in [0,1] -> a 1-5 rating, with a session offset plus noise."""
    raw = 1 + 4 * q + offset + rng.normal(0, noise)
    return float(np.clip(np.round(raw), 1, 5))


def pairwise_vote(qa, qb, offset, noise, rng):
    """Same grader, same session, easier question: which of these is better?"""
    a = 1 + 4 * qa + offset + rng.normal(0, noise)
    b = 1 + 4 * qb + offset + rng.normal(0, noise)   # offset cancels here
    return int(a > b)


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def demo_1_sample_size():
    """How many test cases before a difference means anything."""
    line("DEMO 1: your golden set of 20 cases cannot tell you anything")

    p_a, p_b = 0.72, 0.78          # B is truly 6 points better. We set this.
    trials = 4000
    rng = np.random.default_rng(1)

    print(f"true accuracy:  system A = {p_a:.2f}   system B = {p_b:.2f}"
          f"   true gap = {p_b - p_a:+.2f}\n")
    print(" n      mean measured   typical 95% CI        picks the")
    print("        gap             on the gap            WORSE system")
    print(" " + "-" * 62)

    for n in (20, 50, 200, 1000):
        wrong = 0
        gaps = []
        for _ in range(trials):
            a = run_eval(p_a, n, rng)
            b = run_eval(p_b, n, rng)
            d, _, _ = diff_ci(b, a)
            gaps.append(d)
            if d < 0:
                wrong += 1
        # The interval you would expect to see, computed at the TRUE rates.
        se = np.sqrt(p_a * (1 - p_a) / n + p_b * (1 - p_b) / n)
        d = p_b - p_a
        lo, hi = d - 1.96 * se, d + 1.96 * se
        print(f"{n:5d}   {np.mean(gaps):+8.3f}       [{lo:+.3f}, {hi:+.3f}]"
              f"        {wrong / trials:6.1%}")

    print("\nREAD THIS: at n=20 the interval is wider than the effect, and one")
    print("run in four crowns the WORSE system. At n=1000 the interval finally")
    print("excludes zero. Nobody computes this. Everybody ships on n=20.")


def demo_2_judge_agreement():
    """An imperfect judge compresses every gap you measure."""
    line("DEMO 2: a judge that is right 80% of the time shrinks your gap")

    p_a, p_b = 0.60, 0.80
    n = 20000
    rng = np.random.default_rng(2)
    truth_a = run_eval(p_a, n, rng)
    truth_b = run_eval(p_b, n, rng)
    true_gap = truth_b.mean() - truth_a.mean()

    print(f"true gap between A and B, graded by a perfect human: {true_gap:.4f}\n")
    print(" judge      measured    measured    measured   gap kept")
    print(" agreement  acc(A)      acc(B)      gap")
    print(" " + "-" * 62)

    for agree in (1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50):
        ma = judged_accuracy(truth_a, agree, rng)
        mb = judged_accuracy(truth_b, agree, rng)
        gap = mb - ma
        print(f"  {agree:.2f}      {ma:.4f}      {mb:.4f}      "
              f"{gap:+.4f}     {gap / true_gap:6.1%}")

    print("\nREAD THIS: the closed form is measured_gap = (2*agree - 1)*true_gap.")
    print("At 80% agreement you keep about 60% of the real difference. Your")
    print("model got 20 points better; the eval reports 12. At 50% agreement")
    print("the judge is a coin and every system on the leaderboard ties.")


def demo_3_judge_biases():
    """Position, length and self-preference, measured and then mitigated."""
    line("DEMO 3: the three biases, each one measured, then mitigated")

    n = 20000
    rng = np.random.default_rng(3)
    q = 0.50                        # IDENTICAL quality. A fair judge -> 50%.

    def naive(pos=0.0, length=0.0, self_pref=0.0, la=100, lb=100,
              oa=0, ob=0):
        w = sum(pairwise_judge(q, q, la, lb, oa, ob, pos, length,
                               self_pref, rng) for _ in range(n))
        return w / n

    print("Two answers of IDENTICAL quality. A fair judge must return 50.0%.\n")
    print(" bias                       naive     after mitigation  residual")
    print(" " + "-" * 62)

    # (a) POSITION: A is always shown first, judge favours whoever is first.
    pos_naive = naive(pos=0.10)
    pos_fixed = swapped_win_rate(n, q, q, 100, 100, 0, 0, 0.10, 0.0, 0.0, rng)
    print(f" position (+0.10 to first)  {pos_naive:6.1%}    {pos_fixed:6.1%}"
          f"            {abs(pos_fixed - 0.5):.4f}")

    # (b) LENGTH: A's answer is longer, judge favours the longer answer.
    len_naive = naive(length=0.12, la=400, lb=100)
    # Mitigation: compare only within a length bucket, so length is constant.
    len_fixed = naive(length=0.12, la=100, lb=100)
    print(f" length   (+0.12 to longer) {len_naive:6.1%}    {len_fixed:6.1%}"
          f"            {abs(len_fixed - 0.5):.4f}")

    # (c) SELF-PREFERENCE: A comes from the judge's own model family.
    self_naive = naive(self_pref=0.08, oa=1, ob=0)
    # Mitigation: two judges from different families, average their verdicts.
    j1 = naive(self_pref=0.08, oa=1, ob=0)   # judge of A's family: A gets +
    j2 = naive(self_pref=0.08, oa=0, ob=1)   # judge of B's family: B gets +
    self_fixed = 0.5 * (j1 + j2)             # panel of two, opposite leans
    print(f" self-preference (+0.08)    {self_naive:6.1%}    {self_fixed:6.1%}"
          f"            {abs(self_fixed - 0.5):.4f}")

    print("\nREAD THIS: every one of those naive numbers is a lie about equal")
    print("systems. Swapping positions and averaging cancels position bias")
    print("almost exactly. Bucketing by length removes the length effect.")
    print("A panel of judges from different families cancels self-preference.")
    print("The residual is what your mitigation did NOT remove -- report it.")


def demo_4_contamination():
    """Benchmark score rises with leakage while real ability does not move."""
    line("DEMO 4: contamination -- the score goes up, the model does not")

    true_ability = 0.55             # fixed. The model never actually improves.
    print(f"true held-out ability, held CONSTANT at {true_ability:.2f}\n")
    print(" leaked fraction   benchmark score   clean score   inflation")
    print(" " + "-" * 62)
    for leak in (0.0, 0.05, 0.15, 0.30, 0.50, 0.80):
        s = contaminated_score(true_ability, leak)
        print(f"      {leak:4.0%}            {s:.4f}           "
              f"{true_ability:.4f}       {s - true_ability:+.4f}")

    print("\nA leaderboard-topping model on a 30% contaminated benchmark:")
    weak, strong = 0.55, 0.62
    print(f"  model X: contaminated {contaminated_score(weak, 0.30):.4f}   "
          f"clean {weak:.4f}")
    print(f"  model Y: contaminated {contaminated_score(strong, 0.0):.4f}   "
          f"clean {strong:.4f}")
    print("  ranking on the benchmark: X wins. Ranking on clean data: Y wins.")

    print("\nREAD THIS: leakage is not noise, it is bias, and it only ever")
    print("points up. There is no run-it-again that averages it out. This")
    print("simulation says nothing about any real model or benchmark -- what")
    print("it shows is the SHAPE: a 30% leak buys 13 free points of nothing.")


def demo_5_pairwise_vs_absolute():
    """Why comparisons are more stable than a 1-5 score."""
    line("DEMO 5: pairwise beats a 1-to-5 score, and this is why")

    rng = np.random.default_rng(5)
    qa, qb = 0.60, 0.70             # B is truly better by 0.10
    reps, per_rep = 400, 30
    noise = 0.7                     # per-item wobble, on the 1-5 scale
    drift = 0.6                     # per-session calibration offset

    abs_gaps, pair_rates = [], []
    for _ in range(reps):
        # Absolute: A and B were rated in two different sessions.
        off_a, off_b = rng.normal(0, drift), rng.normal(0, drift)
        ra = [absolute_rating(qa, off_a, noise, rng) for _ in range(per_rep)]
        rb = [absolute_rating(qb, off_b, noise, rng) for _ in range(per_rep)]
        abs_gaps.append(np.mean(rb) - np.mean(ra))
        # Pairwise: same session, so one offset, applied to both.
        off = rng.normal(0, drift)
        v = [pairwise_vote(qb, qa, off, noise, rng) for _ in range(per_rep)]
        pair_rates.append(np.mean(v))

    abs_gaps = np.array(abs_gaps)
    pair_rates = np.array(pair_rates)
    # Same question of both methods: how often do they call the winner right?
    abs_right = (abs_gaps > 0).mean()
    pair_right = (pair_rates > 0.5).mean()
    print(f"true quality: A = {qa:.2f}, B = {qb:.2f}. "
          f"{reps} repeats of a {per_rep}-case eval.")
    print(f"grader noise {noise} per item, calibration drift {drift} "
          f"per session.\n")
    print(f"  absolute 1-5   mean gap {abs_gaps.mean():+.3f}   "
          f"std {abs_gaps.std():.3f}   picks B {abs_right:6.1%}")
    print(f"  pairwise A/B   mean rate {pair_rates.mean():.3f}   "
          f"std {pair_rates.std():.3f}   picks B {pair_right:6.1%}")
    print(f"\n  signal-to-spread: absolute "
          f"{abs(abs_gaps.mean()) / abs_gaps.std():.2f}, "
          f"pairwise {abs(pair_rates.mean() - 0.5) / pair_rates.std():.2f}")

    print("\nREAD THIS: the absolute grader has to hold a scale in its head and")
    print("the scale drifts between sessions, so the drift lands straight in")
    print("your reported gap and more test cases will not average it out. The")
    print("pairwise grader sees both answers at once, the drift cancels, and it")
    print("calls the winner right far more often. This is exactly why the reward")
    print("model in ../rlhf/ is trained from A-vs-B votes and not from 1-5 stars.")


def demo_6_regression_by_slice():
    """The aggregate goes up. One category is on fire."""
    line("DEMO 6: the aggregate rose. Ship it. (Do not ship it.)")

    rng = np.random.default_rng(6)
    cats = ["summarise", "extract", "classify", "translate", "code"]
    before = {"summarise": 0.70, "extract": 0.72, "classify": 0.75,
              "translate": 0.68, "code": 0.71}
    after = {"summarise": 0.82, "extract": 0.84, "classify": 0.86,
             "translate": 0.80, "code": 0.34}          # code collapsed
    n_per_cat = 2000

    print(" category      before    after     delta")
    print(" " + "-" * 62)
    agg_b, agg_a = [], []
    for c in cats:
        b = run_eval(before[c], n_per_cat, rng)
        a = run_eval(after[c], n_per_cat, rng)
        agg_b.append(b)
        agg_a.append(a)
        flag = "  <-- REGRESSION" if a.mean() - b.mean() < -0.05 else ""
        print(f" {c:12s}  {b.mean():.4f}   {a.mean():.4f}   "
              f"{a.mean() - b.mean():+.4f}{flag}")

    ab = np.concatenate(agg_b)
    aa = np.concatenate(agg_a)
    print(" " + "-" * 62)
    print(f" AGGREGATE     {ab.mean():.4f}   {aa.mean():.4f}   "
          f"{aa.mean() - ab.mean():+.4f}   <-- ships green")

    d, lo, hi = diff_ci(aa, ab)
    print(f"\n aggregate 95% CI on the change: [{lo:+.4f}, {hi:+.4f}] "
          f"-- 'significant improvement'")
    print("\nREAD THIS: the aggregate went UP and the interval excludes zero.")
    print("Every dashboard is green. One in five of your users just lost more")
    print("than half their success rate. The slice check is four lines and")
    print("it is the entire reason an eval harness exists.")


def demo_7_tune_and_report_on_the_same_set():
    """Pick the best of k on a set, then report on that same set."""
    line("DEMO 7: tuning and reporting on the same test set")

    rng = np.random.default_rng(7)
    n, k = 200, 20                  # 20 prompt variants, all equally good
    true_p = 0.70

    best_measured, best_holdout = [], []
    for _ in range(600):
        runs = [run_eval(true_p, n, rng) for _ in range(k)]
        scores = np.array([r.mean() for r in runs])
        winner = int(scores.argmax())
        best_measured.append(scores[winner])
        best_holdout.append(run_eval(true_p, n, rng).mean())   # fresh set

    bm, bh = np.mean(best_measured), np.mean(best_holdout)
    print(f"{k} candidate prompts, ALL with identical true accuracy "
          f"{true_p:.2f}.")
    print(f"Pick the winner on a {n}-case set, then report that set's score.\n")
    print(f"  reported score of the winner (same set):  {bm:.4f}")
    print(f"  its score on a fresh held-out set:        {bh:.4f}")
    print(f"  optimistic bias you just shipped:         {bm - bh:+.4f}")

    print("\nREAD THIS: nothing improved. Every candidate was identical. Taking")
    print("the max of 20 noisy measurements and reporting that maximum is a")
    print("guaranteed overstatement -- the same reason you do not report")
    print("training loss as your test metric. Split the set, or stop reporting.")


def demo_8_changing_judges_midway():
    """Two judges, same systems, incomparable numbers."""
    line("DEMO 8: you changed the judge model. The numbers are now fiction.")

    rng = np.random.default_rng(8)
    n = 20000
    truth = run_eval(0.70, n, rng)
    print("one system, true accuracy 0.70, graded by two different judges:\n")
    for name, agree in (("judge v1 (agrees 0.92)", 0.92),
                        ("judge v2 (agrees 0.84)", 0.84)):
        print(f"  {name}: measured accuracy {judged_accuracy(truth, agree, rng):.4f}")
    m1 = judged_accuracy(truth, 0.92, rng)
    m2 = judged_accuracy(truth, 0.84, rng)
    print(f"\n  apparent change from v1 to v2: {m2 - m1:+.4f}")
    print("  actual change in the system:   +0.0000")
    print("\nREAD THIS: the system did not move. The ruler did. Pin the judge")
    print("model and its prompt to a version string, and when you change it,")
    print("re-run the whole history or start a new series. Never diff across.")


if __name__ == "__main__":
    demo_1_sample_size()
    demo_2_judge_agreement()
    demo_3_judge_biases()
    demo_4_contamination()
    demo_5_pairwise_vs_absolute()
    demo_6_regression_by_slice()
    demo_7_tune_and_report_on_the_same_set()
    demo_8_changing_judges_midway()

    line("THE WHOLE THING, COMPRESSED")
    print("""
  1. n matters   -- a 20-case golden set cannot see a 6-point difference
  2. judge noise -- measured_gap = (2*agreement - 1) * true_gap
  3. bias        -- position, length, self-preference; swap, bucket, panel
  4. leakage     -- only ever inflates, never averages out
  5. pairwise    -- less variance than a 1-5 score, same information
  6. slices      -- the aggregate hides the category that broke
  7. hygiene     -- tune on one set, report on another, pin the judge

  None of this needs a language model to demonstrate, and none of it is
  about language. It is a test suite with a flaky assertion, and every rule
  you already know about test suites applies.
""")
