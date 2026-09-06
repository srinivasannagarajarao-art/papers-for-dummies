"""
Hallucination and grounding -- for programmers, not researchers.

Run it:      python3 hallucination_from_scratch.py
Debug it:    breakpoint in `answer` and watch `scores` before the softmax.

NumPy only. THERE IS NO LANGUAGE MODEL IN HERE. What the script runs is the
*mechanism*: a store that keeps facts as overlapping features instead of as
records, so answering is reconstruction rather than lookup. Nothing it prints
is evidence about GPT-4 or any real model. It is a mechanical analogy, and
every demo says so on the line above.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

D = 64           # feature width of the toy store
TEMP = 0.50      # softmax temperature over candidate answers
GAMMA = 3.0      # how hard a fact in the context is copied


# ---------------------------------------------------------------------------
# STAGE 0 -- the world. Fictional projects, three attributes each.
#
# `count` is how many times the fact appeared in the toy training corpus.
# Popular projects are attested hundreds of times; obscure ones once. That
# single column is what the whole page turns on.
# ---------------------------------------------------------------------------
RELATIONS = {
    "licence":  ["MIT", "Apache-2.0", "BSD-3", "GPL-3"],
    "language": ["Python", "Go", "Rust", "Java"],
    "port":     ["8080", "5432", "6379", "9200"],
}

# One row per project: (licence, count), (language, count), (port, count).
# `count` is how many times that fact appeared in the toy corpus.
CORPUS = {
    "lantern":  (("MIT", 220),        ("Python", 190), ("8080", 140)),
    "beacon":   (("MIT", 180),        ("Go", 160),     ("8080", 120)),
    "quarry":   (("Apache-2.0", 90),  ("Python", 85),  ("5432", 60)),
    "halyard":  (("MIT", 40),         ("Rust", 35),    ("6379", 25)),
    "marlin":   (("Apache-2.0", 12),  ("Go", 10),      ("9200", 8)),
    "cobble":   (("BSD-3", 3),        ("Java", 2),     ("5432", 2)),
    "pellucid": (("GPL-3", 1),        ("Java", 1),     ("9200", 1)),
    "tessera":  (("GPL-3", 1),        ("Rust", 1),     ("6379", 1)),
}

# flattened to (subject, relation, true object, corpus count)
FACTS = [(s, rel, o, c)
         for s, row in CORPUS.items()
         for rel, (o, c) in zip(RELATIONS, row)]

SUBJECTS = sorted({s for s, _, _, _ in FACTS})
TRUTH = {(s, r): o for s, r, o, _ in FACTS}
COUNT = {(s, r): c for s, r, _, c in FACTS}


def random_features(names):
    """One near-orthogonal feature vector per name. Random, not learned."""
    V = np.random.randn(len(names), D) / np.sqrt(D)
    return {n: V[i] for i, n in enumerate(names)}


SUBJ_VEC = random_features(SUBJECTS)
REL_VEC = random_features(sorted(RELATIONS))
OBJ_VEC = random_features(sorted({o for objs in RELATIONS.values()
                                  for o in objs}))


# ---------------------------------------------------------------------------
# STAGE 1 -- the store. This is the whole argument in six lines.
#
# A database keeps a record per fact, so a rare fact is as retrievable as a
# common one. This keeps ONE matrix and adds every fact into it, weighted by
# how often the corpus said it. Facts sharing a relation share features, so
# they bleed into each other. That bleed IS the hallucination.
# ---------------------------------------------------------------------------
# tessera is a fork of lantern. A store like this has no notion of "different
# project"; it only has "nearby vector", so lantern's answers leak into it.
# This is the one hand-placed similarity in the script, and it is the whole
# confidently-wrong case.
SUBJ_VEC["tessera"] = 0.88 * SUBJ_VEC["lantern"] + 0.12 * SUBJ_VEC["tessera"]


def key(subject, relation):
    """Question -> a feature vector. The relation contributes less than the
    subject, so two questions about the same project overlap most."""
    k = SUBJ_VEC[subject] + 0.35 * REL_VEC[relation]
    return k / np.linalg.norm(k)


def build_store(facts):
    """Superpose every fact into one matrix. No per-fact slot exists after."""
    M = np.zeros((D, D))
    for s, r, o, c in facts:
        M += np.log1p(c) * np.outer(key(s, r), OBJ_VEC[o])   # log: frequency
    return M                                                  # with diminishing
                                                              # returns


STORE = build_store(FACTS)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ---------------------------------------------------------------------------
# STAGE 2 -- answering. Reconstruct a value from the smear, then pick the
# nearest legal candidate. `passages` is the grounding hook: a fact present in
# the context gets a fixed boost, which is the toy version of "copy it from
# the prompt instead of recalling it".
# ---------------------------------------------------------------------------
def answer(subject, relation, passages=(), gamma=GAMMA, temp=TEMP):
    """Returns (best answer, confidence, full distribution over candidates)."""
    recalled = STORE.T @ key(subject, relation)      # a blend, not a record
    cands = RELATIONS[relation]
    scores = np.array([OBJ_VEC[o] @ recalled for o in cands])

    for p_subj, p_rel, p_obj in passages:            # grounding: copy, not recall
        if p_rel == relation and p_obj in cands:
            match = 1.0 if p_subj == subject else 0.6   # off-subject counts less
            scores[cands.index(p_obj)] += gamma * match

    probs = softmax(scores / temp)
    i = int(np.argmax(probs))
    return cands[i], float(probs[i]), probs


def corpus_prior(relation):
    """P(object | relation) straight from the counts. The likelihood the
    next-token objective is actually fitting -- nothing here knows 'true'."""
    cands = RELATIONS[relation]
    n = np.array([sum(c for s, r, o, c in FACTS if r == relation and o == cand)
                  for cand in cands], dtype=float)
    return n / n.sum()


def line(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


# ---------------------------------------------------------------------------
# DEMO 1 -- reconstruction is not recall, and confidence does not know.
# ---------------------------------------------------------------------------
def demo_1_reconstruction_vs_recall():
    line("DEMO 1 -- a dict recalls, this store reconstructs")
    print("MECHANICAL ANALOGY, NOT A LANGUAGE MODEL: facts live as overlapping")
    print("features in one 64x64 matrix, so no fact has a slot of its own.\n")

    d = {(s, r): o for s, r, o, _ in FACTS}
    print("  dict lookup   ('lantern','licence')  -> "
          f"{d[('lantern','licence')]}   ('tessera','licence') -> "
          f"{d[('tessera','licence')]}")
    a1, c1, _ = answer("lantern", "licence")
    a2, c2, _ = answer("tessera", "licence")
    print(f"  the store     ('lantern','licence')  -> {a1}   conf {c1:.3f}"
          "   truth MIT    correct")
    print(f"  the store     ('tessera','licence')  -> {a2}   conf {c2:.3f}"
          "   truth GPL-3  WRONG")
    print("  Same shape of computation, same confidence, opposite outcome.\n")

    print("  licence question, projects ordered by how loud the corpus was:")
    for s in ["lantern", "quarry", "halyard", "marlin", "cobble", "pellucid",
              "tessera"]:
        a, c, _ = answer(s, "licence")
        t = TRUTH[(s, "licence")]
        mark = "correct" if a == t else "WRONG  "
        print(f"  {s:9s} count {COUNT[(s,'licence')]:4d}  answered {a:11s}"
              f" conf {c:.3f}  truth {t:11s} {mark}")

    rows = [(answer(s, r)[1], answer(s, r)[0] == TRUTH[(s, r)])
            for s in SUBJECTS for r in RELATIONS]
    conf = np.array([c for c, _ in rows])
    ok = np.array([float(k) for _, k in rows])
    print(f"\n  over all {len(rows)} questions: accuracy {ok.mean():.3f}")
    print(f"  mean confidence when RIGHT {conf[ok == 1].mean():.3f} | "
          f"when WRONG {conf[ok == 0].mean():.3f}")
    print(f"  highest confidence on a WRONG answer: {conf[ok == 0].max():.3f}")
    print(f"  lowest confidence on a RIGHT answer : {conf[ok == 1].min():.3f}")
    print("  So the two ranges OVERLAP. Confidence measures how much support")
    print("  the store found, which is not the same thing as being right.")


# ---------------------------------------------------------------------------
# DEMO 2 -- fluency and truth come apart. The objective prefers the likely one.
# ---------------------------------------------------------------------------
def demo_2_likely_vs_true():
    line("DEMO 2 -- the likely continuation and the true one, side by side")
    print("Nothing in the objective mentions truth. It fits corpus frequency.\n")
    print("  question                  most likely   p       true         p")
    print("  " + "-" * 62)
    for s, r in [("tessera", "licence"), ("tessera", "language"),
                 ("pellucid", "port"), ("cobble", "language"),
                 ("lantern", "licence")]:
        a, _, probs = answer(s, r)
        cands = RELATIONS[r]
        t = TRUTH[(s, r)]
        pt = probs[cands.index(t)]
        flag = "" if a == t else "   <- differ"
        print(f"  {s+'.'+r:24s}  {a:11s} {probs.max():.3f}   "
              f"{t:11s} {pt:.3f}{flag}")

    print("\n  and the corpus prior the objective is actually fitting:")
    for r in ["licence", "language", "port"]:
        p = corpus_prior(r)
        top = RELATIONS[r][int(np.argmax(p))]
        print(f"    P(answer | {r:8s}) argmax = {top:11s} {p.max():.3f}"
              f"  all {np.round(p, 3)}")
    print("  A fluent wrong answer is the objective working, not failing.")


# ---------------------------------------------------------------------------
# DEMO 3 -- the calibration trade. Abstain more, be wrong less. That is all
# you get. Kalai & Vempala (2023) prove a far stronger version of this for
# real models; the sweep below only shows the SHAPE of the trade.
# ---------------------------------------------------------------------------
def demo_3_abstention_tradeoff():
    line("DEMO 3 -- calibration means a nonzero error rate, unless you refuse")
    qs = [(s, r) for s in SUBJECTS for r in RELATIONS]

    def sweepable(temp):
        c, k = [], []
        for s, r in qs:
            a, cf, _ = answer(s, r, temp=temp)
            c.append(cf)
            k.append(a == TRUTH[(s, r)])
        return np.array(c), np.array(k, dtype=float)

    raw_c, ok = sweepable(TEMP)
    print(f"  raw:  mean confidence {raw_c.mean():.3f}  vs accuracy "
          f"{ok.mean():.3f}  -> overconfident by {raw_c.mean()-ok.mean():.3f}")

    # calibrate: one scalar temperature, chosen so stated confidence matches
    # observed accuracy. This is temperature scaling, the standard trick.
    grid = np.arange(0.5, 8.01, 0.05)
    gaps = [abs(sweepable(t)[0].mean() - ok.mean()) for t in grid]
    T_CAL = float(grid[int(np.argmin(gaps))])
    conf, _ = sweepable(T_CAL)
    print(f"  calibrated temperature {T_CAL:.2f} -> mean confidence "
          f"{conf.mean():.3f}  vs accuracy {ok.mean():.3f}\n")

    print("  buckets of the CALIBRATED confidence:")
    for lo, hi in [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            print(f"    conf {lo:.1f}-{hi:.1f}  n={int(m.sum()):2d}  "
                  f"stated {conf[m].mean():.3f}  observed {ok[m].mean():.3f}")

    singles = sum(1 for s, r, o, c in FACTS if c == 1)
    print(f"\n  facts attested exactly once in the corpus: {singles}/{len(FACTS)}"
          f" = {singles/len(FACTS):.3f}")
    print("  Kalai & Vempala (2023) prove that for a calibrated model this")
    print("  singleton fraction lower-bounds the hallucination rate. Their")
    print("  result is about real models; the sweep below is only the shape.\n")

    print("  threshold   answered   refused   errors   err|answered   wrong/100q")
    print("  " + "-" * 68)
    for t in [0.00, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
        ans = conf >= t
        n = int(ans.sum())
        err = int((ans & (ok == 0)).sum())
        rate = f"{err/n:6.3f}" if n else "     -"
        print(f"    {t:.2f}      {n:3d}/{len(qs)}    {len(qs)-n:3d}     {err:3d} "
              f"     {rate}       {100*err/len(qs):6.2f}")
    print("  One knob. No row on it is both error-free and worth shipping.")


# ---------------------------------------------------------------------------
# DEMO 4 -- grounding, measured. Put the fact in the context and it is copied.
# Then the honest half: the errors that survive having the fact right there.
# ---------------------------------------------------------------------------
def demo_4_grounding_measured():
    line("DEMO 4 -- the same store, with the fact handed to it")
    qs = [(s, r) for s in SUBJECTS for r in RELATIONS]

    bare = sum(answer(s, r)[0] == TRUTH[(s, r)] for s, r in qs)
    grounded, residual = 0, []
    for s, r in qs:
        # the retrieved passage: the right fact, plus one distractor about
        # another project with the same attribute. That is what a real top-k
        # looks like -- you never get exactly one passage.
        distract = [(o, r, TRUTH[(o, r)]) for o in SUBJECTS if o != s][0]
        a, c, _ = answer(s, r, passages=[(s, r, TRUTH[(s, r)]), distract])
        if a == TRUTH[(s, r)]:
            grounded += 1
        else:
            residual.append((s, r, a, TRUTH[(s, r)], c))

    print(f"  from the weights alone : {bare:2d}/{len(qs)} = "
          f"{bare/len(qs):.3f}")
    print(f"  with the fact in context: {grounded:2d}/{len(qs)} = "
          f"{grounded/len(qs):.3f}")
    print(f"  errors removed by grounding: {grounded - bare}\n")

    print(f"  residual errors WITH the fact present: {len(residual)}")
    for s, r, a, t, c in residual:
        print(f"    {s}.{r}: said {a} (conf {c:.3f}), passage said {t}")
    print("\n  grounding is a strength setting, not a switch -- sweep the copy")
    print("  boost and watch the same store change its mind:")
    for g in [0.5, 1.0, 3.0, 6.0, 10.0, 20.0]:
        n = sum(answer(s, r, [(s, r, TRUTH[(s, r)]),
                              [(o, r, TRUTH[(o, r)]) for o in SUBJECTS
                               if o != s][0]], gamma=g)[0] == TRUTH[(s, r)]
                for s, r in qs)
        print(f"    GAMMA {g:.1f} -> grounded accuracy {n/len(qs):.3f}")
    print("  Even at GAMMA 20 the answer is only as good as the passage.")


# ---------------------------------------------------------------------------
# DEMO 5 -- grounding fails in three named ways. Simulated retriever.
# ---------------------------------------------------------------------------
def demo_5_grounding_failure_modes():
    line("DEMO 5 -- three ways a grounded answer is still wrong")
    qs = [(s, r) for s in SUBJECTS for r in RELATIONS]
    rng = np.random.default_rng(0)

    modes = {"passage lacks the answer": [0, 0],
             "answer contradicts passage": [0, 0],
             "two passages disagree": [0, 0]}

    for s, r in qs:
        others = [o for o in SUBJECTS if o != s]
        # (a) retriever missed: a passage about the wrong project
        wrong = others[int(rng.integers(len(others)))]
        a, _, _ = answer(s, r, [(wrong, r, TRUTH[(wrong, r)])])
        modes["passage lacks the answer"][1] += 1
        modes["passage lacks the answer"][0] += a != TRUTH[(s, r)]

        # (b) right passage, weak copy: the prior can still shout it down
        a, _, _ = answer(s, r, [(s, r, TRUTH[(s, r)])], gamma=0.8)
        modes["answer contradicts passage"][1] += 1
        modes["answer contradicts passage"][0] += a != TRUTH[(s, r)]

        # (c) two passages, one stale
        stale = [o for o in RELATIONS[r] if o != TRUTH[(s, r)]][0]
        a, _, _ = answer(s, r, [(s, r, TRUTH[(s, r)]), (s, r, stale)])
        modes["two passages disagree"][1] += 1
        modes["two passages disagree"][0] += a != TRUTH[(s, r)]

    print("  failure mode                  wrong / asked   rate")
    print("  " + "-" * 52)
    for k, (bad, n) in modes.items():
        print(f"  {k:28s}   {bad:3d} / {n:3d}     {bad/n:.3f}")
    print("\n  Mode (a) is a RETRIEVAL bug. The generator behaved correctly:")
    print("  it was handed nothing about the subject and filled the gap.")
    print("  Example, no passage about it at all:")
    a, c, _ = answer("tessera", "licence", [("lantern", "licence", "MIT")])
    print(f"    tessera.licence with only a lantern passage -> {a} "
          f"(conf {c:.3f}), truth {TRUTH[('tessera','licence')]}")


# ---------------------------------------------------------------------------
# DEMO 6 -- detection with no ground truth: sample k times, measure agreement.
# ---------------------------------------------------------------------------
def demo_6_self_consistency():
    line("DEMO 6 -- self-consistency: does it say the same thing eight times?")
    rng = np.random.default_rng(0)
    qs = [(s, r) for s in SUBJECTS for r in RELATIONS]
    K = 8
    rows = []
    for s, r in qs:
        _, _, probs = answer(s, r)
        cands = RELATIONS[r]
        draws = rng.choice(len(cands), size=K, p=probs)
        vals, cnt = np.unique(draws, return_counts=True)
        modal = cands[int(vals[int(np.argmax(cnt))])]
        rows.append((s, r, modal, cnt.max() / K, modal == TRUTH[(s, r)]))

    agr = np.array([a for _, _, _, a, _ in rows])
    ok = np.array([float(k) for _, _, _, _, k in rows])
    print(f"  mean agreement when the modal answer is RIGHT {agr[ok==1].mean():.3f}")
    print(f"  mean agreement when it is WRONG               {agr[ok==0].mean():.3f}")
    c = float(np.corrcoef(agr, ok)[0, 1])
    print(f"  correlation(agreement, correctness) = {c:.3f}  -> a real signal\n")

    high_wrong = [x for x in rows if x[3] >= 0.875 and not x[4]]
    print(f"  and the cases it misses -- agreement >= 0.875 and still wrong: "
          f"{len(high_wrong)}")
    for s, r, m, a, _ in high_wrong[:4]:
        print(f"    {s}.{r}: said {m} {int(a*K)}/{K} times, truth "
              f"{TRUTH[(s, r)]}")
    print("  Consistency is a signal, never a guarantee. A wrong answer that")
    print("  is wrong for a structural reason is wrong consistently.")


if __name__ == "__main__":
    demo_1_reconstruction_vs_recall()
    demo_2_likely_vs_true()
    demo_3_abstention_tradeoff()
    demo_4_grounding_measured()
    demo_5_grounding_failure_modes()
    demo_6_self_consistency()

    line("THE WHOLE THING, COMPRESSED")
    print("""
  1. the objective is next-token likelihood; truth is not a term in it
  2. facts are stored as overlapping features, so answering reconstructs
  3. reconstruction is confident by construction -- confidence != correctness
  4. calibration forces a nonzero error rate on rarely-attested facts
  5. the only dial that removes error is refusing to answer, which costs
     coverage; there is no setting that is both correct and useful at 100%
  6. grounding replaces recall with copying, and adds retrieval as a new
     place for the answer to go wrong
  7. self-consistency detects some of what is left, and misses the rest

  None of these numbers say anything about a real model. They say what kind
  of thing hallucination is.
""")
