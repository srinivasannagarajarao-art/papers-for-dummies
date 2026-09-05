"""
Chinchilla -- Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)
for programmers, not researchers.

Run it:      python3 scaling_laws_from_scratch.py
Debug it:    put a breakpoint in optimal_split and watch the sweep.

No torch. No training. The paper is a curve fit plus a constrained
optimisation, and both are a few lines of NumPy. Every function is short.

Every coefficient below is a PUBLISHED fitted value, not something derived
here. This script only re-solves the optimisation they imply -- and that
part is arithmetic you can check.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)


def compute_flops(N, D):
    """Training FLOPs for N parameters over D tokens."""
    # 2 FLOPs forward + 4 backward, per parameter per token. That is the
    # whole derivation of the 6. It ignores attention's own quadratic term,
    # which is small while context is short relative to model width.
    return 6.0 * N * D


def tokens_for_budget(N, C):
    """Invert C = 6ND: how many tokens does budget C buy at size N?"""
    return C / (6.0 * N)


# Two coefficient sets, BOTH published, NEITHER derived here.
#   PAPER -- Hoffmann et al. 2022, the "Approach 3" parametric fit as printed.
#   REFIT -- Besiroglu et al. 2024 replication, which refitted the same data
#            because the printed coefficients do not reproduce the paper's
#            own headline. Demo 2 shows the gap. REFIT is the default.
#
#   L(N, D) = E + A/N^alpha + B/D^beta
#     E            an irreducible floor: no model in the fit beats it
#     A/N^alpha    the penalty for being too small
#     B/D^beta     the penalty for not having seen enough text
PAPER = dict(E=1.69,   A=406.4,  B=410.7,   alpha=0.34,   beta=0.28)
REFIT = dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)


def loss(N, D, coef=REFIT):
    """Predicted loss in nats per token for N params trained on D tokens."""
    return (coef["E"] + coef["A"] / N ** coef["alpha"]
            + coef["B"] / D ** coef["beta"])


def optimal_split(C, coef=REFIT, n_lo=1e6, n_hi=1e15, steps=20000):
    """Return (N*, D*, loss*) minimising loss under budget C = 6ND."""
    # THE WHOLE PAPER: minimise L(N, D) subject to 6ND = C.
    # One budget, one free variable: pick N, and D falls out of the
    # constraint. So sweep N and take the argmin. No calculus needed.
    Ns = np.logspace(np.log10(n_lo), np.log10(n_hi), steps)
    Ds = tokens_for_budget(Ns, C)
    Ls = loss(Ns, Ds, coef)
    i = int(np.argmin(Ls))
    return Ns[i], Ds[i], Ls[i]


# The correction the paper deliberately did NOT make. Serving costs about
# 2 FLOPs per parameter per generated token, no backward pass. If you expect
# to serve D_inf tokens, the budget you really control is
#     C_total = 6*N*D_train + 2*N*D_inf
# Same minimisation, different constraint. This is the LLaMA argument.
def optimal_split_with_inference(C_total, D_inf, n_lo=1e6, n_hi=1e15,
                                 steps=20000):
    """Minimise loss subject to 6*N*D_train + 2*N*D_inf = C_total."""
    Ns = np.logspace(np.log10(n_lo), np.log10(n_hi), steps)
    Ds = (C_total - 2.0 * Ns * D_inf) / (6.0 * Ns)   # may go negative
    ok = Ds > 0
    Ls = np.where(ok, loss(Ns, np.where(ok, Ds, 1.0)), np.inf)
    i = int(np.argmin(Ls))
    return Ns[i], Ds[i], Ls[i]


# Kaplan et al. (2020) fitted N_opt ~ C^0.73: spend it on size. Their paper
# gives an exponent, not a constant, so anchor on a model built under that
# advice -- GPT-3. The exponent is theirs; the anchor is mine.
KAPLAN_EXPONENT = 0.73
ANCHOR_N, ANCHOR_D = 175e9, 300e9
ANCHOR_C = compute_flops(ANCHOR_N, ANCHOR_D)


def kaplan_split(C):
    """Kaplan-style (N, D) for budget C, anchored on GPT-3."""
    N = ANCHOR_N * (C / ANCHOR_C) ** KAPLAN_EXPONENT
    return N, tokens_for_budget(N, C)


def fmt(x):
    """Human-readable big number: 7.0e+10 -> '70.0B'."""
    if abs(x) >= 1e15:
        return f"{x:.1e}"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:.1f}{suf}"
    return f"{x:.1f}"


def line(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ===========================================================================
# DEMOS
# ===========================================================================

def demo_1_compute_rule():
    line("DEMO 1: C = 6ND -- the only arithmetic in this paper")

    configs = [
        ("GPT-3",      175e9, 300e9),
        ("Gopher",     280e9, 300e9),
        ("Chinchilla",  70e9, 1.4e12),
        ("LLaMA-7B",     7e9, 1.0e12),
    ]
    print(f"{'model':<12}{'params':>9}{'tokens':>9}{'tok/param':>11}"
          f"{'FLOPs':>12}")
    for name, N, D in configs:
        print(f"{name:<12}{fmt(N):>9}{fmt(D):>9}{D / N:>11.1f}"
              f"{compute_flops(N, D):>12.2e}")

    print("\nREAD THIS: 6 = 2 forward + 4 backward, per parameter per token.")
    print("That is the whole budget model. Everything else in the paper is")
    print("deciding how to split N against D under this one equation.")


def demo_2_the_optimisation():
    line("DEMO 2: fix the budget, sweep the split, watch the loss")

    C = 5.76e23    # Chinchilla's own training budget: 6 * 70e9 * 1.4e12
    print(f"budget C = {C:.2e} FLOPs (roughly what Chinchilla spent)\n")
    print(f"{'N (params)':>12}{'D (tokens)':>12}{'tok/param':>11}"
          f"{'predicted loss':>16}")
    for N in (1e9, 5e9, 20e9, 50e9, 72e9, 100e9, 200e9, 500e9, 2e12):
        D = tokens_for_budget(N, C)
        print(f"{fmt(N):>12}{fmt(D):>12}{D / N:>11.1f}{loss(N, D):>16.4f}")

    N1, D1, L1 = optimal_split(C)
    print(f"\nargmin over 20000 candidate splits (REFIT coefficients):")
    print(f"  N* = {fmt(N1)} params")
    print(f"  D* = {fmt(D1)} tokens")
    print(f"  D*/N* = {D1 / N1:.1f} tokens per parameter")
    print(f"  loss  = {L1:.4f} nats/token")
    print(f"  Chinchilla as actually built: 70.0B params, 1.4T tokens, "
          f"20.0 tok/param")

    print("\nsame sweep, ratio only, across five budgets:")
    print(f"{'budget C':>10}{'REFIT tok/param':>17}{'PAPER tok/param':>17}")
    for C2 in (1e21, 1e22, 1e23, 5.76e23, 1e25):
        r = optimal_split(C2)
        p = optimal_split(C2, PAPER)
        print(f"{C2:>10.0e}{r[1] / r[0]:>17.1f}{p[1] / p[0]:>17.1f}")

    print("\nREAD THIS: the curve is a bowl. Too small and the A/N term")
    print("dominates; too big and you starve it of tokens and the B/D term")
    print("dominates. With the replication's refitted coefficients the bottom")
    print("sits near 20 tokens per parameter at every budget -- the paper's")
    print("headline, recovered rather than asserted, and landing within a few")
    print("percent of the model they actually built. With the coefficients as")
    print("PRINTED in the paper it lands near 90, which is the discrepancy")
    print("the 2024 replication reported. I use the refit and say so.")


def demo_3_kaplan_vs_chinchilla():
    line("DEMO 3: Kaplan vs Chinchilla, same money")

    print(f"{'budget C':>10}{'Kaplan N':>11}{'Kaplan D':>10}{'t/p':>7}"
          f"{'loss':>9} {'Chin. N':>9}{'Chin. D':>10}{'t/p':>7}{'loss':>9}")
    for C in (1e21, 1e22, 1e23, 5.76e23, 1e24, 1e25):
        kn, kd = kaplan_split(C)
        cn, cd, cl = optimal_split(C)
        print(f"{C:>10.0e}{fmt(kn):>11}{fmt(kd):>10}{kd / kn:>7.1f}"
              f"{loss(kn, kd):>9.4f} {fmt(cn):>9}{fmt(cd):>10}"
              f"{cd / cn:>7.1f}{cl:>9.4f}")

    C = 5.76e23
    kn, kd = kaplan_split(C)
    cn, cd, cl = optimal_split(C)
    print(f"\nat C = {C:.2e}:")
    print(f"  Kaplan wants a model {kn / cn:.1f}x bigger, fed "
          f"{cd / kd:.1f}x fewer tokens")
    print(f"  and pays {loss(kn, kd) - cl:.4f} nats/token for it")

    print("\nREAD THIS: Kaplan's rule buys parameters it cannot afford to")
    print("train. Same spend, worse model. The two papers were not doing the")
    print("same experiment: Kaplan held the learning-rate schedule fixed")
    print("across run lengths (so short runs looked artificially bad) and")
    print("counted parameters excluding embeddings. Chinchilla tuned the")
    print("schedule to each run's length. Different setup, different answer.")


def demo_4_chinchilla_vs_gopher():
    line("DEMO 4: Chinchilla vs Gopher -- same compute, half the size")

    for name, N, D in (("Gopher", 280e9, 300e9), ("Chinchilla", 70e9, 1.4e12)):
        C = compute_flops(N, D)
        print(f"{name:<12} N={fmt(N):>7}  D={fmt(D):>7}  "
              f"tok/param={D / N:6.1f}  C={C:.3e}  loss={loss(N, D):.4f}")

    c_gopher = compute_flops(280e9, 300e9)
    c_chin = compute_flops(70e9, 1.4e12)
    print(f"\ncompute ratio Chinchilla/Gopher = {c_chin / c_gopher:.3f}"
          f"  (i.e. the same training bill)")
    print(f"predicted loss gap = {loss(280e9, 300e9) - loss(70e9, 1.4e12):+.4f}"
          f" nats/token in Chinchilla's favour")

    print("\nREAD THIS: Chinchilla is Gopher's budget spent differently. 4x")
    print("smaller, 4.7x more data. In the paper it beat Gopher on the")
    print("benchmark suite, and it is also cheaper to serve forever after.")
    print("The loss numbers above are what THIS fit predicts, not measured")
    print("benchmark scores -- do not quote them as the paper's results.")


def demo_5_inference_changes_the_answer():
    line("DEMO 5: add the serving bill and the optimum shrinks")

    C_total = 5.76e23
    print(f"total budget (train + serve) = {C_total:.2e} FLOPs\n")
    print(f"{'inference tokens':>17}{'N*':>9}{'D* train':>11}{'tok/param':>11}"
          f"{'train loss':>12}")
    for D_inf in (0.0, 1e11, 1e12, 1e13, 5e13, 1e14):
        N, D, L = optimal_split_with_inference(C_total, D_inf)
        label = "none" if D_inf == 0 else fmt(D_inf)
        print(f"{label:>17}{fmt(N):>9}{fmt(D):>11}{D / N:>11.1f}{L:>12.4f}")

    print("\nREAD THIS: as the number of tokens you expect to SERVE rises,")
    print("the best model under the same total budget gets smaller and the")
    print("training set gets larger. Chinchilla answers 'what is the best")
    print("model I can TRAIN for this money'. LLaMA answered 'what is the")
    print("best model I can RUN', which is why LLaMA-7B saw ~1T tokens --")
    print("about 140 tokens per parameter, seven times past Chinchilla.")


def demo_6_extrapolation_danger():
    line("DEMO 6: where the fit stops being trustworthy")

    print("the paper's runs covered roughly 70M to 16B params and 5B to")
    print("400B tokens. here is the same formula far outside that box:")
    print()
    print(f"{'N':>10}{'D':>10}{'tok/param':>11}{'predicted loss':>16}")
    for N, D in ((7e7, 1.4e9), (7e10, 1.4e12), (1e13, 2e14), (1e16, 2e17),
                 (1e19, 2e20), (1e3, 2e4)):
        print(f"{fmt(N):>10}{fmt(D):>10}{D / N:>11.1f}{loss(N, D):>16.4f}")

    E = REFIT["E"]
    print(f"\nthe fit's floor is E = {E}: at infinite N and D the formula")
    print(f"says loss -> {E:.4f} nats/token and never lower.")
    print(f"loss(1e19 params, 2e20 tokens) - E = {loss(1e19, 2e20) - E:.2e}")

    print("\nREAD THIS: three decimal places of confidence about a model a")
    print("million times larger than anything that was fitted. E is a fitted")
    print("constant, not a measured entropy of English -- treat it as the")
    print("asymptote of a curve drawn through a specific box of runs. Two")
    print("orders of magnitude out, this is arithmetic, not evidence.")


def demo_7_break_it():
    line("DEMO 7: BREAK IT -- three ways to get this wrong, side by side")

    C = 5.76e23
    cn, cd, cl = optimal_split(C)

    print("(a) train at Kaplan's ratio instead of the optimum")
    kn, kd = kaplan_split(C)
    print(f"    correct : N={fmt(cn):>8} D={fmt(cd):>8} "
          f"loss={cl:.4f}   <- optimal")
    print(f"    broken  : N={fmt(kn):>8} D={fmt(kd):>8} "
          f"loss={loss(kn, kd):.4f}   <- same money, worse model")

    print("\n(b) use C = 6ND on a mixture-of-experts model")
    total, active, D = 47e9, 13e9, 1e12
    print(f"    correct : active params {fmt(active)}, "
          f"C={compute_flops(active, D):.3e}")
    print(f"    broken  : total  params {fmt(total)}, "
          f"C={compute_flops(total, D):.3e}   <- overstated "
          f"{compute_flops(total, D) / compute_flops(active, D):.1f}x")

    print("\n(c) fix the token count and only buy parameters")
    D_fixed = 300e9
    print(f"    D pinned at {fmt(D_fixed)} tokens:")
    prev = None
    for N in (10e9, 70e9, 280e9, 1e12, 1e13):
        L = loss(N, D_fixed)
        delta = "" if prev is None else f"   delta {L - prev:+.4f}"
        print(f"      N={fmt(N):>8}  loss={L:.4f}{delta}")
        prev = L
    floor = REFIT["E"] + REFIT["B"] / D_fixed ** REFIT["beta"]
    print(f"    floor with D pinned = {floor:.4f} (N -> infinity)")

    print("\nREAD THIS: (a) costs you real loss for the same spend. (b) is the")
    print("commonest scaling-law error in production -- MoE breaks the 6ND")
    print("rule because only the active experts do work. (c) is the curve")
    print("flattening: with tokens pinned, every 4x of parameters buys less")
    print("than the last, and the whole column is trapped above the 1.9493")
    print("floor, because the B/D term does not care how big your model is.")


if __name__ == "__main__":
    demo_1_compute_rule()
    demo_2_the_optimisation()
    demo_3_kaplan_vs_chinchilla()
    demo_4_chinchilla_vs_gopher()
    demo_5_inference_changes_the_answer()
    demo_6_extrapolation_danger()
    demo_7_break_it()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. budget      C = 6ND. 2 FLOPs forward, 4 back, per param per token.
  2. loss model  L(N,D) = E + A/N^a + B/D^b, fitted over 400+ runs.
  3. the answer  minimise L subject to 6ND = C  ->  ~20 tokens/param.
  4. the result  Chinchilla 70B on 1.4T beat Gopher 280B on 300B, same C.
  5. the caveat  this is compute-optimal TRAINING, not optimal SERVING.
  6. the limit   E, A, B, a, b are a fit inside one box of runs. Outside
                 that box the formula still returns a number. It is not
                 evidence, it is arithmetic.
""")
