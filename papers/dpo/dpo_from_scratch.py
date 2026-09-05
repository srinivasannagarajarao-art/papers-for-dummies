"""
Direct Preference Optimization -- for programmers, not researchers.

Run it:      python3 papers/dpo/dpo_from_scratch.py
Debug it:    set a breakpoint in train_dpo and watch the implicit reward
             beta * (log pi - log ref) move while no reward model exists.

The same toy world as the RLHF page: 4 prompts, 8 candidate answers each, a
hidden "true" reward nobody in the algorithm ever sees, and noisy pairwise
votes. NumPy for the mechanism (the closed form, the inversion, the KL, the
policy gradient). Torch only for the two training loops that need backprop:
the RLHF baseline's reward model, and DPO itself. CPU, a few seconds.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(precision=3, suppress=True)

N_PROMPTS, N_RESP = 4, 8
BETA = 0.5                       # the KL coefficient, same meaning as in RLHF


def log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)          # overflow safety, cancels
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# STAGE 0 -- the toy world. Identical to the RLHF page's, on purpose: this
# page is that page's sequel, and the only fair comparison is the same data.
# ---------------------------------------------------------------------------
def make_world(rng):
    """(4, 8, 4): 4 prompts x 8 answers x [correct, length, hedges, polite].
    Answer 0 is a one-liner, answer 7 a wall of text, in every prompt."""
    u = lambda: rng.uniform(0, 1, (N_PROMPTS, N_RESP))
    length = np.tile(np.linspace(0.1, 1.0, N_RESP), (N_PROMPTS, 1))
    correct = (u() < 0.15 + 0.8 * length).astype(float)
    hedges = u() ** 2 * (1.0 - 0.7 * length)
    polite = u() * (0.4 + 0.6 * length)
    return np.stack([correct, length, hedges, polite], axis=-1)


def what_the_reward_model_sees(feats, rng):
    """A reward model reads TEXT. Length, hedging and politeness are on the
    surface. Whether an answer is RIGHT it guesses, wrong one time in four."""
    seen = feats.copy()
    flip = rng.uniform(0, 1, feats.shape[:2]) < 0.25
    seen[..., 0] = np.where(flip, 1 - feats[..., 0], feats[..., 0])
    return seen


def true_reward(feats):
    """What people ACTUALLY want. Nothing in this file ever sees it except
    the scoreboard; the algorithms only ever see which of two answers won."""
    correct, length, hedges, polite = np.moveaxis(feats, -1, 0)
    return (3.0 * correct                          # being right is most of it
            + 1.0 * polite                         # manners help a little
            - 1.5 * hedges                         # "it depends..." is annoying
            - 8.0 * np.maximum(0, length - 0.6))   # past 0.6 nobody reads it


def reference_policy(feats):
    """The SFT model's habits: short-to-medium answers. Log-probs, (4, 8)."""
    length = feats[..., 1]
    return log_softmax(-8.0 * np.maximum(0, length - 0.55))


def long_winded_policy(feats):
    """A distribution the reference would essentially never produce: it
    writes the LONG answers. Used to break DPO's off-policy assumption."""
    length = feats[..., 1]
    return log_softmax(8.0 * length)


# ---------------------------------------------------------------------------
# STEP 1 -- the closed form.
#
# RLHF maximises   E_{y~pi}[ r(x, y) ]  -  beta * KL( pi || pi_ref ).
# For a FIXED prompt that objective has an exact maximiser, no RL required:
#
#     pi*(y|x) = pi_ref(y|x) * exp( r(x, y) / beta ) / Z(x)
#
# Reference times an exponential tilt by the reward, renormalised. Appendix
# A.1 of the paper; the same identity turns up wherever a KL-regularised
# objective does. Everything else in DPO is algebra on this one line.
# ---------------------------------------------------------------------------
def optimal_policy(r, ref_logp, beta):
    """The closed form above, in log space so nothing overflows. (4, 8)."""
    return log_softmax(ref_logp + r / beta)


def kl_objective(logp, r, ref_logp, beta):
    """The RLHF objective itself: expected reward minus beta * KL."""
    p = np.exp(logp)
    return ((p * r).sum(-1) - beta * (p * (logp - ref_logp)).sum(-1)).mean()


def adam_step(grad, state, lr, t, b1=0.9, b2=0.999):
    """One Adam update. state = [first moment, second moment], in place."""
    state[0] = b1 * state[0] + (1 - b1) * grad
    state[1] = b2 * state[1] + (1 - b2) * grad ** 2
    m_hat, v_hat = state[0] / (1 - b1 ** t), state[1] / (1 - b2 ** t)
    return lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def solve_objective_numerically(r, ref_logp, beta, steps=4000, lr=0.05):
    """Climb the SAME objective by brute force, with no formula, so the
    formula has something to be checked against. Exact gradient: with
    p = softmax(theta) and u = r - beta * (log p - log ref), the derivative
    of the objective w.r.t. theta_j is p_j * (u_j - sum_i p_i u_i). The
    entropy term's own derivative cancels, which is why it's this short."""
    theta = ref_logp.copy()
    state = [np.zeros_like(theta), np.zeros_like(theta)]
    for t in range(1, steps + 1):
        logp = log_softmax(theta)
        p = np.exp(logp)
        u = r - beta * (logp - ref_logp)
        grad = p * (u - (p * u).sum(-1, keepdims=True))
        theta += adam_step(grad, state, lr, t)
    return log_softmax(theta)


# ---------------------------------------------------------------------------
# STEP 2 -- the inversion. Take logs of the closed form and rearrange:
#
#     r(x, y) = beta * ( log pi*(y|x) - log pi_ref(y|x) ) + beta * log Z(x)
#
# The reward is a function of the policy. It was never a separate object.
# The leftover beta*log Z(x) depends only on the PROMPT, so in any pairwise
# comparison of two answers to the same prompt it subtracts away.
# ---------------------------------------------------------------------------
def implicit_reward(logp, ref_logp, beta):
    """beta * log-ratio. The reward, recovered from the policy, up to a
    per-prompt constant. This is the whole 'secretly a reward model' claim."""
    return beta * (logp - ref_logp)


def spearman(a, b):
    """Rank correlation: +1 same order, -1 reversed. Ranks, not values,
    because a recovered reward's scale and offset are its own."""
    ra = np.argsort(np.argsort(a.ravel()))
    rb = np.argsort(np.argsort(b.ravel()))
    return np.corrcoef(ra, rb)[0, 1]


# ---------------------------------------------------------------------------
# STEP 3 -- substitute the inversion into the Bradley-Terry preference loss
# and the reward model vanishes. What is left is a classification loss on
# pairs, over the policy's own log-probs:
#
#   loss = -log sigmoid( beta*log(pi(yw)/ref(yw)) - beta*log(pi(yl)/ref(yl)) )
#
# No reward model, no sampling, no RL. The reference is still needed, in
# memory, for both log ref terms.
# ---------------------------------------------------------------------------
def dpo_loss(logp, ref_logp, p, w, l, beta, drop_reference=False, no_pair=False):
    """logp: torch (4, 8) log-probs of the policy. p/w/l: prompt, winner,
    loser index tensors. The two flags are the deliberate breaks."""
    ref = torch.zeros_like(logp) if drop_reference else ref_logp
    margin_w = beta * (logp[p, w] - ref[p, w])
    if no_pair:
        return -F.logsigmoid(margin_w).mean()      # the break: no contrast
    margin_l = beta * (logp[p, l] - ref[p, l])
    return -F.logsigmoid(margin_w - margin_l).mean()


def train_dpo(comparisons, ref_logp, beta=BETA, steps=600, lr=0.05,
              log_every=0, drop_reference=False, no_pair=False, swap=False):
    """Fit the policy directly on the votes. Returns its log-probs, (4, 8).
    The policy is a table of logits here; in the paper it is a language
    model and logp is the sum of token log-probs of the whole answer."""
    torch.manual_seed(0)
    ref = torch.tensor(ref_logp, dtype=torch.float32)
    logits = ref.clone().requires_grad_(True)       # start AT the reference
    opt = torch.optim.Adam([logits], lr=lr)
    p, w, l = (torch.tensor([c[i] for c in comparisons]) for i in range(3))
    if swap:
        w, l = l, w                                 # the break: winner/loser
    for step in range(1, steps + 1):
        loss = dpo_loss(F.log_softmax(logits, dim=-1), ref, p, w, l, beta,
                        drop_reference=drop_reference, no_pair=no_pair)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step == 1 or step % log_every == 0):
            print(f"  step {step:4d}   loss {loss.item():.4f}")
    with torch.no_grad():
        return F.log_softmax(logits, dim=-1).numpy()


# ---------------------------------------------------------------------------
# THE VOTES. Sample two answers to the same prompt from some sampling policy,
# show a person, record the winner. Noisy the Bradley-Terry way:
# P(A wins) = sigmoid(true_A - true_B). Explained on the RLHF page.
# ---------------------------------------------------------------------------
def collect_comparisons(n, sample_logp, R_true, rng):
    out = []
    for _ in range(n):
        p = rng.integers(N_PROMPTS)
        a, b = rng.choice(N_RESP, size=2, replace=False,
                          p=np.exp(sample_logp[p]))
        win = a if rng.uniform() < sigmoid(R_true[p, a] - R_true[p, b]) else b
        out.append((p, win, a + b - win))          # (prompt, winner, loser)
    return out


# ---------------------------------------------------------------------------
# THE RLHF BASELINE, for the one table where the two are compared. This is
# the RLHF page's code, unchanged: fit a reward model with Bradley-Terry,
# then climb it with REINFORCE and a KL penalty. Two loops, two models.
# ---------------------------------------------------------------------------
class RewardModel(nn.Module):
    def __init__(self, n_features=4, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_reward_model(comparisons, feats, ref_logp, steps=400, lr=0.02):
    """Fit r, return its frozen (4, 8) score table, normalised so answers
    the reference writes sit at mean 0, std 1."""
    torch.manual_seed(4)
    rm = RewardModel()
    opt = torch.optim.Adam(rm.parameters(), lr=lr)
    X = torch.tensor(feats, dtype=torch.float32)
    p, w, l = (torch.tensor([c[i] for c in comparisons]) for i in range(3))
    for _ in range(steps):
        loss = -F.logsigmoid(rm(X[p, w]) - rm(X[p, l])).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        table = rm(X).numpy()
    ref = np.exp(ref_logp)
    mean = (ref * table).sum() / N_PROMPTS
    std = np.sqrt((ref * (table - mean) ** 2).sum() / N_PROMPTS)
    return (table - mean) / std


def train_policy_rl(rm_table, ref_logp, beta, steps=300, lr=0.05, batch=64):
    """REINFORCE against the reward model, KL penalty folded into the reward.
    Sampling, a second model, and a hyperparameter DPO does not have."""
    rng = np.random.default_rng(0)
    logits = ref_logp.copy()
    state = [np.zeros_like(logits), np.zeros_like(logits)]
    for step in range(steps):
        logp = log_softmax(logits)
        probs = np.exp(logp)
        grad = np.zeros_like(logits)
        for p in range(N_PROMPTS):
            a = rng.choice(N_RESP, size=batch, p=probs[p])
            R = rm_table[p, a] - beta * (logp[p, a] - ref_logp[p, a])
            adv = R - R.mean()
            grad[p] = (adv[:, None] * (np.eye(N_RESP)[a] - probs[p])).mean(0)
        logits += adam_step(grad, state, lr, step + 1)
    return log_softmax(logits)


# ---------------------------------------------------------------------------
# SCOREBOARD helpers -- the numbers you never get in real life.
# ---------------------------------------------------------------------------
def kl_to_reference(logp, ref_logp):
    return (np.exp(logp) * (logp - ref_logp)).sum(-1).mean()


def true_score(logp, R_true):
    return (np.exp(logp) * R_true).sum(-1).mean()


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_world():
    rng = np.random.default_rng(0)
    feats = make_world(rng)
    seen = what_the_reward_model_sees(feats, rng)
    return feats, seen, true_reward(feats), reference_policy(feats)


def demo_1_closed_form(R_true, ref_logp):
    line("DEMO 1: the KL-constrained objective has a closed-form maximiser")
    print("objective(pi) = E_pi[r] - beta * KL(pi || pi_ref),  beta = 0.5")
    print("claim:  pi*(y) = pi_ref(y) * exp(r(y)/beta) / Z(x)\n")
    closed = optimal_policy(R_true, ref_logp, BETA)
    solved = solve_objective_numerically(R_true, ref_logp, BETA)
    print("prompt 0, in probabilities:")
    print("  formula :", np.exp(closed[0]))
    print("  solver  :", np.exp(solved[0]))
    print(f"\n  max |formula - solver| over all 32 entries: "
          f"{np.abs(np.exp(closed) - np.exp(solved)).max():.6f}")
    print(f"  objective at the formula's policy: "
          f"{kl_objective(closed, R_true, ref_logp, BETA):.6f}")
    print(f"  objective at the solver's policy:  "
          f"{kl_objective(solved, R_true, ref_logp, BETA):.6f}")
    print("\nREAD THIS: 4000 steps of gradient ascent land where one line of")
    print("algebra already was. The RL in RLHF is a numerical search for a")
    print("point you can write down -- IF you have r. You don't. Yet.")
    return closed


def demo_2_inversion(R_true, ref_logp, closed):
    line("DEMO 2: invert it -- the reward is recoverable FROM the policy")
    print("take logs of the closed form and rearrange:")
    print("  r(x, y) = beta * ( log pi*(y|x) - log ref(y|x) ) + beta*log Z(x)\n")
    rec = implicit_reward(closed, ref_logp, BETA)
    print("prompt 0:")
    print("  hidden true reward :", R_true[0])
    print("  beta * log-ratio   :", rec[0])
    print("  difference         :", (R_true[0] - rec[0]))
    print("\nthe difference is CONSTANT within a prompt. That constant is")
    print("beta*log Z(x), one per prompt, and here it is:")
    for p in range(N_PROMPTS):
        d = R_true[p] - rec[p]
        print(f"  prompt {p}:  beta*log Z = {d.mean():7.4f}   "
              f"spread within prompt = {d.max() - d.min():.2e}")
    within = np.mean([spearman(rec[p], R_true[p]) for p in range(N_PROMPTS)])
    print(f"\nrank correlation within each prompt, averaged: {within:.3f}")
    print(f"rank correlation across all 32 answers at once:  "
          f"{spearman(rec, R_true):.3f}")
    print("the first is what matters. The second is lower only because each")
    print("prompt carries its own beta*log Z offset -- which is exactly the")
    print("term that cancels when you compare two answers to one prompt.")
    print("\npairwise, the constant cancels. Prompt 0, answers 3 and 7:")
    a, b = 3, 7
    print(f"  true r[3] - true r[7]      = {R_true[0, a] - R_true[0, b]:7.4f}")
    print(f"  recovered[3] - recovered[7] = {rec[0, a] - rec[0, b]:7.4f}")
    print("\nREAD THIS: a preference model only ever compares two answers to")
    print("the SAME prompt, so beta*log Z(x) -- the only term that needed the")
    print("normaliser, the only term that was expensive -- subtracts away. The")
    print("reward model was a pure function of the policy all along.")


def demo_3_dpo_vs_rlhf(feats, seen, R_true, ref_logp):
    line("DEMO 3: the DPO loss, and the same job with a third of the parts")
    rng = np.random.default_rng(2)
    comps = collect_comparisons(2000, ref_logp, R_true, rng)
    print("2000 noisy votes, sampled from the reference. DPO trains ONE thing")
    print("on them, with plain gradient descent and no sampling:\n")
    dpo_logp = train_dpo(comps, ref_logp, log_every=100)

    rm = train_reward_model(comps, seen, ref_logp)
    rl_logp = train_policy_rl(rm, ref_logp, beta=BETA)
    best = R_true.max(-1).mean()
    print("\n  policy                      true reward    KL to ref   parts")
    print(f"  reference (SFT)             {true_score(ref_logp, R_true):10.3f}   "
          f"{kl_to_reference(ref_logp, ref_logp):10.3f}   1")
    print(f"  RLHF: reward model + RL     {true_score(rl_logp, R_true):10.3f}   "
          f"{kl_to_reference(rl_logp, ref_logp):10.3f}   3")
    print(f"  DPO: one classifier loss    {true_score(dpo_logp, R_true):10.3f}   "
          f"{kl_to_reference(dpo_logp, ref_logp):10.3f}   2")
    print(f"  best possible               {best:10.3f}          -    -")
    print("\nREAD THIS: same votes, same beta, comparable answers. RLHF needed")
    print("a reward model, a sampling loop and a policy-gradient estimator.")
    print("DPO needed a log-probability and a sigmoid. It still needs the")
    print("reference model in memory -- log ref appears twice in the loss.")
    return comps, dpo_logp


def demo_4_implicit_reward(R_true, ref_logp, dpo_logp):
    line("DEMO 4: a reward model WAS trained. It just isn't a separate object")
    rec = implicit_reward(dpo_logp, ref_logp, BETA)
    print("beta * (log pi_dpo - log ref), read straight off the trained policy:\n")
    print("  ans      true    implicit")
    for a in range(N_RESP):
        print(f"  {a:>3}   {R_true[0, a]:7.3f}   {rec[0, a]:9.3f}")
    print(f"\nrank correlation with the hidden true reward, all 32 answers: "
          f"{spearman(rec, R_true):.3f}")
    print("\nREAD THIS: nobody built a reward model. Nobody scored an answer.")
    print("And yet there is a scorer in there, and it ranks answers the way")
    print("people do. That is the paper's title: your language model is")
    print("secretly a reward model. You can use this at eval time.")


def demo_5_beta_sweep(R_true, ref_logp, comps, thin):
    line("DEMO 5: beta is the leash, exactly as it was in RLHF")
    print("2000 votes on the left, 150 votes on the right. Same sweep.\n")
    print("  beta    KL to ref   true (2000)   KL to ref   true (150 votes)")
    for beta in (0.05, 0.1, 0.5, 1.0, 5.0):
        lp = train_dpo(comps, ref_logp, beta=beta)
        tp = train_dpo(thin, ref_logp, beta=beta)
        print(f"  {beta:4.2f}   {kl_to_reference(lp, ref_logp):9.3f}   "
              f"{true_score(lp, R_true):11.3f}   "
              f"{kl_to_reference(tp, ref_logp):9.3f}   "
              f"{true_score(tp, R_true):16.3f}")
    print(f"\n  reference itself:  KL     0.000   true "
          f"{true_score(ref_logp, R_true):11.3f}")
    print(f"  best possible:                     true "
          f"{R_true.max(-1).mean():11.3f}")
    print("\nREAD THIS: beta = 5 barely leaves the reference either way. Small")
    print("beta means the policy trusts the votes completely and collapses onto")
    print("their argmax. With 2000 votes that argmax is mostly right, so it")
    print("looks free. With 150 it is not, and the small-beta rows fall behind")
    print("beta = 0.5. Same trade as the KL coefficient on the RLHF page: how")
    print("much do you trust the thing you are optimising against?")


def demo_6_off_distribution(R_true, ref_logp, feats, comps):
    line("DEMO 6: the honest limitation -- votes the reference never produced")
    rng = np.random.default_rng(5)
    far_logp = long_winded_policy(feats)
    far = collect_comparisons(2000, far_logp, R_true, rng)
    on_p = np.exp(ref_logp)
    seen_mass = np.mean([on_p[p, w] + on_p[p, l] for p, w, l in far]) / 2
    print("the derivation assumed the comparisons came from the reference.")
    print("break that: sample the pairs from a long-winded policy instead.")
    print(f"average reference probability of an answer in those pairs: "
          f"{seen_mass:.4f}")
    print(f"same for the on-policy pairs:                              "
          f"{np.mean([on_p[p, w] + on_p[p, l] for p, w, l in comps]) / 2:.4f}\n")
    on = train_dpo(comps, ref_logp)
    off = train_dpo(far, ref_logp)
    print("  preference pairs drawn from    true reward    KL to ref")
    print(f"  the reference (as derived)     {true_score(on, R_true):10.3f}   "
          f"{kl_to_reference(on, ref_logp):10.3f}")
    print(f"  a far-off policy               {true_score(off, R_true):10.3f}   "
          f"{kl_to_reference(off, ref_logp):10.3f}")
    print(f"  no training at all (reference) {true_score(ref_logp, R_true):10.3f}"
          f"   {kl_to_reference(ref_logp, ref_logp):10.3f}")
    print("\nREAD THIS: DPO only gets gradient on pairs it is shown. Shown only")
    print("long answers, it happily raises the best of a bad bunch and lets the")
    print("good short answers lose mass by normalisation. Nothing warns you.")
    print("This is why people still collect on-policy preference data, and why")
    print("'DPO or PPO' is still argued about rather than settled.")


def demo_7_the_breaks(R_true, ref_logp, comps, steps=4000):
    line("DEMO 7: four more ways to break it, side by side")
    print(f"all runs {steps} steps, so drift has time to show.\n")
    base = train_dpo(comps, ref_logp, steps=steps)
    print(f"correct DPO:              true {true_score(base, R_true):6.3f}   KL "
          f"{kl_to_reference(base, ref_logp):7.3f}   implicit-reward corr "
          f"{spearman(implicit_reward(base, ref_logp, BETA), R_true):+.3f}")

    nr = train_dpo(comps, ref_logp, drop_reference=True, steps=steps)
    print(f"no reference in the loss: true {true_score(nr, R_true):6.3f}   KL "
          f"{kl_to_reference(nr, ref_logp):7.3f}   implicit-reward corr "
          f"{spearman(implicit_reward(nr, ref_logp, BETA), R_true):+.3f}")

    sw = train_dpo(comps, ref_logp, swap=True, steps=steps)
    print(f"winner/loser swapped:     true {true_score(sw, R_true):6.3f}   KL "
          f"{kl_to_reference(sw, ref_logp):7.3f}   implicit-reward corr "
          f"{spearman(implicit_reward(sw, ref_logp, BETA), R_true):+.3f}")

    np_ = train_dpo(comps, ref_logp, no_pair=True, steps=steps)
    print(f"winners only, no pairs:   true {true_score(np_, R_true):6.3f}   KL "
          f"{kl_to_reference(np_, ref_logp):7.3f}   implicit-reward corr "
          f"{spearman(implicit_reward(np_, ref_logp, BETA), R_true):+.3f}")

    z = train_dpo(comps, ref_logp, beta=0.0)
    print(f"\nbeta = 0 exactly:   max |pi - ref| = "
          f"{np.abs(np.exp(z) - np.exp(ref_logp)).max():.2e}  (the policy "
          f"never moved)")
    tiny = train_dpo(comps, ref_logp, beta=0.01)
    print(f"beta = 0.01:        KL {kl_to_reference(tiny, ref_logp):.3f}   true "
          f"{true_score(tiny, R_true):.3f}   max prob "
          f"{np.exp(tiny).max():.3f}")
    print("\nREAD THIS: drop log ref and the loss still trains -- it just stops")
    print("being DPO. Without the reference terms the margin is plain ranking")
    print("of log-probs, the fixed point is no longer the KL-constrained")
    print("optimum, the policy drifts further (KL 1.183 -> 1.308) and the")
    print("recovered reward degrades (0.763 -> 0.698). In a real LM the")
    print("reference is the only thing keeping the text fluent, so this is how")
    print("you get gibberish that still wins comparisons. Swap the labels and")
    print("it learns the preference backwards, correlation goes negative.")
    print("Train on single winners and there is nothing to contrast against:")
    print("it degenerates into imitating whatever the sampler produced, and")
    print("barely improves on the reference at all. And beta = 0 is")
    print("NOT the RLHF page's unleashed run: beta multiplies the margin, so")
    print("at zero the gradient dies and the policy never moves at all. The")
    print("unleashed behaviour is the small-beta end of demo 5.")


if __name__ == "__main__":
    feats, seen, R_true, ref_logp = build_world()
    closed = demo_1_closed_form(R_true, ref_logp)
    demo_2_inversion(R_true, ref_logp, closed)
    comps, dpo_logp = demo_3_dpo_vs_rlhf(feats, seen, R_true, ref_logp)
    demo_4_implicit_reward(R_true, ref_logp, dpo_logp)
    thin = collect_comparisons(150, ref_logp, R_true, np.random.default_rng(3))
    demo_5_beta_sweep(R_true, ref_logp, comps, thin)
    demo_6_off_distribution(R_true, ref_logp, feats, comps)
    demo_7_the_breaks(R_true, ref_logp, comps)

    line("THE WHOLE IDEA, COMPRESSED")
    print("""
  1. RLHF's objective, E[r] - beta*KL, has an exact maximiser:
         pi*(y) = ref(y) * exp(r(y)/beta) / Z(x)
  2. rearrange it:  r(y) = beta*log(pi*(y)/ref(y)) + beta*log Z(x)
  3. in a PAIRWISE comparison the log Z term cancels
  4. so substitute (2) into Bradley-Terry and the reward model disappears
  5. what's left is -log sigmoid(beta*logratio(win) - beta*logratio(lose)):
     a classification loss on pairs. No reward model. No RL. No sampling.

  Still needed: the reference model in memory, and preference data that
  looks like what the reference would say. That second one is the argument.
""")
