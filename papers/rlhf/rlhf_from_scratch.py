"""
Reinforcement Learning from Human Feedback -- for programmers, not researchers.

Run it:      python3 papers/rlhf/rlhf_from_scratch.py
Debug it:    set a breakpoint in train_policy and watch `true` part from `rm`.

A toy world: 4 prompts, 8 candidate answers each, and a hidden "true" reward
that the script defines but never shows the reward model. NumPy for the
mechanism (the preference model, the policy, the policy gradient). Torch only
for the one part that needs backprop through a network: fitting the reward
model. CPU, a few seconds.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(precision=2, suppress=True)

N_PROMPTS, N_RESP = 4, 8
FEATURES = ["correct", "length", "hedges", "polite"]      # the world
BETA = 0.5                       # the KL coefficient used on the happy path


def log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)          # overflow safety, cancels
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# STAGE 0 -- the toy world.
#
# A real answer is text, and the reward model reads it with a transformer.
# Here an answer is 4 numbers you can read by eye. The PROMPT is just which
# 8 candidates are on the table.
# ---------------------------------------------------------------------------
def make_world(rng):
    """(4, 8, 4): 4 prompts x 8 answers x [correct, length, hedges, polite].
    Answer 0 is a one-liner, answer 7 a wall of text, in every prompt."""
    u = lambda: rng.uniform(0, 1, (N_PROMPTS, N_RESP))
    length = np.tile(np.linspace(0.1, 1.0, N_RESP), (N_PROMPTS, 1))
    # Longer answers cover more ground: likelier right, less hedging, and
    # room for pleasantries. Every visible virtue goes up with length.
    correct = (u() < 0.15 + 0.8 * length).astype(float)
    hedges = u() ** 2 * (1.0 - 0.7 * length)
    polite = u() * (0.4 + 0.6 * length)
    return np.stack([correct, length, hedges, polite], axis=-1)


def what_the_reward_model_sees(feats, rng):
    """The reward model reads TEXT, not ground truth. Length, hedging and
    politeness sit on the surface; it reads those exactly. Whether an answer
    is RIGHT it can only guess, and it guesses wrong one time in four.
    Column 0 becomes `sounds_right` instead of `correct`."""
    seen = feats.copy()
    flip = rng.uniform(0, 1, feats.shape[:2]) < 0.25
    seen[..., 0] = np.where(flip, 1 - feats[..., 0], feats[..., 0])
    return seen


def true_reward(feats):
    """What people ACTUALLY want. The reward model never sees this function;
    it only ever sees which of two answers a person preferred. You never
    have this function in real life. That is the whole reason RLHF exists."""
    correct, length, hedges, polite = np.moveaxis(feats, -1, 0)
    return (3.0 * correct                          # being right is most of it
            + 1.0 * polite                         # manners help a little
            - 1.5 * hedges                         # "it depends..." is annoying
            - 8.0 * np.maximum(0, length - 0.6))   # past 0.6 nobody reads it


def reference_policy(feats):
    """The SFT model's habits. It copied human demonstrations, which were
    short-to-medium, so it almost never writes a long answer. Log-probs, (4, 8)."""
    length = feats[..., 1]
    return log_softmax(-8.0 * np.maximum(0, length - 0.55))


# ---------------------------------------------------------------------------
# STAGE 1 -- ask people. Not "score this 1 to 10". Just "which of these two?"
#
# Sample two different answers to the same prompt FROM THE REFERENCE POLICY
# (that is where answers come from in production), show both to a person,
# record the winner. The person is noisy in the Bradley-Terry way:
# P(prefers A) = sigmoid(true_A - true_B). A close call is nearly a coin flip.
# ---------------------------------------------------------------------------
def ask_a_person(ref_logp, R_true, rng):
    p = rng.integers(N_PROMPTS)
    a, b = rng.choice(N_RESP, size=2, replace=False, p=np.exp(ref_logp[p]))
    p_a_wins = sigmoid(R_true[p, a] - R_true[p, b])
    win = a if rng.uniform() < p_a_wins else b
    return p, a, b, p_a_wins, win


def collect_comparisons(n, ref_logp, R_true, rng):
    out = []
    for _ in range(n):
        p, a, b, _, win = ask_a_person(ref_logp, R_true, rng)
        out.append((p, win, a + b - win))          # (prompt, winner, loser)
    return out


# ---------------------------------------------------------------------------
# STAGE 2 -- the reward model, and the Bradley-Terry loss.
#
#     loss = -log sigmoid( r(winner) - r(loser) )
#
# That is the ONLY signal. No scores, no rubric, no "helpfulness" spec.
# Read it as: push the winner's score above the loser's, by a margin, and
# stop caring once the margin is comfortably positive. Christiano et al.
# section 2.2 ("fitting the reward function"); Stiennon et al. use the same
# equation in their method section.
# ---------------------------------------------------------------------------
class RewardModel(nn.Module):
    """4 features -> 16 hidden -> 1 scalar. In the papers this is the language
    model itself with its output layer swapped for a single unit."""
    def __init__(self, n_features=4, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def bradley_terry_loss(r_win, r_lose):
    return -F.logsigmoid(r_win - r_lose).mean()


def train_reward_model(comparisons, feats, ref_logp, steps=400, lr=0.02,
                       log_every=0):
    """Fit the reward model. Returns its FROZEN score table, shape (4, 8): one
    number per (prompt, answer). That table is all the policy ever sees."""
    torch.manual_seed(4)      # same init every call. Try 0..4: the blind spot
    rm = RewardModel()        # in demo 4 moves around, it never goes away.
    opt = torch.optim.Adam(rm.parameters(), lr=lr)
    X = torch.tensor(feats, dtype=torch.float32)
    p, w, l = (torch.tensor([c[i] for c in comparisons]) for i in range(3))
    for step in range(1, steps + 1):
        loss = bradley_terry_loss(rm(X[p, w]), rm(X[p, l]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step == 1 or step % log_every == 0):
            print(f"  step {step:4d}   loss {loss.item():.4f}")
    with torch.no_grad():
        table = rm(X).numpy()
    # Normalise so answers the REFERENCE writes score mean 0, std 1. Stiennon
    # et al. do the mean (reference summaries score 0); the std is mine, so
    # that beta means the same thing for every reward model in this file.
    ref = np.exp(ref_logp)
    mean = (ref * table).sum()  / N_PROMPTS
    std = np.sqrt((ref * (table - mean) ** 2).sum() / N_PROMPTS)
    return (table - mean) / std


def spearman(a, b):
    """Rank correlation: +1 same order, -1 reversed. Ranks, not values,
    because a reward model's scale is its own; only the ORDER is checkable."""
    ra = np.argsort(np.argsort(a.ravel()))
    rb = np.argsort(np.argsort(b.ravel()))
    return np.corrcoef(ra, rb)[0, 1]


# ---------------------------------------------------------------------------
# STAGE 3 -- optimise the policy against the reward model, on a leash.
#
# Policy = softmax over the 8 answers, one row of logits per prompt, and it
# STARTS AS the reference. Per sampled answer the reward is
#
#     R = rm(answer) - beta * ( log pi(answer) - log ref(answer) )
#
# The second term is the KL penalty, folded into the reward exactly as
# InstructGPT does it. beta is the leash length.
#
# The papers use PPO. This is REINFORCE: the same objective with the plainest
# policy gradient there is. The gradient of log softmax with respect to the
# logits is onehot(answer) - probs. That's the whole policy gradient. The
# optimiser is Adam, as in the papers; it matters here (see demo 6).
# ---------------------------------------------------------------------------
def kl_to_reference(logp, ref_logp):
    """Exact KL(policy || reference), averaged over prompts. Exact because
    there are 8 answers; with a real LM you estimate it from samples."""
    return (np.exp(logp) * (logp - ref_logp)).sum(-1).mean()


def adam_step(grad, state, lr, t, b1=0.9, b2=0.999):
    """One Adam update. state = [first moment, second moment], updated in place."""
    state[0] = b1 * state[0] + (1 - b1) * grad
    state[1] = b2 * state[1] + (1 - b2) * grad ** 2
    m_hat, v_hat = state[0] / (1 - b1 ** t), state[1] / (1 - b2 ** t)
    return lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def train_policy(rm_table, R_true, ref_logp, beta, steps=300, lr=0.05,
                 batch=64, log_every=25, verbose=False, moving_reference=False,
                 plain_sgd=False, seed=0):
    rng = np.random.default_rng(seed)
    logits = ref_logp.copy()                         # start AT the reference
    penalty_ref = ref_logp
    state = [np.zeros_like(logits), np.zeros_like(logits)]
    history = []
    for step in range(0, steps + 1):
        logp = log_softmax(logits)
        probs = np.exp(logp)
        if moving_reference:
            penalty_ref = logp.copy()          # the break: leash tied to self
        if step % log_every == 0:
            rm_now = (probs * rm_table).sum(-1).mean()        # optimiser's view
            true_now = (probs * R_true).sum(-1).mean()        # what you never see
            row = (step, rm_now, true_now, kl_to_reference(logp, ref_logp))
            history.append(row)
            if verbose:
                print("  step {:4d}   rm {:6.3f}   true {:6.3f}   KL {:6.3f}"
                      .format(*row))
        grad = np.zeros_like(logits)
        for p in range(N_PROMPTS):
            a = rng.choice(N_RESP, size=batch, p=probs[p])    # sample answers
            R = rm_table[p, a] - beta * (logp[p, a] - penalty_ref[p, a])
            adv = R - R.mean()                                        # baseline
            grad[p] = (adv[:, None] * (np.eye(N_RESP)[a] - probs[p])).mean(0)
        if plain_sgd:
            logits += lr * grad                                       # the break
        else:
            logits += adam_step(grad, state, lr, step + 1)            # ascent
    return np.exp(log_softmax(logits)), history


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def build_world():
    rng = np.random.default_rng(0)
    feats = make_world(rng)
    seen = what_the_reward_model_sees(feats, rng)
    return feats, seen, true_reward(feats), reference_policy(feats)


def show_prompt(p, feats, seen, R_true, ref_logp, rm_table=None):
    print("  ans  correct sounds length hedges polite   TRUE   ref%" +
          ("      rm" if rm_table is not None else ""))
    for a in range(N_RESP):
        c, ln, h, po = feats[p, a]
        row = (f"  {a:>3}  {c:7.0f} {seen[p, a, 0]:6.0f} {ln:6.2f} {h:6.2f} {po:6.2f}"
               f"  {R_true[p, a]:5.2f}  {100 * np.exp(ref_logp[p, a]):4.1f}%")
        if rm_table is not None:
            row += f"  {rm_table[p, a]:6.2f}"
        print(row)


def where_did_it_go(probs, R_true):
    print("  prompt   policy's pick   its TRUE   truly best   its TRUE")
    for p in range(N_PROMPTS):
        pick, best = int(probs[p].argmax()), int(R_true[p].argmax())
        print(f"  {p:>6}   {pick:>6} ({100 * probs[p, pick]:3.0f}%)   {R_true[p, pick]:8.2f}"
              f"   {best:>10}   {R_true[p, best]:8.2f}")


def demo_1_ask_the_humans():
    line("DEMO 1: you cannot write the loss, but you can ask 'which is better?'")
    feats, seen, R_true, ref_logp = build_world()
    print("prompt 0 -- its 8 candidate answers, the HIDDEN true reward, and how")
    print("often the reference (SFT) policy would actually write each one.")
    print("'sounds' is the reward model's guess at 'correct'; it can't see the truth.\n")
    show_prompt(0, feats, seen, R_true, ref_logp)
    print("\nanswers 5, 6, 7 are long. The reference rarely writes them.")

    rng = np.random.default_rng(1)
    print("\nsix comparisons, as the reward model will see them:")
    print("  prompt  A  B   true_A  true_B   P(A wins)   label")
    for _ in range(6):
        p, a, b, pw, win = ask_a_person(ref_logp, R_true, rng)
        print(f"  {p:>6} {a:>2} {b:>2}   {R_true[p, a]:6.2f}  {R_true[p, b]:6.2f}"
              f"   {pw:9.2f}   {win}")
    print("\nREAD THIS: the label is one bit. No '7/10', no rubric. And the")
    print("person is noisy exactly where the two answers are close, which is")
    print("what the Bradley-Terry model assumes. Nothing in the data says WHY")
    print("one won. The reward model has to guess the why from what it can see.")
    return feats, seen, R_true, ref_logp


def times_shown(comps, answer):
    """How many times a given answer index appeared in the comparisons."""
    return sum((a == answer) + (b == answer) for _, a, b in comps)


def demo_2_reward_model(feats, seen, R_true, ref_logp):
    line("DEMO 2: fit the reward model with the Bradley-Terry loss")
    rng = np.random.default_rng(2)
    comps = collect_comparisons(2000, ref_logp, R_true, rng)
    print(f"2000 comparisons = 4000 answers shown; the wall of text (answer 7)")
    print(f"was one of them {times_shown(comps, 7)} times. Adam, full batch:\n")
    rm_good = train_reward_model(comps, seen, ref_logp, log_every=100)
    print(f"\nrank correlation (Spearman) learned vs true, all 32 answers: "
          f"{spearman(rm_good, R_true):.3f}")
    print("\nprompt 0 again, with the reward model's score alongside the truth:\n")
    show_prompt(0, feats, seen, R_true, ref_logp, rm_good)
    print("\nREAD THIS: the model never saw a single true reward. It saw 2000")
    print("bits of 'this beat that' and reverse-engineered the ordering. Its")
    print("scale is its own (normalised so reference answers sit at mean 0);")
    print("only the ORDER is comparable with the truth.")
    return rm_good


def demo_3_policy_with_kl(feats, seen, R_true, ref_logp, rm_good):
    line(f"DEMO 3: optimise the policy against the reward model, beta = {BETA}")
    print("REINFORCE, 64 samples per prompt per step, KL penalty to the reference.")
    print("'rm' is what the optimiser sees. 'true' is what you never get to see.\n")
    probs, _ = train_policy(rm_good, R_true, ref_logp, beta=BETA,
                            log_every=50, verbose=True)
    print()
    where_did_it_go(probs, R_true)
    _, h0 = train_policy(rm_good, R_true, ref_logp, beta=0.0)
    peak = max(h0, key=lambda r: r[2])
    print(f"\nsame reward model, beta = 0: true reward peaked at step {peak[0]}")
    print(f"({peak[2]:.3f}), then ended at {h0[-1][2]:.3f}")
    print("\nREAD THIS: rm and true rise together and the KL stays modest. This")
    print("is the happy path: a decent reward model, a leash, and the policy")
    print("moves toward answers people actually prefer. Drop the leash and even")
    print("this reward model gets exploited, a little. Now watch a thin one.")


def demo_4_reward_hacking(feats, seen, R_true, ref_logp):
    line("DEMO 4: reward hacking on purpose -- 150 comparisons, beta = 0")
    rng = np.random.default_rng(3)
    comps = collect_comparisons(150, ref_logp, R_true, rng)
    print(f"150 comparisons = 300 answers shown; the wall of text was one of them "
          f"{times_shown(comps, 7)} times")
    rm_thin = train_reward_model(comps, seen, ref_logp)
    print(f"rank correlation learned vs true: {spearman(rm_thin, R_true):.3f} overall,"
          f" {spearman(rm_thin[:, :5], R_true[:, :5]):.3f} on the short answers 0-4\n")
    probs, hist = train_policy(rm_thin, R_true, ref_logp, beta=0.0,
                               log_every=25, verbose=True)
    best = max(hist, key=lambda r: r[2])
    print(f"\ntrue reward peaked at step {best[0]} ({best[2]:.3f}) and ended at "
          f"{hist[-1][2]:.3f}.")
    print(f"rm never stopped climbing: {hist[-1][1]:.3f}\n")
    where_did_it_go(probs, R_true)
    worst = int(np.argmin([R_true[p, probs[p].argmax()] - R_true[p].max()
                           for p in range(N_PROMPTS)]))
    print(f"\nprompt {worst}, the one that went furthest wrong, with the thin")
    print("reward model's opinion of each answer:\n")
    show_prompt(worst, feats, seen, R_true, ref_logp, rm_thin)
    gap = rm_thin[worst, 7] - rm_thin[worst, 6]
    print(f"\nanswers 6 and 7: a hair apart to a person, {gap:.1f} points apart to")
    print("the reward model.")
    print("\nREAD THIS: the thin reward model is fine where it has data. Off its")
    print("data it is guessing: it almost never saw a wall of text, because the")
    print("reference almost never wrote one, so its score there is whatever the")
    print("network happens to extrapolate. The policy, with no leash, went")
    print("looking for the highest rm score, and the highest rm score is where")
    print("the guess landed high. Goodhart's law, in a table. Note the peak:")
    print("at step 75 it was doing well.")
    return rm_thin


def demo_5_kl_sweep(feats, seen, R_true, ref_logp, rm_thin):
    line("DEMO 5: the leash length -- same thin reward model, three betas")
    print("  beta    final rm   final true   final KL")
    for beta in (0.0, BETA, 2.0):
        _, hist = train_policy(rm_thin, R_true, ref_logp, beta=beta)
        step, rm, tr, kl = hist[-1]
        print(f"  {beta:4.1f}    {rm:8.3f}   {tr:10.3f}   {kl:8.3f}")
    ref_true = (np.exp(ref_logp) * R_true).sum(-1).mean()
    print(f"\n  the reference itself:          true {ref_true:10.3f}   KL    0.000")
    print("\nREAD THIS: beta = 0 wins on rm and loses on true. beta = 2 barely")
    print("leaves the reference. The middle is the point: improve on the")
    print("reference without leaving the region where the reward model was")
    print("trained. beta is tuned, not derived; the papers pick it to land on")
    print("a target KL.")


def demo_6_the_other_breaks(feats, seen, R_true, ref_logp, rm_thin):
    line("DEMO 6: four more ways to break it, side by side")
    rng = np.random.default_rng(2)
    comps = collect_comparisons(2000, ref_logp, R_true, rng)
    flipped = [(p, l, w) for p, w, l in comps]      # == wrong sign in the loss
    rm_wrong = train_reward_model(flipped, seen, ref_logp)
    rm_right = train_reward_model(comps, seen, ref_logp)
    print(f"wrong sign in Bradley-Terry:   rank correlation "
          f"{spearman(rm_wrong, R_true):+.3f}   (right sign: "
          f"{spearman(rm_right, R_true):+.3f})")

    _, h_fixed = train_policy(rm_thin, R_true, ref_logp, beta=BETA)
    _, h_move = train_policy(rm_thin, R_true, ref_logp, beta=BETA,
                             moving_reference=True)
    print(f"\nmoving reference, beta = {BETA}:   true {h_move[-1][2]:.3f}   KL "
          f"{h_move[-1][3]:.3f}")
    print(f"fixed reference, beta = {BETA}:    true {h_fixed[-1][2]:.3f}   KL "
          f"{h_fixed[-1][3]:.3f}")

    _, h_sgd = train_policy(rm_thin, R_true, ref_logp, beta=0.0, lr=0.3,
                            steps=3000, log_every=3000, plain_sgd=True)
    _, h_adam = train_policy(rm_thin, R_true, ref_logp, beta=0.0)
    print(f"\nplain SGD instead of Adam, beta = 0, 3000 steps:   true "
          f"{h_sgd[-1][2]:.3f}   KL {h_sgd[-1][3]:.3f}")
    print(f"Adam, beta = 0, 300 steps (demo 4 again):          true "
          f"{h_adam[-1][2]:.3f}   KL {h_adam[-1][3]:.3f}")

    _, h_oracle = train_policy(R_true, R_true, ref_logp, beta=0.0)
    print(f"\noptimise TRUE reward directly:   true {h_oracle[-1][2]:.3f}"
          f"   (best possible: {R_true.max(-1).mean():.3f})")
    print("\nREAD THIS: flip the sign and the model learns the opposite order.")
    print("Tie the leash to yourself and it reads zero while you wander off.")
    print("Plain SGD never finds the wall of text: a rare answer's probability")
    print("grows as its own square, so 3000 steps is not enough. Nothing is")
    print("fixed; the reward model is exactly as wrong, nobody asked it. A real")
    print("LM has a million ways to be longer, so it always gets asked.")
    print("And if you had the true reward you would just climb it: the whole")
    print("apparatus exists because you don't.")


if __name__ == "__main__":
    feats, seen, R_true, ref_logp = demo_1_ask_the_humans()
    rm_good = demo_2_reward_model(feats, seen, R_true, ref_logp)
    demo_3_policy_with_kl(feats, seen, R_true, ref_logp, rm_good)
    rm_thin = demo_4_reward_hacking(feats, seen, R_true, ref_logp)
    demo_5_kl_sweep(feats, seen, R_true, ref_logp, rm_thin)
    demo_6_the_other_breaks(feats, seen, R_true, ref_logp, rm_thin)

    line("THE WHOLE IDEA, COMPRESSED")
    print("""
  1. you can't write loss("helpful"), but people can pick A or B
  2. reward model = fit r(x) so sigmoid(r(win) - r(lose)) matches the picks
  3. policy       = start at the reference, climb r(x)...
  4. leash        = ...minus beta * KL(policy || reference)
  5. Goodhart     = drop the leash and the policy finds where r is wrong

  Everything else is training machinery: PPO's clipping, value heads, GAE.
  Real, and it matters for stability at scale, but not the idea.
""")
