"""
GRPO -- Group Relative Policy Optimization, for programmers, not researchers.

Run it:      python3 papers/grpo/grpo_from_scratch.py
Debug it:    breakpoint in group_advantages and print r, r.mean(), r.std().

The paper: DeepSeekMath (Shao et al., 2024, arXiv 2402.03300), section 4.
GRPO is plain policy gradient with the critic deleted. The baseline that the
critic used to predict is estimated instead from a GROUP of answers sampled
for the SAME prompt: advantage = (reward - group mean) / group std.

Toy world, same shape as the RLHF page: a handful of prompts, several candidate
answers each, and a reward. Here the reward is DIRECT (a verifier that grades
maths answers), not a learned reward model -- that is the setting GRPO was
introduced for. NumPy does the mechanism (advantages, the policy gradient,
the variance measurements). Torch does the training loops and the critic.
CPU, a few seconds.
"""

import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(precision=3, suppress=True)

N_PROMPTS, N_ANSWERS, D_EMB = 6, 12, 8
BETA = 0.02                      # KL coefficient on the happy path


def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)          # overflow safety, cancels
    return np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)


def log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


# ---------------------------------------------------------------------------
# STAGE 0 -- the toy world.
#
# 6 prompts (maths questions), 12 candidate answers each. An answer scores
# 0..10 from a verifier. Prompts differ in DIFFICULTY: prompt 0 is easy and
# almost everything scores well, prompt 5 is hard and almost nothing does.
# That difficulty spread is the whole reason the baseline must be per-prompt.
# ---------------------------------------------------------------------------
def make_world(rng):
    """(6, 12) rewards, plus (6, 8) prompt embeddings for the critic to read."""
    difficulty = np.linspace(0.0, 1.0, N_PROMPTS)[:, None]     # 0 easy, 1 hard
    quality = rng.uniform(0, 1, (N_PROMPTS, N_ANSWERS))
    rewards = 10.0 * quality * (1.0 - 0.8 * difficulty)
    emb = rng.normal(0, 1, (N_PROMPTS, D_EMB))                 # "the prompt"
    return rewards, emb


def reference_logits(rng):
    """The SFT model's habits, frozen. Not uniform: it already has opinions."""
    return rng.normal(0, 0.5, (N_PROMPTS, N_ANSWERS))


# ---------------------------------------------------------------------------
# STAGE 1 -- the policy gradient, by hand.
#
# The policy is a softmax over the 12 answers, one row of logits per prompt.
# For a sampled answer a with advantage A, the gradient of A * log pi(a) with
# respect to that prompt's logits is exactly:
#
#     A * ( onehot(a) - probs )
#
# That is the entire estimator. Everything below only changes what A is.
# ---------------------------------------------------------------------------
def policy_gradient(logits, samples, advantages):
    """samples: (P, G) sampled answer indices. advantages: (P, G) scalars."""
    probs = softmax(logits)
    grad = np.zeros_like(logits)
    G = samples.shape[1]
    for p in range(logits.shape[0]):
        for i in range(G):
            onehot = np.zeros(logits.shape[1])
            onehot[samples[p, i]] = 1.0
            grad[p] += advantages[p, i] * (onehot - probs[p]) / G
    return grad


def sample_group(logits, G, rng):
    """Sample G answers per prompt, WITH replacement -- a group. (P, G)."""
    probs = softmax(logits)
    return np.stack([rng.choice(N_ANSWERS, size=G, p=probs[p])
                     for p in range(logits.shape[0])])


# ---------------------------------------------------------------------------
# STAGE 2 -- THE WHOLE PAPER. Two lines.
#
# A critic predicts "what should this prompt score?" and you subtract it.
# GRPO says: you already sampled G answers for this prompt. Their own mean IS
# that prediction, and it costs no parameters. Divide by the group's standard
# deviation so the update size does not depend on how spread out the rewards
# happened to be. Grading on a curve within the class.
# ---------------------------------------------------------------------------
def group_advantages(r, normalise=True):
    """r: (P, G) rewards for a group. Returns (P, G) advantages."""
    adv = r - r.mean(axis=1, keepdims=True)               # the group baseline
    if normalise:
        adv = adv / (r.std(axis=1, keepdims=True) + 1e-8)  # scale invariance
    return adv


def batch_advantages(r, normalise=True):
    """The WRONG baseline: one mean over the whole batch, ignoring prompts."""
    adv = r - r.mean()
    if normalise:
        adv = adv / (r.std() + 1e-8)
    return adv


# ---------------------------------------------------------------------------
# STAGE 3 -- the critic GRPO deletes. A small MLP: prompt embedding -> value.
# In a real run this is a second network the size of the policy. Here it is
# 161 parameters against the policy's 72, which already tells the story.
# ---------------------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_EMB, 16), nn.Tanh(),
                                 nn.Linear(16, 1))

    def forward(self, emb):
        return self.net(emb).squeeze(-1)


def n_params(module):
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# STAGE 4 -- the training loop. Torch only because Adam and .backward() are
# not the idea. The GRPO loss, with the KL term IN THE LOSS (not folded into
# the reward, which is where RLHF/PPO put it):
#
#     loss = -mean( A * log pi(a) ) + beta * KL(pi || pi_ref)
# ---------------------------------------------------------------------------
def train(rewards, emb, ref_logits, rng, steps=300, G=8, beta=BETA,
          baseline="group", normalise=True, use_critic=False,
          train_critic=True, lr=0.05):
    logits = torch.tensor(ref_logits.copy(), requires_grad=True)
    critic = Critic()
    opt = torch.optim.Adam([logits], lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=0.05)
    ref = torch.tensor(ref_logits)
    emb_t = torch.tensor(emb, dtype=torch.float32)

    for _ in range(steps):
        lp = torch.log_softmax(logits, dim=-1)
        samples = sample_group(logits.detach().numpy(), G, rng)
        r = np.take_along_axis(rewards, samples, axis=1)          # (P, G)

        if use_critic:
            v = critic(emb_t)                                     # (P,)
            adv_t = torch.tensor(r, dtype=torch.float32) - v.detach()[:, None]
            if train_critic:
                loss_v = ((v - torch.tensor(r.mean(1), dtype=torch.float32))
                          ** 2).mean()
                opt_c.zero_grad(); loss_v.backward(); opt_c.step()
        else:
            adv = (group_advantages(r, normalise) if baseline == "group"
                   else batch_advantages(r, normalise) if baseline == "batch"
                   else r)                                        # no baseline
            adv_t = torch.tensor(adv, dtype=torch.float32)

        chosen = lp.gather(1, torch.tensor(samples))              # (P, G)
        pg = -(adv_t * chosen).mean()
        kl = (lp.exp() * (lp - torch.log_softmax(ref, dim=-1))).sum(-1).mean()
        loss = pg + beta * kl                                     # KL in LOSS

        opt.zero_grad(); loss.backward(); opt.step()

    return logits.detach().numpy(), critic


def evaluate(logits, rewards, ref_logits):
    """Expected true reward under the policy, and its KL from the reference."""
    p = softmax(logits)
    true = float((p * rewards).sum(axis=1).mean())
    kl = float((p * (log_softmax(logits) - log_softmax(ref_logits)))
               .sum(axis=1).mean())
    return true, kl


def grad_variance(logits, rewards, rng, G=8, kind="group", repeats=400,
                  critic_value=None):
    """E||g - E g||^2 over `repeats` independent gradient estimates."""
    grads = []
    for _ in range(repeats):
        s = sample_group(logits, G, rng)
        r = np.take_along_axis(rewards, s, axis=1)
        if kind == "none":
            a = r
        elif kind == "group":
            a = group_advantages(r, normalise=False)
        elif kind == "group_norm":
            a = group_advantages(r, normalise=True)
        elif kind == "batch":
            a = batch_advantages(r, normalise=False)
        elif kind == "critic":
            a = r - critic_value[:, None]
        grads.append(policy_gradient(logits, s, a))
    g = np.stack(grads)
    return float(((g - g.mean(0)) ** 2).sum(axis=(1, 2)).mean())


# ===========================================================================
# DEMOS
# ===========================================================================
def demo_1_why_a_baseline():
    line("DEMO 1: why a baseline exists at all")
    rng = np.random.default_rng(0)
    rewards, _ = make_world(rng)
    ref = reference_logits(rng)

    print("every reward in this world is POSITIVE (a verifier score, 0..10):")
    print("  min %.3f   max %.3f" % (rewards.min(), rewards.max()))
    print("\nso with no baseline, every sampled answer gets pushed UP. The")
    print("signal 'this one was better than the others' is buried in a")
    print("constant that is the same for good and bad answers alike.\n")

    v_none = grad_variance(ref, rewards, np.random.default_rng(1), kind="none")
    v_grp = grad_variance(ref, rewards, np.random.default_rng(1), kind="group")
    print("gradient variance, G=8, 400 independent estimates:")
    print("  no baseline           E||g - Eg||^2 = %9.4f" % v_none)
    print("  group-mean baseline   E||g - Eg||^2 = %9.4f" % v_grp)
    print("  reduction: %.1fx" % (v_none / v_grp))

    print("\nboth estimators point the same way on average:")
    print("  cosine(Eg_none, Eg_group) = %.4f" % cos_mean_grad(ref, rewards))
    print("\nREAD THIS: the baseline changes NOTHING in expectation -- it is")
    print("subtracting a constant that the gradient of a probability")
    print("distribution integrates to zero against. All it buys is variance.")
    print("That is the whole job. Cheaper variance is the whole of GRPO.")


def cos_mean_grad(ref, rewards, repeats=400, G=8):
    """Mean gradient with and without a baseline, compared by direction."""
    def mean_g(kind):
        rng = np.random.default_rng(7)
        acc = np.zeros_like(ref)
        for _ in range(repeats):
            s = sample_group(ref, G, rng)
            r = np.take_along_axis(rewards, s, axis=1)
            a = r if kind == "none" else group_advantages(r, normalise=False)
            acc += policy_gradient(ref, s, a)
        return (acc / repeats).ravel()
    a, b = mean_g("none"), mean_g("group")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def demo_2_worked_group():
    line("DEMO 2: one group, by hand")
    rng = np.random.default_rng(0)
    rewards, _ = make_world(rng)
    ref = reference_logits(rng)

    s = sample_group(ref, 8, np.random.default_rng(3))
    p = 2
    r = rewards[p][s[p]]
    print("prompt 2, G=8 sampled answers:", s[p])
    print("rewards            :", np.round(r, 3))
    print("group mean         : %.4f" % r.mean())
    print("group std          : %.4f" % r.std())
    adv = (r - r.mean()) / (r.std() + 1e-8)
    print("advantage = (r - mean) / std :")
    for i in range(8):
        print("   answer %2d  r %6.3f  ->  A %+7.4f" % (s[p, i], r[i], adv[i]))
    print("sum of advantages  : %+.6f   (zero, by construction)" % adv.sum())
    print("\nREAD THIS: no critic, no second network, no extra forward pass.")
    print("The baseline is the mean of numbers you already had to compute.")


def demo_3_critic_vs_no_critic():
    line("DEMO 3: the critic, and doing without it")
    rng = np.random.default_rng(0)
    rewards, emb = make_world(rng)
    ref = reference_logits(rng)

    # A perfectly-fit critic knows each prompt's expected reward exactly.
    fitted = (softmax(ref) * rewards).sum(axis=1)
    v_none = grad_variance(ref, rewards, np.random.default_rng(1), kind="none")
    v_crit = grad_variance(ref, rewards, np.random.default_rng(1),
                           kind="critic", critic_value=fitted)
    v_grp = grad_variance(ref, rewards, np.random.default_rng(1), kind="group")

    print("gradient variance, G=8, 400 estimates:")
    print("  no baseline                    %9.4f" % v_none)
    print("  fitted value baseline (PPO)    %9.4f" % v_crit)
    print("  group-mean baseline (GRPO)     %9.4f" % v_grp)

    critic = Critic()
    print("\nparameters:")
    print("  policy (6 prompts x 12 logits)   %4d" % (N_PROMPTS * N_ANSWERS))
    print("  critic MLP (8 -> 16 -> 1)        %4d" % n_params(critic))
    print("  group baseline                   %4d   <- the point" % 0)

    a, _ = train(rewards, emb, ref, np.random.default_rng(11), use_critic=True)
    b, _ = train(rewards, emb, ref, np.random.default_rng(11),
                 baseline="group")
    ta, ka = evaluate(a, rewards, ref)
    tb, kb = evaluate(b, rewards, ref)
    t0, _ = evaluate(ref, rewards, ref)
    print("\n300 steps, G=8, beta=0.02:")
    print("  reference policy         true reward %.4f" % t0)
    print("  with critic              true reward %.4f   KL %.4f" % (ta, ka))
    print("  group baseline           true reward %.4f   KL %.4f" % (tb, kb))
    print("\nREAD THIS: same destination, and one of the two needed a whole")
    print("second network to get there. At 7B parameters that second network")
    print("is 7B more parameters of optimiser state you have to hold in VRAM.")


def demo_4_group_size():
    line("DEMO 4: group size G -- the compute-versus-variance trade")
    rng = np.random.default_rng(0)
    rewards, emb = make_world(rng)
    ref = reference_logits(rng)

    print("  G   grad variance   final true reward   note")
    for G in [1, 2, 4, 8, 16, 32]:
        v = grad_variance(ref, rewards, np.random.default_rng(1), G=G,
                          kind="group_norm")
        pol, _ = train(rewards, emb, ref, np.random.default_rng(11), G=G)
        t, _ = evaluate(pol, rewards, ref)
        note = "advantage is identically 0 -- nothing trains" if G == 1 else ""
        print(" %3d   %13.4f   %17.4f   %s" % (G, v, t, note))

    t0, _ = evaluate(ref, rewards, ref)
    print("\n  reference policy, untrained:  true reward %.4f" % t0)
    print("\nREAD THIS: G=1 means the group mean IS the one reward, so every")
    print("advantage is exactly zero and the policy never moves off the")
    print("reference. G=2 already works. Past G=8 the variance keeps falling")
    print("but you are paying G forward passes per prompt for it.")


def demo_5_the_kl_term():
    line("DEMO 5: the KL term -- the same leash, in a different place")
    rng = np.random.default_rng(0)
    rewards, emb = make_world(rng)
    ref = reference_logits(rng)

    print("  beta    final true reward   final KL(pi || pi_ref)")
    for beta in [0.0, 0.02, 0.1, 0.5, 2.0]:
        pol, _ = train(rewards, emb, ref, np.random.default_rng(11),
                       beta=beta)
        t, k = evaluate(pol, rewards, ref)
        print("  %-6.2f  %17.4f   %20.4f" % (beta, t, k))

    print("\nGRPO puts beta * KL as a TERM IN THE LOSS. RLHF/PPO fold it into")
    print("the per-sample reward instead. Same leash, different attachment:")
    for beta in [0.1, 0.5]:
        pol, _ = train(rewards, emb, ref, np.random.default_rng(11),
                       beta=beta)
        t1, k1 = evaluate(pol, rewards, ref)
        pol2 = train_kl_in_reward(rewards, ref, np.random.default_rng(11),
                                  beta=beta)
        t2, k2 = evaluate(pol2, rewards, ref)
        print("  beta %.1f  KL in loss   true %.4f  KL %.4f" % (beta, t1, k1))
        print("  beta %.1f  KL in reward true %.4f  KL %.4f" % (beta, t2, k2))

    print("\nREAD THIS: folding the KL into the reward pushes it through the")
    print("advantage -- it gets group-mean-subtracted and std-divided along")
    print("with everything else, so the leash length you asked for is not the")
    print("leash length you get. As a loss term its gradient is exact and")
    print("beta means what it says. Look at the KL columns above.")


def train_kl_in_reward(rewards, ref_logits, rng, steps=300, G=8, beta=0.1,
                       lr=0.05):
    """The RLHF/PPO placement: subtract beta*(log pi - log pi_ref) from the
    per-sample reward, BEFORE the group baseline touches it."""
    logits = torch.tensor(ref_logits.copy(), requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    ref_lp = log_softmax(ref_logits)
    for _ in range(steps):
        lp_np = log_softmax(logits.detach().numpy())
        s = sample_group(logits.detach().numpy(), G, rng)
        r = np.take_along_axis(rewards, s, axis=1)
        pen = np.take_along_axis(lp_np - ref_lp, s, axis=1)
        adv = group_advantages(r - beta * pen, normalise=True)
        lp = torch.log_softmax(logits, dim=-1)
        chosen = lp.gather(1, torch.tensor(s))
        loss = -(torch.tensor(adv, dtype=torch.float32) * chosen).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return logits.detach().numpy()


def demo_6_wrong_baseline():
    line("DEMO 6: within-prompt versus across-prompt baselines")
    rng = np.random.default_rng(0)
    rewards, emb = make_world(rng)
    ref = reference_logits(rng)

    s = sample_group(ref, 8, np.random.default_rng(3))
    r = np.take_along_axis(rewards, s, axis=1)
    a_grp = group_advantages(r, normalise=False)
    a_bat = batch_advantages(r, normalise=False)
    print("mean advantage per prompt, one batch (easy -> hard):")
    print(" prompt  mean reward   group baseline   batch baseline")
    for p in range(N_PROMPTS):
        print("   %d     %10.3f   %14.3f   %14.3f"
              % (p, r[p].mean(), a_grp[p].mean(), a_bat[p].mean()))

    v_grp = grad_variance(ref, rewards, np.random.default_rng(1), kind="group")
    v_bat = grad_variance(ref, rewards, np.random.default_rng(1), kind="batch")
    print("\ngradient variance   group %.4f   batch %.4f" % (v_grp, v_bat))

    g, _ = train(rewards, emb, ref, np.random.default_rng(11),
                 baseline="group")
    b, _ = train(rewards, emb, ref, np.random.default_rng(11),
                 baseline="batch")
    print("\nfinal true reward per prompt after 300 steps:")
    print(" prompt   best possible   group baseline   batch baseline")
    pg, pb = softmax(g), softmax(b)
    for p in range(N_PROMPTS):
        print("   %d      %13.3f   %14.3f   %14.3f"
              % (p, rewards[p].max(), (pg[p] * rewards[p]).sum(),
                 (pb[p] * rewards[p]).sum()))
    tg, _ = evaluate(g, rewards, ref)
    tb, _ = evaluate(b, rewards, ref)
    print("  overall %.4f (group)  vs  %.4f (batch)" % (tg, tb))

    print("\nREAD THIS: with one mean over the whole batch, the advantage for")
    print("a hard prompt is negative for EVERY answer it sampled, good ones")
    print("included -- the number measures the prompt's difficulty, not the")
    print("answer's quality. Probability drains off everything that was")
    print("sampled and onto whatever was not. The baseline must be per-prompt.")


def demo_7_break_it():
    line("DEMO 7: three deliberate breaks, side by side")
    rng = np.random.default_rng(0)
    rewards, emb = make_world(rng)
    ref = reference_logits(rng)
    t0, _ = evaluate(ref, rewards, ref)

    print("BREAK 1 -- G=1: the group mean is the sample, advantage == 0")
    s = sample_group(ref, 1, np.random.default_rng(3))
    r = np.take_along_axis(rewards, s, axis=1)
    print("  advantages, G=1 :", group_advantages(r).ravel())
    pol, _ = train(rewards, emb, ref, np.random.default_rng(11), G=1)
    t, _ = evaluate(pol, rewards, ref)
    print("  true reward after 300 steps: %.4f  (reference %.4f)" % (t, t0))
    pol, _ = train(rewards, emb, ref, np.random.default_rng(11), G=8)
    t, _ = evaluate(pol, rewards, ref)
    print("  same run with G=8          : %.4f" % t)

    print("\nBREAK 2 -- drop the /std: update size follows reward spread")
    for G in [8]:
        s = sample_group(ref, G, np.random.default_rng(3))
        r = np.take_along_axis(rewards, s, axis=1)
        gn = policy_gradient(ref, s, group_advantages(r, normalise=False))
        gy = policy_gradient(ref, s, group_advantages(r, normalise=True))
        print("  easy prompt 0 group std %.3f  hard prompt 5 group std %.3f"
              % (r[0].std(), r[5].std()))
        print("  ||grad row|| unnormalised   prompt 0 %.4f   prompt 5 %.4f"
              % (np.linalg.norm(gn[0]), np.linalg.norm(gn[5])))
        print("  ||grad row|| normalised     prompt 0 %.4f   prompt 5 %.4f"
              % (np.linalg.norm(gy[0]), np.linalg.norm(gy[5])))

    print("\nBREAK 3 -- keep the critic, never train it (a constant baseline)")
    zero = np.zeros(N_PROMPTS)
    v_dead = grad_variance(ref, rewards, np.random.default_rng(1),
                           kind="critic", critic_value=zero)
    fitted = (softmax(ref) * rewards).sum(axis=1)
    v_live = grad_variance(ref, rewards, np.random.default_rng(1),
                           kind="critic", critic_value=fitted)
    v_grp = grad_variance(ref, rewards, np.random.default_rng(1), kind="group")
    print("  untrained critic (V=0)  variance %9.4f" % v_dead)
    print("  trained critic          variance %9.4f" % v_live)
    print("  group baseline          variance %9.4f" % v_grp)

    print("\nBREAK 4 -- beta=0: no leash")
    pol, _ = train(rewards, emb, ref, np.random.default_rng(11), beta=0.0)
    t, k = evaluate(pol, rewards, ref)
    print("  beta 0.00  true %.4f  KL %.4f" % (t, k))
    pol, _ = train(rewards, emb, ref, np.random.default_rng(11), beta=0.5)
    t, k = evaluate(pol, rewards, ref)
    print("  beta 0.50  true %.4f  KL %.4f" % (t, k))
    print("  (here the reward IS the truth, so drifting costs nothing. Swap")
    print("   the verifier for a learned reward model and it costs plenty --")
    print("   that is the RLHF page.)")


if __name__ == "__main__":
    demo_1_why_a_baseline()
    demo_2_worked_group()
    demo_3_critic_vs_no_critic()
    demo_4_group_size()
    demo_5_the_kl_term()
    demo_6_wrong_baseline()
    demo_7_break_it()

    line("GRPO, COMPRESSED")
    print("""
  1. policy gradient  = A * (onehot(a) - probs), summed over samples
  2. A needs a baseline, or the constant part of the reward is pure noise
  3. PPO's baseline   = a critic, a second network the size of the policy
  4. GRPO's baseline  = the mean reward of G answers to the SAME prompt
  5. normalise        = divide by that group's std, so step size is stable
  6. the leash        = beta * KL(pi || pi_ref), as a TERM IN THE LOSS
  7. what you drop    = the critic, its optimiser state, and its VRAM

  Introduced in DeepSeekMath (arXiv 2402.03300) for maths reasoning, where
  the reward is a verifier and not a learned model. It is the algorithm
  DeepSeek-R1 (arXiv 2501.12948) was trained with.
""")
