"""
InstructGPT: Training Language Models to Follow Instructions with Human
Feedback (Ouyang et al., 2022) -- for programmers, not researchers.

Run it:      python3 instructgpt_from_scratch.py
Debug it:    set a breakpoint in rlhf() and watch one row of logits move.

A toy world: 4 prompts, 6 candidate responses each, and a hidden helpfulness
score that nothing in training ever reads. A "policy" here is a table of
logits, one row per prompt; answering = softmax the row and sample. No tokens,
no transformer. The three-stage RECIPE is the paper, and the recipe is the
same whether the policy is a 4x6 table or a 175B-parameter GPT-3.

NumPy for the world and every measurement. torch (CPU, a few hundred steps)
for the three training loops, because each stage IS a training objective.
"""

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)      # match NumPy, no silent casts
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- the toy world.
#
# Each response is (text, kind, length, hidden helpfulness).
#   kind    "cont" = continues the prompt the way web text would. GPT-3's habit.
#           "inst" = actually answers the instruction.
#   length  rough log scale: 0.1 one word, 0.5 a sentence or two, 1.0 essays.
#   hidden  what a careful human REALLY thinks of it, 0..10. The policy, the
#           reward model, nothing in training ever sees this column. It exists
#           only so we can grade the pipeline at the end.
#
# Order per prompt is fixed: three continuations, then a terse answer, a good
# answer, a padded answer. So indices 3, 4, 5 are the instruction-following
# ones, and 4 is always the one a careful human would want.
# ---------------------------------------------------------------------------
PROMPTS = [
    "Explain what a mutex is.",
    "Translate 'good morning' to French.",
    "Write a one-line Python function that reverses a string.",
    "Is 91 a prime number?",
]

RESPONSES = [
    [("Explain what a semaphore is. Explain what a deadlock is.",
      "cont", 0.45, 0.5),
     ("(10 marks) Q2. Describe a race condition.", "cont", 0.30, 0.0),
     ("I still don't get it after three lectures.", "cont", 0.35, 1.0),
     ("A lock one thread holds at a time.", "inst", 0.25, 5.0),
     ("A lock one thread holds while the rest wait; guards shared state.",
      "inst", 0.60, 8.0),
     ("Great question! A mutex, or mutual exclusion... [4 paragraphs]",
      "inst", 1.00, 3.5)],

    [("Translate 'good night' to French. Translate 'thanks' to German.",
      "cont", 0.50, 0.5),
     ("Answers are on page 212.", "cont", 0.25, 0.0),
     ("Translate 'good morning' to Spanish.", "cont", 0.35, 0.5),
     ("Bonjour.", "inst", 0.15, 5.5),
     ("Bonjour. (Informally, 'salut' works too.)", "inst", 0.50, 8.0),
     ("Certainly! 'Good morning' is 'bonjour'... [etymology, 3 caveats]",
      "inst", 1.00, 4.0)],

    [("Write a one-line Python function that sorts a list.", "cont", 0.45, 0.5),
     ("Bonus: do it without slicing.", "cont", 0.30, 0.5),
     ("Hint: strings are sequences.", "cont", 0.30, 1.5),
     ("lambda s: s[::-1]", "inst", 0.20, 4.5),
     ("def rev(s): return s[::-1]   # slice, step -1", "inst", 0.55, 8.5),
     ("Here is a complete solution with docstring, type hints... [40 lines]",
      "inst", 1.00, 3.0)],

    [("Is 97 a prime number? Is 101 a prime number?", "cont", 0.40, 0.5),
     ("Show your working. (3 marks)", "cont", 0.30, 0.0),
     ("My brother says yes but I'm not sure.", "cont", 0.35, 1.0),
     ("No.", "inst", 0.10, 4.0),
     ("No. 91 = 7 x 13.", "inst", 0.45, 8.0),
     ("Great question! To determine whether 91 is prime... [a table]",
      "inst", 1.00, 3.5)],
]

N_PROMPTS, N_RESP = len(PROMPTS), len(RESPONSES[0])
TEXT   = [[r[0] for r in row] for row in RESPONSES]
IS_INST = np.array([[r[1] == "inst" for r in row] for row in RESPONSES])
LENGTH = np.array([[r[2] for r in row] for row in RESPONSES])
HIDDEN = np.array([[r[3] for r in row] for row in RESPONSES])

# What the reward model is allowed to see about a response. In the paper the
# RM reads the tokens; here it reads three numbers. Same loss, same lesson.
FEATURES = np.stack([IS_INST.astype(float), LENGTH, LENGTH ** 2], axis=-1)

# GPT-3's prior, as logits. A continuation is e^2 = 7x more "natural" to it
# than a one-word answer, and an answer gets rarer the longer it is. Nobody
# on the web opens with "Great question!" -- that style is not in the prior.
# "Completion-shaped" means: it predicts what web text would do next.
BASE_LOGITS = np.where(IS_INST, 1.0 - 5.5 * LENGTH, 2.0)

# The labelers. In the paper: 40 contractors, ~73% inter-labeler agreement.
# Here: P(a preferred to b) = sigmoid((hidden_a - hidden_b) / LABELER_TEMP).
# The temperature is tuned so simulated labelers agree about as often as the
# real ones did. Raise it and the labelers get noisier (see the break table).
LABELER_TEMP = 2.3


# ---------------------------------------------------------------------------
# Measurements. All NumPy, all read HIDDEN, none used by training.
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def mass_on_instructions(logits):
    """Fraction of the policy's probability that lands on actual answers."""
    return float((softmax(logits) * IS_INST).sum(-1).mean())


def true_helpfulness(logits):
    """Expected hidden score of what the policy produces, averaged over prompts."""
    return float((softmax(logits) * HIDDEN).sum(-1).mean())


def kl(logits, ref_logits):
    """KL(policy || reference), averaged over prompts. 'How far did it drift.'"""
    p, q = softmax(logits), softmax(ref_logits)
    return float((p * (np.log(p) - np.log(q))).sum(-1).mean())


def favourite(logits, p):
    return TEXT[p][int(np.argmax(logits[p]))]


# ---------------------------------------------------------------------------
# STAGE 1 -- SFT (supervised fine-tuning). Section 3.5 of the paper.
#
# Labelers WRITE the answer they want (13k prompts). Train the policy to put
# probability on those. Ordinary supervised learning, cross-entropy loss --
# the same loss as pretraining, just on 13k examples of the right FORMAT.
# ---------------------------------------------------------------------------
def sft(base_logits, demos, steps=60, lr=1.0):
    """demos[p] = indices of the responses a labeler wrote for prompt p."""
    theta = torch.tensor(base_logits, requires_grad=True)
    opt = torch.optim.SGD([theta], lr=lr)
    prompts = torch.tensor([p for p, ds in enumerate(demos) for _ in ds])
    targets = torch.tensor([d for ds in demos for d in ds])
    for _ in range(steps):
        logp = torch.log_softmax(theta, dim=-1)
        loss = -logp[prompts, targets].mean()      # cross-entropy on the demos
        opt.zero_grad(); loss.backward(); opt.step()
    return theta.detach().numpy()


# ---------------------------------------------------------------------------
# STAGE 2 -- reward model from comparisons. Section 3.5 of the paper.
#
# Labelers do NOT write answers now. They see two of the model's own outputs
# and pick the better one (33k prompts). Cheaper than writing, and it captures
# a preference nobody could write a rule for. Bradley-Terry turns "a beat b"
# into a scalar score: P(a beats b) = sigmoid(r(a) - r(b)). The RLHF page at
# ../rlhf/ derives this; here it is the loss in three lines.
# ---------------------------------------------------------------------------
def labeler(p, a, b, rng):
    """One human, one pair. Returns (winner, loser). Not perfectly reliable."""
    p_a_wins = 1 / (1 + np.exp(-(HIDDEN[p, a] - HIDDEN[p, b]) / LABELER_TEMP))
    return (a, b) if rng.random() < p_a_wins else (b, a)


def collect_comparisons(sample_logits, n, rng):
    """Sample two DISTINCT responses from the given policy, ask a labeler.
    Which policy you sample from is the whole distribution-shift story."""
    probs, out = softmax(sample_logits), []
    for _ in range(n):
        p = rng.integers(N_PROMPTS)
        a, b = rng.choice(N_RESP, size=2, replace=False, p=probs[p])
        out.append((p, *labeler(p, a, b, rng)))
    return out


def train_reward_model(comparisons, steps=400, lr=0.1):
    """Linear reward on FEATURES. Returns a (prompt, response) reward table."""
    F = torch.tensor(FEATURES)
    w = torch.zeros(F.shape[-1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    ps, ws, ls = (torch.tensor(c) for c in zip(*comparisons))
    for _ in range(steps):
        r = F @ w + b                              # reward of every candidate
        margin = r[ps, ws] - r[ps, ls]             # winner minus loser
        loss = -torch.nn.functional.logsigmoid(margin).mean()   # Bradley-Terry
        opt.zero_grad(); loss.backward(); opt.step()
    return (F @ w + b).detach().numpy()


# ---------------------------------------------------------------------------
# STAGE 3 -- RL against the reward model, on a KL leash. Section 3.5.
#
#   maximise  E_pi[ r(y) ]  -  beta * KL( pi || pi_SFT )
#
# The paper does this with PPO on sampled responses (31k prompts) and a
# per-token KL penalty, beta = 0.02. The RLHF page covers the sampling and
# the clipping. Here there are six candidates per prompt, so the expectation
# is computed exactly and the gradient comes straight from autograd.
# ---------------------------------------------------------------------------
def rlhf(init_logits, ref_logits, reward, beta, steps=300, lr=0.1):
    """Returns the whole trajectory of logits, so demos can watch it move."""
    theta = torch.tensor(init_logits, requires_grad=True)
    ref_logp = torch.log_softmax(torch.tensor(ref_logits), dim=-1)
    R = torch.tensor(reward)
    opt = torch.optim.Adam([theta], lr=lr)
    history = [init_logits.copy()]
    for _ in range(steps):
        logp = torch.log_softmax(theta, dim=-1)
        p = logp.exp()
        expected_reward = (p * R).sum(-1)              # what the RM thinks
        leash = (p * (logp - ref_logp)).sum(-1)        # KL(pi || pi_ref)
        objective = (expected_reward - beta * leash).mean()
        opt.zero_grad(); (-objective).backward(); opt.step()
        history.append(theta.detach().numpy().copy())
    return np.stack(history)


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def report(name, logits):
    print(f"  {name:<16} mass on answers: {mass_on_instructions(logits):.3f}"
          f"   true helpfulness: {true_helpfulness(logits):.2f}")


def steps_to_reach(traj, target):
    for i, lg in enumerate(traj):
        if true_helpfulness(lg) >= target:
            return f"{i:>5d}"
    return "never"


def agreement(comparisons):
    """If two of our labelers saw the same pairs, how often would they agree?"""
    out = []
    for p, a, b in comparisons:
        q = 1 / (1 + np.exp(-(HIDDEN[p, a] - HIDDEN[p, b]) / LABELER_TEMP))
        out.append(q * q + (1 - q) * (1 - q))
    return float(np.mean(out))


BETA = 0.7                       # the paper uses 0.02, per token; scale differs
DEMOS = [[3, 4]] * N_PROMPTS     # labelers wrote the terse and the good answer
STAGES = {}                      # filled in as the demos run


def demo_1_base_is_a_completion_engine():
    line("DEMO 1: the base model completes text, it does not answer")
    for p, prompt in enumerate(PROMPTS):
        print(f'\n  prompt:    "{prompt}"')
        print(f'  base says: "{favourite(BASE_LOGITS, p)}"')
    print()
    report("base", BASE_LOGITS)
    probs = softmax(BASE_LOGITS)[0]
    print(f"  base policy on prompt 0: continuations {probs[:3].sum():.3f},"
          f" terse {probs[3]:.3f},")
    print(f"                           good {probs[4]:.4f},"
          f" padded {probs[5]:.4f}")
    STAGES["base"] = BASE_LOGITS
    print("\nREAD THIS: every favourite is a continuation. The base model is not")
    print("being unhelpful on purpose -- it is predicting what would come next")
    print("in a document that contains that line. Exam papers, forums, lists of")
    print("exercises. That is the only thing it was ever trained to do.")


def demo_2_sft_teaches_the_format():
    line("DEMO 2: stage 1, SFT -- show it examples of answering")
    sft_logits = sft(BASE_LOGITS, DEMOS)
    STAGES["sft"] = sft_logits
    for p, prompt in enumerate(PROMPTS[:2]):
        print(f'\n  prompt:   "{prompt}"')
        print(f'  SFT says: "{favourite(sft_logits, p)}"')
    print()
    report("base", BASE_LOGITS)
    report("SFT", sft_logits)
    probs = softmax(sft_logits)[0]
    print(f"\n  SFT policy on prompt 0, per response: {probs}")
    print(f"  (terse answer {probs[3]:.2f}, good answer {probs[4]:.2f},"
          f" padded answer {probs[5]:.4f})")
    print("\nREAD THIS: eight demonstrations turned a completion engine into")
    print("something that answers. But SFT only copies what the labelers wrote:")
    print("it is split between the terse answer and the good one, because the")
    print("demos were. It has no idea one is better. That needs a preference.")


def demo_3_reward_model_from_comparisons():
    global LABELER_TEMP
    line("DEMO 3: stage 2, a reward model from 'a is better than b'")
    comps = collect_comparisons(STAGES["sft"], n=240, rng=np.random.default_rng(1))
    STAGES["comparisons"] = comps
    print(f"  comparisons collected on SFT outputs: {len(comps)}")
    print(f"  simulated inter-labeler agreement:    {agreement(comps):.3f}"
          f"   (paper: about 0.73)")

    seen = np.zeros((N_PROMPTS, N_RESP), dtype=bool)
    for p, a, b in comps:
        seen[p, a] = seen[p, b] = True
    STAGES["seen"] = seen
    print(f"  responses that ever reached a labeler: {seen.sum()} of {seen.size}")
    print(f"  padded answers that reached a labeler: {int(seen[:, 5].sum())} of 4")

    reward = train_reward_model(comps)
    STAGES["reward"] = reward
    print(f"\n  reward model vs hidden score, on responses labelers saw: "
          f"corr = {np.corrcoef(reward[seen], HIDDEN[seen])[0, 1]:.3f}")
    print(f"  reward model vs hidden score, on ALL 24 responses:       "
          f"corr = {np.corrcoef(reward.ravel(), HIDDEN.ravel())[0, 1]:.3f}")
    print("\n  prompt 0, learned reward vs hidden score:")
    for r in range(N_RESP):
        flag = "" if seen[0, r] else "   <- never seen"
        print(f"    reward {reward[0, r]:5.2f}   hidden {HIDDEN[0, r]:3.1f}"
              f"   {TEXT[0][r][:30]!r}{flag}")

    # Side by side: the same pipeline with labelers who barely agree.
    keep, LABELER_TEMP = LABELER_TEMP, 8.0
    noisy = collect_comparisons(STAGES["sft"], n=240, rng=np.random.default_rng(1))
    noisy_reward = train_reward_model(noisy)
    noisy_rl = rlhf(STAGES["sft"], STAGES["sft"], noisy_reward, beta=BETA)[-1]
    clean_rl = rlhf(STAGES["sft"], STAGES["sft"], reward, beta=BETA)[-1]
    print(f"\n  BREAK IT -- LABELER_TEMP = 8.0, so labelers agree only "
          f"{agreement(noisy):.2f} of the time:")
    LABELER_TEMP = keep
    print(f"    corr with hidden score on seen responses: "
          f"{np.corrcoef(noisy_reward[seen], HIDDEN[seen])[0, 1]:.3f}"
          f"   (ordering survives)")
    print(f"    reward gap, good minus terse, prompt 0:   "
          f"{noisy_reward[0, 4] - noisy_reward[0, 3]:.2f}"
          f"   (was {reward[0, 4] - reward[0, 3]:.2f}: signal gone)")
    print(f"    true helpfulness after RL on this RM:     "
          f"{true_helpfulness(noisy_rl):.2f}"
          f"   (was {true_helpfulness(clean_rl):.2f})")
    print("\nREAD THIS: the RM is near-perfect on what labelers actually saw and")
    print("guessing on what they never saw. It learned 'longer is better' from")
    print("terse-vs-good and extrapolated straight past the good answer to the")
    print("padded one. Hold that thought for demo 6. And labeler noise goes")
    print("straight into the reward: the ranking survives, the margins do not,")
    print("and margins are what RL trades against the KL leash.")


def demo_4_rl_with_a_leash():
    line("DEMO 4: stage 3, optimise against the reward model, KL leash to SFT")
    traj = rlhf(STAGES["sft"], STAGES["sft"], STAGES["reward"], beta=BETA)
    rl_logits = traj[-1]
    STAGES["rlhf"] = rl_logits
    for p, prompt in enumerate(PROMPTS[:2]):
        print(f'\n  prompt:    "{prompt}"')
        print(f'  RLHF says: "{favourite(rl_logits, p)}"')
    print("\n  stage          mass on answers   true helpfulness   KL to SFT")
    for name in ("base", "sft", "rlhf"):
        lg = STAGES[name]
        print(f"  {name:<14} {mass_on_instructions(lg):>15.3f}"
              f"   {true_helpfulness(lg):>16.2f}   {kl(lg, STAGES['sft']):>9.3f}")
    print(f"\n  ceiling (always the good answer): {HIDDEN[:, 4].mean():.2f}")
    probs = softmax(rl_logits)[0]
    print(f"  RLHF policy on prompt 0: terse {probs[3]:.2f}, good {probs[4]:.2f},"
          f" padded {probs[5]:.3f}")

    # The objective has a closed-form optimum: pi_SFT(y) * exp(r(y) / beta),
    # renormalised. Check the trained policy landed on it.
    closed = softmax(STAGES["sft"] + STAGES["reward"] / BETA)
    print(f"  closed form pi_SFT * exp(r/beta): max |diff| from trained policy "
          f"= {np.abs(closed - softmax(rl_logits)).max():.4f}")
    print("\nREAD THIS: RL did what SFT could not -- it moved mass from the terse")
    print("answer to the good one, because the reward model ranks them. It also")
    print("leaned toward the padded answer, because the RM over-rates it. The")
    print("KL leash is what stopped it going all the way. Demo 6 removes it.")


def demo_5_skip_sft_and_wrong_leash():
    line("DEMO 5: break it -- skip SFT / leash to the wrong model")
    target = 7.0
    good = rlhf(STAGES["sft"], STAGES["sft"], STAGES["reward"], beta=BETA)
    # Break 1: no SFT model exists, so RL starts from base AND leashes to base.
    no_sft = rlhf(BASE_LOGITS, BASE_LOGITS, STAGES["reward"], beta=BETA)
    # Break 2: SFT happened, but somebody pointed the KL at the base model.
    wrong_ref = rlhf(STAGES["sft"], BASE_LOGITS, STAGES["reward"], beta=BETA)

    print(f"  steps until true helpfulness >= {target}, then the final numbers:\n")
    print("  run                     steps   mass on answers   helpfulness")
    for name, traj in (("SFT -> RL, KL to SFT", good),
                       ("base -> RL, KL to base", no_sft),
                       ("SFT -> RL, KL to base", wrong_ref)):
        lg = traj[-1]
        print(f"  {name:<22} {steps_to_reach(traj, target)}"
              f"   {mass_on_instructions(lg):>15.3f}"
              f"   {true_helpfulness(lg):>11.2f}")
    print(f"\n  helpfulness at step 10:  with SFT {true_helpfulness(good[10]):.2f}"
          f"   without SFT {true_helpfulness(no_sft[10]):.2f}")
    print(f"  base policy's probability of the good answer, prompt 0: "
          f"{softmax(BASE_LOGITS)[0, 4]:.4f}")
    print("\nREAD THIS: from the base model the good answer is a rounding error,")
    print("so RL has almost nothing to push on -- it crawls. And a leash to the")
    print("base model pulls back toward continuations the whole way, so it")
    print("settles far short. Rows 2 and 3 end in the SAME place: the reference")
    print("model decides the destination, the starting point decides the speed.")


def demo_6_no_leash_and_distribution_shift():
    line("DEMO 6: break it -- beta = 0, and the fix: compare on-policy")
    hacked = rlhf(STAGES["sft"], STAGES["sft"], STAGES["reward"], beta=0.0)[-1]
    print(f'\n  beta = 0 says: "{favourite(hacked, 0)}"')
    print(f"  probability on the padded answer, prompt 0: "
          f"{softmax(hacked)[0, 5]:.3f}")
    report(f"beta = {BETA}", STAGES["rlhf"])
    report("beta = 0", hacked)
    print(f"  KL from SFT:  beta = {BETA} -> "
          f"{kl(STAGES['rlhf'], STAGES['sft']):.2f}"
          f"     beta = 0 -> {kl(hacked, STAGES['sft']):.2f}")

    # The paper's fix: labelers compare what the CURRENT policy produces,
    # the RM is retrained on everything, RL runs again.
    fresh = collect_comparisons(hacked, n=240, rng=np.random.default_rng(2))
    reward2 = train_reward_model(STAGES["comparisons"] + fresh)
    fixed = rlhf(STAGES["sft"], STAGES["sft"], reward2, beta=BETA)[-1]
    old = STAGES["reward"]
    print("\n  + 240 comparisons on the drifted policy's own outputs,"
          " RM retrained:")
    print(f"    corr with hidden score on ALL 24 responses: "
          f"{np.corrcoef(reward2.ravel(), HIDDEN.ravel())[0, 1]:.3f}"
          f"   (was {np.corrcoef(old.ravel(), HIDDEN.ravel())[0, 1]:.3f})")
    print(f"    RM on prompt 0: good {reward2[0, 4]:.2f}"
          f"  padded {reward2[0, 5]:.2f}"
          f"   (was good {old[0, 4]:.2f}, padded {old[0, 5]:.2f})")
    report("retrained RM", fixed)
    print(f'    prompt 0: "{favourite(fixed, 0)}"')
    print("\nREAD THIS: with no leash the policy runs straight to whatever the RM")
    print("over-rates -- an answer no labeler ever graded. That is reward")
    print("hacking, and the RM was never wrong on its own data. Two guards in")
    print("the paper: the KL penalty, and collecting comparisons on the policy's")
    print("own samples, so the RM is trained where the policy actually lives.")


if __name__ == "__main__":
    demo_1_base_is_a_completion_engine()
    demo_2_sft_teaches_the_format()
    demo_3_reward_model_from_comparisons()
    demo_4_rl_with_a_leash()
    demo_5_skip_sft_and_wrong_leash()
    demo_6_no_leash_and_distribution_shift()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  0. base    = GPT-3. Predicts the next token of web text. Completes, not answers.
  1. SFT     = cross-entropy on ~13k human-written answers. Teaches the FORMAT.
  2. RM      = Bradley-Terry on ~33k "a beats b" pairs. Captures the PREFERENCE.
  3. RL      = PPO on ~31k prompts: maximise  r(y) - beta * KL(policy || SFT).
     PPO-ptx = the same, plus pretraining gradients mixed in. Alignment tax fix.

  Result: labelers preferred the 1.3B InstructGPT to the 175B GPT-3.
  ChatGPT is this recipe, run on dialogue data.
""")
