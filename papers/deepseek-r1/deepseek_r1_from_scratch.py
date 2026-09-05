"""
DeepSeek-R1 -- for programmers, not researchers.

Run it:      python3 deepseek_r1_from_scratch.py
Debug it:    breakpoint in `reward_correct_format` and watch what the optimiser
             is actually being paid for.

THE HONEST BIT, UP FRONT. There is no language model in this file. Nothing here
is evidence about what a 671B-parameter model does. What it IS evidence for is
the mechanism the paper leans on:

    if longer solutions are genuinely more likely to be right, then a policy
    gradient that is paid ONLY for being right will make solutions longer,
    without anybody rewarding length.

That claim is small enough to prove on a laptop, and it is the claim the whole
R1-Zero result stands on. Everything below is an analogy for it, labelled as
such.

The toy task ("read the smudged meter"):
  * every problem has one true digit 0-9 and SIX noisy readings of it
  * each reading is correct with probability 0.55, otherwise it is a random
    other digit
  * the policy chooses, at every step, whether to spend one more step reading,
    or to stop and answer with the majority of what it has read so far
  * more reads -> better majority -> more likely to be right. Genuinely.
  * but there are only six readings. Step 7 onwards re-reads one you already
    have, which is not new information and can amplify a wrong reading.

NumPy does the environment and the maths. Torch does the training loop only,
CPU, three-layer MLP, because hand-writing backprop through 150 GRPO steps
would bury the one thing worth looking at.
"""

import re
import time

import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(1)

N_DIGITS = 10        # answers are digits 0-9
N_READINGS = 6       # how much genuine evidence a problem contains
K_MAX = 16           # the context window: a hard stop, not a penalty
P_CORRECT = 0.55     # per-reading accuracy on the hard task
BETA = 0.1           # default KL leash (swept in demo 5)

CONTINUE, STOP_TAGGED, STOP_RAW = 0, 1, 2
N_ACTIONS = 3


# ---------------------------------------------------------------------------
# STAGE 0 -- the environment. A problem is a truth plus six noisy readings.
# Reading k for k < 6 is fresh evidence. Reading 7, 8, 9... is a re-read of
# something already seen: no new information, and it double-counts whatever
# noise it happens to land on.
# ---------------------------------------------------------------------------
def make_problem(rng, p_correct=P_CORRECT):
    truth = int(rng.integers(N_DIGITS))
    readings = []
    for _ in range(N_READINGS):
        if rng.random() < p_correct:
            readings.append(truth)
        else:
            step = 1 + int(rng.integers(N_DIGITS - 1))
            readings.append((truth + step) % N_DIGITS)
    return truth, readings


def next_reading(rng, readings, seen):
    """Fresh evidence while there is any left, otherwise a re-read."""
    if len(seen) < N_READINGS:
        return readings[len(seen)]
    return seen[int(rng.integers(len(seen)))]


def majority(seen):
    """The answer readout. NOT learned -- the policy only decides when to stop."""
    if not seen:
        return -1
    return int(np.bincount(seen, minlength=N_DIGITS).argmax())


# ---------------------------------------------------------------------------
# STAGE 1 -- what the policy sees. Four numbers: how much context is spent, how
# decisive the current vote is, how big the leading count is, and a bias term.
# Deliberately tiny, so nothing interesting can hide inside the network.
# ---------------------------------------------------------------------------
def features(seen):
    k = len(seen)
    if k == 0:
        return [0.0, 0.0, 0.0, 1.0]
    counts = np.sort(np.bincount(seen, minlength=N_DIGITS))[::-1]
    return [k / K_MAX, (counts[0] - counts[1]) / k, counts[0] / k, 1.0]


def make_policy(hidden=16):
    return nn.Sequential(nn.Linear(4, hidden), nn.Tanh(),
                         nn.Linear(hidden, N_ACTIONS))


# ---------------------------------------------------------------------------
# STAGE 2 -- the output format. R1 asks for thinking between <think> tags and
# the final answer between <answer> tags, so that a rule -- not a learned
# reward model -- can grade it. STOP_RAW emits the same content with no tags,
# which is the degree of freedom the format reward exists to remove.
# ---------------------------------------------------------------------------
def render(seen, answer, tagged):
    steps = "\n".join(f"step {i + 1}: read -> {v}" for i, v in enumerate(seen))
    if tagged:
        return f"<think>\n{steps}\n</think>\n<answer>{answer}</answer>"
    return f"{steps}\nthe answer is {answer}"


ANSWER_RE = re.compile(r"<answer>\s*(\d+)\s*</answer>")
FORMAT_RE = re.compile(r"<think>.*</think>\s*<answer>.*</answer>", re.S)


def parse_answer(text):
    """Pull the final answer out of the tags. Returns None if it isn't there."""
    m = ANSWER_RE.search(text)
    return int(m.group(1)) if m else None


def parse_answer_broken(text):
    """The bug everyone ships once: first number ANYWHERE. Grabs the working."""
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else None


def format_ok(text):
    return bool(FORMAT_RE.search(text))


# ---------------------------------------------------------------------------
# STAGE 3 -- the reward functions. R1-Zero's is rule-based: accuracy plus
# format, no learned reward model, because a learned one invites the optimiser
# to go hunting for its soft spots (see the RLHF page). Note what is NOT here:
# any term that mentions length.
# ---------------------------------------------------------------------------
def reward_correct_format(text, truth, n_steps, parser=parse_answer):
    return 0.2 * format_ok(text) + 1.0 * (parser(text) == truth)


def reward_format_only(text, truth, n_steps, parser=parse_answer):
    return 0.2 * format_ok(text)                      # never mentions the answer


def reward_with_length(text, truth, n_steps, parser=parse_answer):
    return reward_correct_format(text, truth, n_steps, parser) + 0.05 * n_steps


# ---------------------------------------------------------------------------
# STAGE 4 -- rollouts. GRPO samples a GROUP of answers for the same prompt, so
# all trajectories for one problem are run together and compared against each
# other. Everything is simulated in lockstep so torch sees one batched forward
# per timestep instead of one per token.
# ---------------------------------------------------------------------------
def rollout(policy, rng, problems, group, greedy=False):
    n = len(problems) * group
    probs = [problems[i // group] for i in range(n)]
    rngs = [np.random.default_rng(int(rng.integers(1 << 30))) for _ in range(n)]
    seen = [[] for _ in range(n)]
    alive = [True] * n
    tagged = [True] * n
    feats, acts, owner = [], [], []

    for _ in range(K_MAX):
        idx = [i for i in range(n) if alive[i]]
        if not idx:
            break
        x = torch.tensor([features(seen[i]) for i in idx], dtype=torch.float32)
        with torch.no_grad():
            p = torch.softmax(policy(x), dim=-1)
        for row, i in enumerate(idx):
            a = int(torch.argmax(p[row])) if greedy else \
                int(torch.multinomial(p[row], 1))
            feats.append(features(seen[i]))
            acts.append(a)
            owner.append(i)
            if a == CONTINUE:
                seen[i].append(next_reading(rngs[i], probs[i][1], seen[i]))
                if len(seen[i]) >= K_MAX:
                    alive[i] = False          # context window, forced tagged stop
            else:
                tagged[i] = (a == STOP_TAGGED)
                alive[i] = False

    texts = [render(seen[i], majority(seen[i]), tagged[i]) for i in range(n)]
    lens = [len(seen[i]) for i in range(n)]
    truths = [probs[i][0] for i in range(n)]
    return dict(feats=feats, acts=acts, owner=owner, texts=texts,
                lens=lens, truths=truths, n=n)


# ---------------------------------------------------------------------------
# STAGE 5 -- GRPO, in one function. Two sentences: sample a group of G answers
# for each prompt, score them, and use (r - mean of the group) / (std of the
# group) as the advantage, so no value network is needed -- the group is the
# baseline. Then the usual policy gradient, plus a KL leash back to the
# reference policy. Full derivation on the GRPO page; this is the same
# estimator, in 20 lines.
# ---------------------------------------------------------------------------
def grpo_step(policy, ref, opt, roll, rewards, group, beta):
    r = torch.tensor(rewards, dtype=torch.float32).view(-1, group)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
    adv = adv.view(-1)

    x = torch.tensor(roll["feats"], dtype=torch.float32)
    logits = policy(x)
    logp = torch.log_softmax(logits, dim=-1)
    chosen = logp[torch.arange(len(roll["acts"])), torch.tensor(roll["acts"])]

    # sum the log-probs of each trajectory, weight by that trajectory's advantage
    per_traj = torch.zeros(roll["n"]).index_add_(
        0, torch.tensor(roll["owner"]), chosen)
    pg = -(per_traj * adv).mean()

    # the leash: KL(policy || reference) at every state actually visited
    with torch.no_grad():
        ref_logp = torch.log_softmax(ref(x), dim=-1)
    kl = (logp.exp() * (logp - ref_logp)).sum(-1)
    kl_traj = torch.zeros(roll["n"]).index_add_(0, torch.tensor(roll["owner"]), kl)

    opt.zero_grad()
    (pg + beta * kl_traj.mean()).backward()
    opt.step()
    return float(kl_traj.mean())


def train(reward_fn=reward_correct_format, beta=BETA, iters=150, prompts=8,
          group=8, p_correct=P_CORRECT, hidden=16, seed=0, parser=parse_answer,
          eval_every=0, policy=None):
    """One full GRPO run. Returns the policy and a per-iteration history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = policy if policy is not None else make_policy(hidden)
    ref = make_policy(hidden)
    ref.load_state_dict(policy.state_dict())          # reference = the start
    opt = torch.optim.Adam(policy.parameters(), lr=0.03)

    hist = []
    for it in range(iters + 1):
        probs = [make_problem(rng, p_correct) for _ in range(prompts)]
        roll = rollout(policy, rng, probs, group)
        rewards = [reward_fn(t, tr, L, parser)
                   for t, tr, L in zip(roll["texts"], roll["truths"], roll["lens"])]
        acc = np.mean([parse_answer(t) == tr
                       for t, tr in zip(roll["texts"], roll["truths"])])
        row = dict(it=it, length=float(np.mean(roll["lens"])), acc=float(acc),
                   reward=float(np.mean(rewards)),
                   fmt=float(np.mean([format_ok(t) for t in roll["texts"]])))
        if it < iters:
            row["kl"] = grpo_step(policy, ref, opt, roll, rewards, group, beta)
        hist.append(row)
        if eval_every and it % eval_every == 0:
            # a held-out measurement, so the printed curve is signal not noise
            ev = evaluate(policy, n=300, p_correct=p_correct, seed=4242)
            row.update(ev_acc=ev["acc"], ev_len=ev["length"])
    return policy, hist


def evaluate(policy, n=400, p_correct=P_CORRECT, seed=999, parser=parse_answer):
    """Fresh problems, sampled (not greedy) rollouts, plain averages."""
    rng = np.random.default_rng(seed)
    probs = [make_problem(rng, p_correct) for _ in range(n)]
    roll = rollout(policy, rng, probs, 1)
    acc = np.mean([parser(t) == tr for t, tr in zip(roll["texts"], roll["truths"])])
    return dict(acc=float(acc), length=float(np.mean(roll["lens"])),
                fmt=float(np.mean([format_ok(t) for t in roll["texts"]])),
                texts=roll["texts"], truths=roll["truths"])


def header(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


# ===========================================================================
# DEMO 1 -- the task. Does thinking longer actually make you right more often?
# If the answer is no, nothing else on this page means anything.
# ===========================================================================
def demo_1_the_task():
    header("DEMO 1 -- does spending more steps genuinely help?")
    print("Fixed number of reads k, majority vote, 4000 problems each.")
    print("Six genuine readings exist. Step 7+ is a re-read of one you have.\n")
    print("   k   accuracy (hard, p=0.55)   accuracy (control, p=1.00)")
    for k in [1, 2, 3, 4, 5, 6, 8, 12, 16]:
        row = []
        for p in (P_CORRECT, 1.0):
            rng = np.random.default_rng(7)
            hits = 0
            for _ in range(4000):
                truth, readings = make_problem(rng, p)
                seen = []
                for _ in range(k):
                    seen.append(next_reading(rng, readings, seen))
                hits += (majority(seen) == truth)
            row.append(hits / 4000)
        print(f"  {k:2d}            {row[0]:.3f}                    {row[1]:.3f}")
    print("\nHard task: rises to k=6, then flat-to-worse (re-reads amplify noise).")
    print("Control task: k=1 is already perfect. Extra steps buy exactly nothing.")


# ===========================================================================
# DEMO 2 -- THE CENTREPIECE. Train with a correctness+format reward only, and
# watch length climb next to accuracy. Then the control task, where it doesn't.
# ===========================================================================
def demo_2_emergence():
    header("DEMO 2 -- length rises although nothing rewards length")
    print(f"Reward = 0.2*format + 1.0*correct.  No length term. GRPO, beta={BETA}.")
    print("Every row is 300 held-out problems, so the curve is signal, not noise.\n")

    _, hist = train(reward_correct_format, seed=0, eval_every=15)
    print("  HARD TASK (p=0.55, extra steps genuinely help)")
    print("   iter   mean length   accuracy")
    for h in hist:
        if "ev_acc" in h:
            print(f"   {h['it']:4d}      {h['ev_len']:8.2f}      {h['ev_acc']:.3f}")

    _, hc = train(reward_correct_format, seed=0, p_correct=1.0, eval_every=15)
    print("\n  CONTROL TASK (p=1.00, one read is already enough)")
    print("   iter   mean length   accuracy")
    for h in hc:
        if "ev_acc" in h:
            print(f"   {h['it']:4d}      {h['ev_len']:8.2f}      {h['ev_acc']:.3f}")

    hl = [h for h in hist if "ev_len" in h]
    cl = [h for h in hc if "ev_len" in h]
    print(f"\n  hard task    length {hl[0]['ev_len']:.2f} -> {hl[-1]['ev_len']:.2f}"
          f"   accuracy {hl[0]['ev_acc']:.3f} -> {hl[-1]['ev_acc']:.3f}")
    print(f"  control task length {cl[0]['ev_len']:.2f} -> {cl[-1]['ev_len']:.2f}"
          f"   accuracy {cl[0]['ev_acc']:.3f} -> {cl[-1]['ev_acc']:.3f}")
    print("\nSame reward, same optimiser, same network, same seed. The control")
    print("stops at the one read it needs to answer at all and goes no further;")
    print("nothing pushes it. That is the honest boundary of the claim: length")
    print("emerges only where length buys correctness.")


# ===========================================================================
# DEMO 3 -- reward design. Three reward functions, same everything else.
# ===========================================================================
def demo_3_reward_design():
    header("DEMO 3 -- three reward functions, and what each one buys you")
    runs = [
        ("correct + format", reward_correct_format),
        ("format ONLY", reward_format_only),
        ("correct + format + 0.05*length", reward_with_length),
    ]
    print("  reward function                  accuracy   mean length   format")
    out = {}
    for name, fn in runs:
        pol, _ = train(fn, beta=BETA, seed=0)
        ev = evaluate(pol)
        out[name] = ev
        print(f"  {name:32s}   {ev['acc']:.3f}      {ev['length']:6.2f}"
              f"       {ev['fmt']:.2f}")
    print("\nFormat-only: perfect structure, and the accuracy of a coin toss with")
    print("ten sides. It never learned to look at the answer, because nothing")
    print("ever paid it to. Well-formatted nonsense.")
    print("\nLength-in-reward: it pads to the context window and accuracy does NOT")
    print("follow -- the extra steps are re-reads. You bought tokens, not thought.")
    pad = out["correct + format + 0.05*length"]["texts"][0]
    print(f"\nA padded trace from the length-rewarded run, {len(pad.splitlines())}"
          " lines, first 7 shown:")
    for line in pad.splitlines()[:7]:
        print("   " + line)


# ===========================================================================
# DEMO 4 -- the format reward and the parser bug that eats reasoning pipelines.
# ===========================================================================
def demo_4_parser():
    header("DEMO 4 -- the tags, the parser, and the bug you will ship")
    pol, _ = train(reward_correct_format, beta=BETA, seed=0)
    ev = evaluate(pol)
    i = next(j for j in range(len(ev["texts"]))
             if parse_answer(ev["texts"][j]) == ev["truths"][j])
    print("One trained rollout, verbatim:\n")
    print("   " + ev["texts"][i].replace("\n", "\n   "))
    print(f"\n   truth                       : {ev['truths'][i]}")
    print(f"   parse_answer (tag-anchored) : {parse_answer(ev['texts'][i])}")
    print(f"   parse_answer_broken (first  : {parse_answer_broken(ev['texts'][i])}"
          "   <- that's the step index")
    print("     number anywhere)")
    good = np.mean([parse_answer(t) == tr
                    for t, tr in zip(ev["texts"], ev["truths"])])
    bad = np.mean([parse_answer_broken(t) == tr
                   for t, tr in zip(ev["texts"], ev["truths"])])
    print(f"\n   scored with the correct parser : {good:.3f}")
    print(f"   scored with the broken parser  : {bad:.3f}")
    print("\nSame model, same rollouts. The second number is the parser's opinion")
    print("of the working-out. Train against it and you teach the model to make")
    print("its first step index equal the answer.")
    pol2, hist2 = train(reward_correct_format, beta=BETA, seed=0,
                        parser=parse_answer_broken)
    ev2 = evaluate(pol2)
    print(f"\n   trained AGAINST the broken parser -> real accuracy {ev2['acc']:.3f},"
          f" length {ev2['length']:.2f}")
    print("   (the reward signal is now noise, so the policy stops early)")


# ===========================================================================
# DEMO 5 -- the KL leash. Sweep beta. Same reward, same task.
# ===========================================================================
def demo_5_kl_leash():
    header("DEMO 5 -- the KL leash, swept")
    print("  beta     accuracy   mean length   KL from reference (nats/traj)")
    keep = {}
    for beta in [0.0, 0.02, 0.1, 0.5]:
        pol, hist = train(reward_correct_format, beta=beta, seed=0)
        ev = evaluate(pol)
        keep[beta] = ev
        print(f"  {beta:4.2f}      {ev['acc']:.3f}      {ev['length']:6.2f}"
              f"        {hist[-2]['kl']:8.3f}")
    t0 = keep[0.0]["texts"][0]
    print("\nbeta = 0: nothing holds it back, so it runs to the context window.")
    print("It still scores well -- the answer is in the right tags, and often")
    print(f"right. Here is one of its answers, all {len(t0.splitlines())} lines,")
    print("of which only the first six carry any new information:\n")
    print("   " + t0.replace("\n", "\n   "))
    print("\nbeta = 0.5: barely leaves the reference; it never learns to think.")
    print("beta is tuned, not derived. The number you report is the KL.")


# ===========================================================================
# DEMO 6 -- distillation. Train a small policy on a strong policy's successful
# trajectories, versus running GRPO on the small policy directly.
# ===========================================================================
def demo_6_distillation():
    header("DEMO 6 -- distil the teacher, or RL the small model directly?")
    teacher, _ = train(reward_correct_format, iters=150, hidden=16, seed=0)
    ev_t = evaluate(teacher)
    print(f"  teacher  (hidden=16, 150 GRPO iters)   acc {ev_t['acc']:.3f}"
          f"   length {ev_t['length']:6.2f}")

    # collect the teacher's SUCCESSFUL trajectories -- the R1 recipe
    rng = np.random.default_rng(5)
    probs = [make_problem(rng) for _ in range(300)]
    roll = rollout(teacher, rng, probs, 4)
    ok = [parse_answer(t) == tr and format_ok(t)
          for t, tr in zip(roll["texts"], roll["truths"])]
    keepX = [f for f, o in zip(roll["feats"], roll["owner"]) if ok[o]]
    keepY = [a for a, o in zip(roll["acts"], roll["owner"]) if ok[o]]
    n_roll = len(ok)
    print(f"  kept {sum(ok)}/{n_roll} teacher rollouts -> {len(keepY)} labelled"
          " decisions\n")
    print(f"  Budget for BOTH students: {n_roll} rollouts of the environment.")
    print("  The distilled student spends them by reusing the teacher's.")
    print("  The RL student has to generate its own: 64 per GRPO iteration.\n")

    # (a) supervised: imitate the teacher's decisions. No new rollouts at all.
    torch.manual_seed(0)
    student = make_policy(hidden=4)
    opt = torch.optim.Adam(student.parameters(), lr=0.05)
    X = torch.tensor(keepX, dtype=torch.float32)
    Y = torch.tensor(keepY)
    for _ in range(150):
        opt.zero_grad()
        nn.functional.cross_entropy(student(X), Y).backward()
        opt.step()
    ev_s = evaluate(student)

    # (b) the same small network, GRPO from scratch, same rollout budget
    iters_rl = n_roll // 64
    torch.manual_seed(0)
    rl_small, _ = train(reward_correct_format, iters=iters_rl, hidden=4, seed=0,
                        policy=make_policy(hidden=4))
    ev_r = evaluate(rl_small)

    # (c) and the same network given eight times the rollouts, for context
    torch.manual_seed(0)
    rl_long, _ = train(reward_correct_format, iters=150, hidden=4, seed=0,
                       policy=make_policy(hidden=4))
    ev_l = evaluate(rl_long)

    print("  small network, hidden=4:")
    lbl = "distilled from teacher, 0 new rollouts"
    print(f"    {lbl:41s}acc {ev_s['acc']:.3f}   length {ev_s['length']:6.2f}")
    lbl = f"GRPO directly, {iters_rl} iters ({iters_rl * 64} rollouts)"
    print(f"    {lbl:41s}acc {ev_r['acc']:.3f}   length {ev_r['length']:6.2f}")
    lbl = "GRPO directly, 150 iters (9600 rollouts)"
    print(f"    {lbl:41s}acc {ev_l['acc']:.3f}   length {ev_l['length']:6.2f}")
    print("\nEvery successful teacher trajectory hands the student a label for")
    print("every decision in it. A reward hands it one scalar for the whole")
    print("trajectory. That is the entire difference, and it is why R1's authors")
    print("found distilling into small dense models more effective than running")
    print("large-scale RL on them. Matched on ROLLOUTS, distillation wins here")
    print("too. Matched on gradient steps, my two students are within noise --")
    print("this is an analogy for their result, not evidence of it.")


if __name__ == "__main__":
    t0 = time.time()
    demo_1_the_task()
    demo_2_emergence()
    demo_3_reward_design()
    demo_4_parser()
    demo_5_kl_leash()
    demo_6_distillation()
    print(f"\n[done in {time.time() - t0:.1f}s]")
