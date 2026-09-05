"""
Denoising Diffusion Probabilistic Models (Ho et al., 2020) -- for programmers.

Run it:      python3 diffusion_from_scratch.py
Debug it:    set a breakpoint in p_step and watch the loop run backwards.

The forward (noising) process is pure NumPy: it is a for loop with a
closed-form shortcut, nothing to learn. Torch is used for ONE thing: training
the small MLP that predicts the noise. CPU only, about 30 seconds.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(1)        # tiny model: one thread is faster than ten
np.set_printoptions(precision=3, suppress=True)
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# STAGE 0 -- toy data. A ring of radius 1 with a little radial jitter.
# Two numbers describe it completely: mean radius and radius spread. That is
# how we will grade the generated samples later.
# ---------------------------------------------------------------------------
def make_ring(n, radius=1.0, jitter=0.1):
    theta = RNG.uniform(0, 2 * np.pi, n)
    r = radius + jitter * RNG.normal(size=n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def ring_stats(x):
    """(mean radius, radius spread) of a batch of 2-D points."""
    r = np.sqrt((x ** 2).sum(axis=1))
    return r.mean(), r.std()


# ---------------------------------------------------------------------------
# STAGE 1 -- the schedule. Section 4 of the paper: T = 1000, beta linear from
# 1e-4 to 0.02. Three arrays, all indexed by t = 0..T, index 0 meaning "clean".
#
#   beta_t       how much noise step t adds
#   alpha_t      1 - beta_t: how much of the previous step survives
#   alpha_bar_t  product of all alphas so far: how much of x_0 survives to t
# ---------------------------------------------------------------------------
def make_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    betas = np.concatenate([[0.0], np.linspace(beta_start, beta_end, T)])
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)          # alpha_bar[0] = 1: no noise yet
    return betas, alphas, alpha_bar


# ---------------------------------------------------------------------------
# STAGE 2 -- the forward process. Equation 2: one step of corruption.
#
#   q(x_t | x_{t-1}) = N( sqrt(1 - beta_t) * x_{t-1},  beta_t * I )
#
# Shrink a little, add a little Gaussian noise. Repeat T times and the data is
# gone. Nothing here is learned. It is a for loop.
# ---------------------------------------------------------------------------
def q_step(x_prev, t, betas):
    noise = RNG.normal(size=x_prev.shape)
    return np.sqrt(1 - betas[t]) * x_prev + np.sqrt(betas[t]) * noise


def q_loop(x0, t, betas):
    x = x0
    for s in range(1, t + 1):
        x = q_step(x, s, betas)
    return x


# ---------------------------------------------------------------------------
# STAGE 3 -- the shortcut. Equation 4: jump straight to step t.
#
#   q(x_t | x_0) = N( sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I )
#   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,  eps ~ N(0, I)
#
# Gaussians compose: t small shrink-and-noise steps collapse into one. This is
# why training is cheap -- pick any t, build x_t in one line, no loop.
# The SQUARE ROOTS matter: variances add, standard deviations don't.
# ---------------------------------------------------------------------------
def q_sample(x0, t, alpha_bar, eps):
    ab = alpha_bar[t]                                  # scalar, or (n,) per-sample
    ab = ab.reshape(-1, 1) if np.ndim(ab) else ab
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps


# ---------------------------------------------------------------------------
# STAGE 4 -- the model. eps_theta(x_t, t): given a noisy point and the step
# number, guess the noise that was added. The paper uses a U-Net; for 2-D
# points a 3-layer MLP is plenty. The idea is identical.
#
# t goes in as a sinusoidal embedding -- the same sine/cosine trick as the
# transformer's positional encoding, applied to a step number.
# ---------------------------------------------------------------------------
def t_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class NoisePredictor(nn.Module):
    def __init__(self, hidden=128, t_dim=32, use_t=True):
        super().__init__()
        self.use_t, self.t_dim = use_t, t_dim
        self.net = nn.Sequential(
            nn.Linear(2 + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))

    def forward(self, x, t):
        emb = t_embedding(t, self.t_dim)
        if not self.use_t:
            emb = torch.zeros_like(emb)                # BREAK-IT: blind to t
        return self.net(torch.cat([x, emb], dim=-1))


@torch.no_grad()
def predict_noise(model, x, t):
    """NumPy in, NumPy out. t is one int (sampling) or an array (training)."""
    xt = torch.tensor(x, dtype=torch.float32)
    tt = torch.full((len(x),), int(t)) if np.ndim(t) == 0 else torch.tensor(t)
    return model(xt, tt).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# STAGE 5 -- training. Algorithm 1, one line of the paper per line of code.
# The loss is equation 14, the "simple" one: plain MSE between the noise we
# added and the noise the model guessed. No weighting, no variational bound.
# That unweighted MSE is the paper's key practical finding.
# ---------------------------------------------------------------------------
def train(model, data, alpha_bar, steps=4000, batch=256, lr=1e-3,
          target="eps", log_every=500):
    T = len(alpha_bar) - 1
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    running = 0.0
    for step in range(1, steps + 1):
        x0 = data[RNG.integers(0, len(data), batch)]       # 2: x_0 ~ q(x_0)
        t = RNG.integers(1, T + 1, batch)                   # 3: t ~ Uniform{1..T}
        eps = RNG.normal(size=x0.shape)                     # 4: eps ~ N(0, I)
        xt = q_sample(x0, t, alpha_bar, eps)                # the jump, no loop
        y = eps if target == "eps" else x0                  # x0 = break-it target
        pred = model(torch.tensor(xt, dtype=torch.float32), torch.tensor(t))
        loss = ((pred - torch.tensor(y, dtype=torch.float32)) ** 2).mean()  # 5
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()
        if step % log_every == 0:
            print(f"  step {step:5d}   loss {running / log_every:.4f}")
            running = 0.0
    return model


def eval_loss(model, data, alpha_bar, n=20000, t_max=None, seed=123):
    """The simple loss on a fixed held-out batch, so two models are comparable."""
    T = len(alpha_bar) - 1
    r = np.random.default_rng(seed)
    x0 = data[r.integers(0, len(data), n)]
    t = r.integers(1, (t_max or T) + 1, n)
    eps = r.normal(size=x0.shape)
    pred = predict_noise(model, q_sample(x0, t, alpha_bar, eps), t)
    return ((pred - eps) ** 2).mean()


# ---------------------------------------------------------------------------
# STAGE 6 -- sampling. Algorithm 2: start from pure noise, undo one step at a
# time. Line 4 of the algorithm, exactly:
#
#   x_{t-1} = ( x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_hat ) / sqrt(alpha_t)
#             + sigma_t * z
#   z ~ N(0, I),  sigma_t^2 = beta_t,  and z = 0 on the very last step (t = 1)
#
# Two parts: remove the predicted noise (the "undo"), then add a smaller
# fresh noise back. That second part is not a bug. Without it you get a
# different, deterministic sampler -- see demo 5c.
# ---------------------------------------------------------------------------
def p_step(x_t, t, eps_hat, betas, alphas, alpha_bar, add_noise=True):
    undo = betas[t] / np.sqrt(1 - alpha_bar[t]) * eps_hat
    mean = (x_t - undo) / np.sqrt(alphas[t])
    if t == 1 or not add_noise:
        return mean
    return mean + np.sqrt(betas[t]) * RNG.normal(size=x_t.shape)


def sample(model, n, sched, add_noise=True, log_at=(), x_init=None):
    betas, alphas, alpha_bar = sched
    T = len(betas) - 1
    x = RNG.normal(size=(n, 2)) if x_init is None else x_init   # 1: x_T ~ N(0, I)
    for t in range(T, 0, -1):                                    # 2: for t = T..1
        eps_hat = predict_noise(model, x, t)                     # 3: ask the model
        x = p_step(x, t, eps_hat, betas, alphas, alpha_bar, add_noise)   # 4
        if t in log_at:
            m, s = ring_stats(x)
            print(f"    t = {t:4d}   mean radius {m:.3f}   radius spread {s:.3f}")
    return x


# ---------------------------------------------------------------------------
# STAGE 7 -- DDIM (Song et al., 2020), the fix for slow sampling. Same trained
# model, no retraining. Each step: guess x_0 from the noise prediction, then
# re-noise that guess to a much earlier step. 50 model calls instead of 1000.
# ---------------------------------------------------------------------------
def sample_ddim(model, n, alpha_bar, n_steps=50):
    T = len(alpha_bar) - 1
    ts = np.linspace(T, 0, n_steps + 1).round().astype(int)   # 1000, 980, ..., 0
    x = RNG.normal(size=(n, 2))
    for t, s in zip(ts[:-1], ts[1:]):
        eps_hat = predict_noise(model, x, t)
        x0_hat = (x - np.sqrt(1 - alpha_bar[t]) * eps_hat) / np.sqrt(alpha_bar[t])
        x = np.sqrt(alpha_bar[s]) * x0_hat + np.sqrt(1 - alpha_bar[s]) * eps_hat
    return x


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def describe(label, x):
    m, s = ring_stats(x)
    print(f"  {label:32s} mean radius {m:.3f}   radius spread {s:.3f}")


def demo_1_schedule():
    """The schedule is three arrays. By step T the data is gone."""
    line("DEMO 1: the schedule -- T=1000, beta linear 1e-4 -> 0.02")
    T = 1000
    betas, alphas, alpha_bar = make_schedule(T)
    print(f"beta_1 = {betas[1]:.4f}   beta_T = {betas[T]:.4f}   "
          f"alpha_1 = {alphas[1]:.4f}   alpha_T = {alphas[T]:.4f}\n")
    print("how much of x_0 survives to step t  (alpha_bar_t):")
    for t in (0, T // 4, T // 2, T):
        print(f"  t = {t:4d}   alpha_bar = {alpha_bar[t]:.6f}   "
              f"sqrt(alpha_bar) = {np.sqrt(alpha_bar[t]):.4f}")

    print("\nx_T built by the closed-form jump, 20000 draws from ONE x_0 each:")
    print("  x_0            mean of x_T        std of x_T")
    for x0 in ([1.0, 0.0], [-1.0, 0.0], [3.0, 3.0]):
        x0 = np.array(x0)[None, :].repeat(20000, axis=0)
        xT = q_sample(x0, T, alpha_bar, RNG.normal(size=x0.shape))
        print(f"  {str(x0[0]):14s} {str(xT.mean(0)):18s} {str(xT.std(0))}")

    print("\nREAD THIS: alpha_bar falls from 1 to 0.00004. Whatever x_0 you start")
    print("from -- even (3, 3), nowhere near the ring -- x_T is N(0, 1). That")
    print("is the contract: step T is pure noise, so sampling can START from")
    print("pure noise without ever seeing the data.")


def demo_2_jump_equals_loop():
    """Equation 4 says one jump == t loop steps. Check it with numbers."""
    line("DEMO 2: the closed-form jump equals the step-by-step loop")
    betas, alphas, alpha_bar = make_schedule(1000)
    x0 = np.array([[1.0, 0.0]]).repeat(20000, axis=0)     # same x_0, 20000 times

    print("x_0 = (1, 0). Compare the x-coordinate of x_t, 20000 draws each.\n")
    print("     t   loop  mean   std  |  jump  mean   std  |  theory sqrt(ab)  sqrt(1-ab)")
    for t in (10, 100, 500, 1000):
        loop = q_loop(x0, t, betas)
        jump = q_sample(x0, t, alpha_bar, RNG.normal(size=x0.shape))
        print(f"  {t:4d}   {loop[:, 0].mean():6.3f} {loop[:, 0].std():5.3f}  |"
              f"  {jump[:, 0].mean():6.3f} {jump[:, 0].std():5.3f}  |"
              f"       {np.sqrt(alpha_bar[t]):.3f}     {np.sqrt(1 - alpha_bar[t]):.3f}")

    print("\nREAD THIS: the for loop and the one-liner agree to the second")
    print("decimal at every t, and both agree with the theory columns. So")
    print("training never runs the loop. It picks a random t, jumps there in")
    print("one line, and asks the model what noise it just added.")


def demo_3_train(data, alpha_bar, steps):
    """Algorithm 1. Watch the simple loss fall."""
    line("DEMO 3: train the noise predictor (Algorithm 1, the simple loss)")
    print("baseline -- always predict zero noise -- scores 1.0000. Beat that.\n")
    model = NoisePredictor()
    t0 = time.time()
    train(model, data, alpha_bar, steps=steps)
    print(f"\n  trained {steps} steps on CPU in {time.time() - t0:.1f}s")
    print(f"  held-out loss: {eval_loss(model, data, alpha_bar):.4f}")
    print("\nREAD THIS: the loss is plain MSE between the noise we added and the")
    print("noise the model guessed -- no weighting, no bound. The paper tried the")
    print("full variational objective and found this stripped-down version gave")
    print("BETTER samples. That is the practical finding of the paper.")
    return model


def demo_4_sample(model, data, sched):
    """Algorithm 2. Run the loop backwards, from noise to ring."""
    line("DEMO 4: sample -- start from N(0, 1), undo 1000 steps")
    print("the batch, mid-flight:")
    x = sample(model, 2000, sched, log_at=(1000, 750, 500, 250, 100, 1))
    print()
    describe("real data (ring)", data)
    describe("generated, 1000 DDPM steps", x)
    print("\nREAD THIS: at t = 1000 the batch is a Gaussian blob (mean radius of")
    print("a 2-D standard normal is 1.25). Step by step it tightens onto the")
    print("ring. The model never saw a ring at sampling time; it only ever")
    print("learned 'what noise was added'. Undo that 1000 times and the data")
    print("distribution falls out.")


def demo_5_break_it(model, data, sched, steps):
    """Deliberate damage, each with the correct number beside it."""
    betas, alphas, alpha_bar = sched
    T = len(betas) - 1

    line("DEMO 5a: BREAK IT -- forget the square roots in the jump")
    x0 = np.array([[1.0, 0.0]]).repeat(20000, axis=0)
    print("x_0 = (1, 0), x-coordinate of x_t:\n")
    print("     t   correct  mean   std  |  no-sqrt  mean   std  |  theory  mean   std")
    for t in (100, 300):
        eps = RNG.normal(size=x0.shape)
        ab = alpha_bar[t]
        good = q_sample(x0, t, alpha_bar, eps)
        bad = ab * x0 + (1 - ab) * eps                     # the bug
        print(f"  {t:4d}     {good[:, 0].mean():6.3f}  {good[:, 0].std():5.3f}  |"
              f"     {bad[:, 0].mean():6.3f}  {bad[:, 0].std():5.3f}  |"
              f"       {np.sqrt(ab):6.3f} {np.sqrt(1 - ab):5.3f}")
    print("\nREAD THIS: alpha_bar and 1 - alpha_bar are VARIANCES. The std is the")
    print("square root. Drop it and x_100 has a third of the noise it should;")
    print("the model then trains on one noise level and samples at another.")

    line("DEMO 5b: BREAK IT -- a schedule that reaches pure noise too early")
    _, _, ab_fast = make_schedule(T, beta_start=1e-4, beta_end=0.2)
    print("beta_end = 0.2 instead of 0.02:\n")
    print("     t   alpha_bar (paper)   alpha_bar (too fast)")
    for t in (100, 250, 500, 1000):
        print(f"  {t:4d}   {alpha_bar[t]:17.6f}   {ab_fast[t]:20.6f}")
    print(f"\n  steps with alpha_bar < 0.001:   paper {np.sum(alpha_bar < 1e-3):4d} of {T}"
          f"   too fast {np.sum(ab_fast < 1e-3):4d} of {T}")
    eps = RNG.normal(size=data.shape)
    for name, ab in (("paper", alpha_bar), ("too fast", ab_fast)):
        xt = q_sample(data, 500, ab, eps)
        c_eps = np.corrcoef(xt[:, 0], eps[:, 0])[0, 1]
        c_x0 = np.corrcoef(xt[:, 0], data[:, 0])[0, 1]
        print(f"  at t = 500, {name:9s} corr(x_t, eps) = {c_eps:.4f}"
              f"   corr(x_t, x_0) = {c_x0:.4f}")
    print("\nREAD THIS: once alpha_bar is ~0, x_t IS eps. The training pair is")
    print("(eps, eps): the model learns to copy its input. Those steps teach it")
    print("nothing about the data, and at sampling time they do nothing useful.")
    print("Even the paper's schedule wastes its last ~15% this way; the cosine")
    print("schedule in Improved DDPM (Nichol & Dhariwal, 2021) exists to fix it.")

    line("DEMO 5c: BREAK IT -- drop the noise term in the reverse step")
    x_det = sample(model, 2000, sched, add_noise=False)
    describe("real data (ring)", data)
    describe("DDPM, with noise (correct)", sample(model, 2000, sched))
    describe("DDPM, noise term removed", x_det)
    describe("DDIM, 50 steps, eta = 0", sample_ddim(model, 2000, alpha_bar, 50))
    print("\nREAD THIS: without the fresh noise every step, the samples shrink")
    print("toward the centre of the data and lose their spread -- the sampler")
    print("chases the average instead of drawing from the distribution. Not")
    print("really broken, just a different sampler. DDIM is the principled")
    print("version of that idea: deterministic, and it gets the spread right in")
    print("50 steps instead of 1000, with the SAME trained model.")

    line("DEMO 5d: BREAK IT -- T too small (T = 5)")
    sched5 = make_schedule(5)
    ab5 = sched5[2][5]
    print(f"alpha_bar_T with T = 5:  {ab5:.4f}   (should be ~0)")
    print(f"noise fraction sqrt(1 - alpha_bar_T):  {np.sqrt(1 - ab5):.3f}"
          "   (should be ~1)\n")
    xT_true = q_sample(data, 5, sched5[2], RNG.normal(size=data.shape))
    describe("true x_T (what q makes)", xT_true)
    describe("N(0, 1) (where sampling starts)", RNG.normal(size=data.shape))
    model5 = NoisePredictor()
    train(model5, data, sched5[2], steps=steps // 2, log_every=steps)
    describe("real data (ring)", data)
    describe("T = 5, start from N(0, 1)", sample(model5, 2000, sched5))
    describe("T = 5, start from true x_T", sample(model5, 2000, sched5,
                                                   x_init=xT_true[:2000]))
    print("\nREAD THIS: with 5 steps the forward process never reaches noise, so")
    print("x_T still looks like the ring. Sampling starts from N(0, 1) anyway --")
    print("a distribution the model never trained on -- and five small undo")
    print("steps cannot close the gap. Hand it the true x_T (cheating: that")
    print("needs the data) and the same model works fine. The sampler was")
    print("never the problem; the starting point was.")

    line("DEMO 5e: BREAK IT -- do not tell the model what t is")
    blind = NoisePredictor(use_t=False)
    train(blind, data, alpha_bar, steps=steps, log_every=steps)
    print(f"\n  held-out loss, all t:         with t {eval_loss(model, data, alpha_bar):.4f}"
          f"   without t {eval_loss(blind, data, alpha_bar):.4f}")
    print(f"  held-out loss, t <= 50 only:  with t "
          f"{eval_loss(model, data, alpha_bar, t_max=50):.4f}"
          f"   without t {eval_loss(blind, data, alpha_bar, t_max=50):.4f}")
    describe("real data (ring)", data)
    describe("generated, blind to t", sample(blind, 2000, sched))
    print("\nREAD THIS: the same point x_t can be 'a ring point plus a little")
    print("noise' or 'mostly noise'. Without t the model has to hedge between")
    print("those, so it can never remove the right amount. Loss plateaus higher")
    print("-- at small t it is worse than predicting zero -- and the samples")
    print("are smeared.")

    line("DEMO 5f: BREAK IT -- train on x_0 targets, keep the eps sampler")
    x0_model = NoisePredictor()
    train(x0_model, data, alpha_bar, steps=steps, target="x0", log_every=steps)
    describe("real data (ring)", data)
    describe("x_0-model, eps sampler", sample(x0_model, 2000, sched))

    class Converted(nn.Module):                              # the one-line fix
        def forward(self, x, t):
            ab = torch.tensor(alpha_bar, dtype=torch.float32)[t][:, None]
            return (x - ab.sqrt() * x0_model(x, t)) / (1 - ab).sqrt()
    describe("x_0-model, converted to eps", sample(Converted(), 2000, sched))
    print("\nREAD THIS: the sampler subtracts what the model returns as NOISE.")
    print("Hand it a guess at x_0 instead and every step removes the wrong")
    print("thing. Convert x_0-hat to eps-hat first (one line of algebra) and it")
    print("works again, differently imperfect. The paper's ablation is why")
    print("everyone predicts eps: under the simple loss it beat predicting the")
    print("posterior mean.")


if __name__ == "__main__":
    t_start = time.time()
    STEPS = 4000
    DATA = make_ring(4096)
    SCHED = make_schedule(1000)

    demo_1_schedule()
    demo_2_jump_equals_loop()
    model = demo_3_train(DATA, SCHED[2], STEPS)
    demo_4_sample(model, DATA, SCHED)
    demo_5_break_it(model, DATA, SCHED, STEPS)

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. schedule   = betas -> alphas -> alpha_bar. Three arrays. Not learned.
  2. forward    = x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps.
                  A T-step for loop with a one-line shortcut to any t.
  3. the model  = eps_theta(x_t, t): "what noise was added?" One job.
  4. the loss   = ||eps - eps_theta(x_t, t)||^2. Plain MSE. Equation 14.
  5. sampling   = start at N(0, I); for t = T..1: subtract the predicted
                  noise, add a little fresh noise back. Algorithm 2.
  6. slow       = T model calls per sample. DDIM: 50, same model.
  7. Stable Diffusion = steps 1-6 in a VAE's latent space, with a text
                  encoder feeding cross-attention in the U-Net.

  Everything else is training machinery: the U-Net, EMA of weights,
  the variational bound in section 3 that the simple loss replaces.
""")
    print(f"total runtime: {time.time() - t_start:.1f}s")
