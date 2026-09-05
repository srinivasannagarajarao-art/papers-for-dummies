"""
Auto-Encoding Variational Bayes (the VAE) -- for programmers, not researchers.

Run it:      python3 papers/vae/vae_from_scratch.py
Debug it:    set a breakpoint in reparameterise and in elbo_loss, step through.

NumPy for the two ideas that ARE the paper: the reparameterisation trick and
the closed-form KL. Torch only for the training loop, because hand-written
backprop through two MLPs would bury the idea. CPU, tiny, under 30 seconds.
Every function is short. Deliberate bugs are behind flags, clearly labelled.
"""

import time

import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(1)          # deterministic sums; the model is tiny anyway
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- toy data. 512 points on a unit ring, with a little radial noise.
# A ring is 1-dimensional structure living in 2-D space. Random 2-D noise is
# NOT a ring, so "does noise decode to a ring?" is a real generative test.
# ---------------------------------------------------------------------------
def make_ring(n=512, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = 1.0 + noise * rng.standard_normal(n)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


def ring_stats(points, band=0.1):
    """radius mean, radius std, and fraction of points within `band` of r=1."""
    r = np.sqrt((points ** 2).sum(axis=1))
    return r.mean(), r.std(), float(np.mean(np.abs(r - 1.0) < band))


# ---------------------------------------------------------------------------
# STAGE 1 -- THE TRICK. Section 2.4 of the paper.
#
# You want z ~ N(mu, sigma^2), and you want gradients to flow back into mu and
# sigma. A call like rng.normal(mu, sigma) is a black box: the randomness
# happens INSIDE it, so there is no function of mu to differentiate.
#
# Fix: draw the randomness first, from a distribution that has no parameters,
# then turn it into z with plain arithmetic.
#
#     eps ~ N(0, 1)              # the dice roll. No parameters. No gradient.
#     z   = mu + sigma * eps     # add and multiply. Backprop walks through this.
#
# dz/dmu = 1, dz/dsigma = eps. Same distribution for z, but now differentiable.
# The encoder outputs log(sigma^2), not sigma, so sigma = exp(0.5 * log_var)
# is positive by construction and can never go negative.
# ---------------------------------------------------------------------------
def reparameterise(mu, log_var, eps):
    sigma = np.exp(0.5 * log_var)
    return mu + sigma * eps


# ---------------------------------------------------------------------------
# STAGE 2 -- THE TAX. KL( N(mu, sigma^2) || N(0, 1) ), closed form, Appendix B.
#
#     KL = -1/2 * sum( 1 + log(sigma^2) - mu^2 - sigma^2 )
#
# Zero when mu=0 and sigma=1. Grows as the bell curve moves or changes width.
# This is the penalty for a code not looking like standard normal noise.
# Summed over latent dimensions, because the dims are independent.
# ---------------------------------------------------------------------------
def kl_standard_normal(mu, log_var):
    return -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var), axis=-1)


def kl_monte_carlo(mu, log_var, n=200_000, seed=0):
    """The same KL, estimated the slow way: average of log q(z) - log p(z)
    over samples z ~ q. Used only to check the closed form is right."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n,) + np.shape(mu))
    z = reparameterise(mu, log_var, eps)
    log_q = -0.5 * np.log(2 * np.pi) - 0.5 * log_var - 0.5 * eps ** 2
    log_p = -0.5 * np.log(2 * np.pi) - 0.5 * z ** 2
    return np.sum(log_q - log_p, axis=-1).mean()


# ---------------------------------------------------------------------------
# STAGE 3 -- the model. Two tiny MLPs. Appendix C of the paper, shrunk.
# The encoder returns TWO vectors: a mean and a log-variance per latent dim.
# That pair IS the distribution q(z|x). The decoder is an ordinary MLP.
# ---------------------------------------------------------------------------
class TinyVAE(nn.Module):
    def __init__(self, d_in=2, d_hidden=64, d_latent=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(),
                                 nn.Linear(d_hidden, d_hidden), nn.Tanh())
        self.enc_mu = nn.Linear(d_hidden, d_latent)
        self.enc_log_var = nn.Linear(d_hidden, d_latent)
        self.dec = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.Tanh(),
                                 nn.Linear(d_hidden, d_hidden), nn.Tanh(),
                                 nn.Linear(d_hidden, d_in))

    def encode(self, x):
        h = self.enc(x)
        return self.enc_mu(h), self.enc_log_var(h)

    def decode(self, z):
        return self.dec(z)


# ---------------------------------------------------------------------------
# STAGE 4 -- the loss. The bound in section 2.2, negated so we can minimise it.
#
#     -ELBO = -E[ log p(x|z) ]  +  KL( q(z|x) || p(z) )
#           =  reconstruction   +  tax
#
# For a Gaussian decoder with a fixed output std SIGMA_X, -log p(x|z) is
# squared error / (2 * SIGMA_X^2) plus a constant. So "MSE" is a log-likelihood
# in disguise, and SIGMA_X is the exchange rate between the two terms.
# beta=1 is the paper. beta != 1 is the beta-VAE (Higgins et al., 2017).
# ---------------------------------------------------------------------------
SIGMA_X = 0.1


def elbo_loss(x, x_hat, mu, log_var, beta=1.0):
    recon = ((x - x_hat) ** 2).sum(dim=-1) / (2 * SIGMA_X ** 2)
    kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=-1)
    return (recon + beta * kl).mean(), recon.mean(), kl.mean()


# ---------------------------------------------------------------------------
# STAGE 5 -- the training loop. Algorithm 1 (AEVB) in the paper.
# Full batch, one eps per datapoint per step. The paper says L=1 sample per
# datapoint is enough when the minibatch is big (section 2.3).
#
# `bug` switches on one deliberate mistake so the demos can show what breaks:
#   "no_noise"    z = mu               -> a plain autoencoder in disguise
#   "forgot_exp"  z = mu + log_var*eps -> "sigma" is not a std any more
#   "fixed_eps"   one eps, reused every step -> the model memorises the dice
# ---------------------------------------------------------------------------
def train_vae(X, beta=1.0, steps=1500, bug=None, log_every=0, seed=0):
    torch.manual_seed(seed)
    model = TinyVAE()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = torch.tensor(X, dtype=torch.float32)
    eps_fixed = torch.randn(x.shape[0], 2)          # only used by "fixed_eps"

    for step in range(steps + 1):
        mu, log_var = model.encode(x)

        eps = eps_fixed if bug == "fixed_eps" else torch.randn_like(mu)
        if bug == "no_noise":
            z = mu
        elif bug == "forgot_exp":
            z = mu + log_var * eps
        else:
            z = mu + torch.exp(0.5 * log_var) * eps    # the trick, in torch

        x_hat = model.decode(z)
        loss, recon, kl = elbo_loss(x, x_hat, mu, log_var, beta)

        if log_every and step % log_every == 0:
            mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
            print(f"  step {step:5d}  loss {loss.item():8.3f}  "
                  f"recon {recon.item():8.3f}  kl {kl.item():6.3f}  "
                  f"mse {mse:.4f}")
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# ---------------------------------------------------------------------------
# STAGE 6 -- evaluation. Two questions, both answered with numbers:
#   1. can it rebuild its input?         (mse, with fresh noise in z)
#   2. does PRIOR noise decode to data?  (sample z ~ N(0,I), decode, measure)
# Question 2 is the generative claim. A plain autoencoder fails it.
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, X, n_samples=2000, eps=None):
    x = torch.tensor(X, dtype=torch.float32)
    mu, log_var = model.encode(x)
    if eps is None:
        eps = torch.randn_like(mu)
    x_hat_mu = model.decode(mu)                              # the usual report
    x_hat_z = model.decode(mu + torch.exp(0.5 * log_var) * eps)   # what training saw
    z_prior = torch.randn(n_samples, 2, generator=torch.Generator().manual_seed(1))
    samples = model.decode(z_prior).numpy()
    return {
        "mse": ((x - x_hat_mu) ** 2).sum(dim=-1).mean().item(),
        "mse_z": ((x - x_hat_z) ** 2).sum(dim=-1).mean().item(),
        "kl": (-0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(-1)).mean().item(),
        "sigma": torch.exp(0.5 * log_var).mean().item(),
        "mu_abs": mu.abs().mean().item(),
        "recon_std": x_hat_mu.std(dim=0).mean().item(),
        "samples": samples,
        "noise": z_prior.numpy(),
        "ring": ring_stats(samples),
    }


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_1_reparameterisation():
    line("DEMO 1: the reparameterisation trick -- move the dice outside")

    rng = np.random.default_rng(0)
    for mu, sigma in [(0.0, 1.0), (1.5, 0.4), (-2.0, 3.0)]:
        log_var = np.log(sigma ** 2)
        z = reparameterise(mu, log_var, rng.standard_normal(100_000))
        print(f"  want N({mu:5.2f}, {sigma:.2f})   100000 draws of "
              f"mu + sigma*eps:  mean {z.mean():6.3f}  std {z.std():.3f}")

    print("\nnow the gradient. dz/dmu by finite difference, (z(mu+h) - z(mu)) / h")
    print("  'direct' = two separate rng.normal(mu, sigma) calls, fresh dice each")
    mu, sigma, log_var = 1.5, 0.4, np.log(0.4 ** 2)
    eps = rng.standard_normal()                      # ONE dice roll, kept
    for h in (1e-1, 1e-2, 1e-3):
        z0 = reparameterise(mu, log_var, eps)
        z1 = reparameterise(mu + h, log_var, eps)
        direct = (rng.normal(mu + h, sigma, 1000) - rng.normal(mu, sigma, 1000)) / h
        print(f"  h={h:<6} reparam: {(z1 - z0) / h:6.3f}   direct once: {direct[0]:9.3f}"
              f"   direct rms/1000: {np.sqrt(np.mean(direct ** 2)):7.1f}")
    lv_h = np.log((sigma + 1e-3) ** 2)
    d_sigma = (reparameterise(mu, lv_h, eps) - reparameterise(mu, log_var, eps)) / 1e-3
    print(f"  dz/dsigma, reparameterised: {d_sigma:.3f}   and eps was {eps:.3f}")

    print("\nREAD THIS: same distribution either way. But the direct sample")
    print("re-rolls the dice on every call, so 'nudge mu, see how z moves' is")
    print("noise of size ~sigma divided by h. Shrink h and it gets WORSE, not")
    print("better: there is no derivative there to find. Reparameterised, the")
    print("dice sit outside and dz/dmu is exactly 1, dz/dsigma is exactly eps.")
    print("That is the entire trick. It is what lets loss.backward() reach mu.")


def demo_2_kl_closed_form():
    line("DEMO 2: the KL tax -- 0 at N(0,1), grows as you move away")

    print(f"  {'mu':>5} {'sigma':>6}   {'closed form':>11}   {'monte carlo':>11}")
    for mu, sigma in [(0.0, 1.0), (1.0, 1.0), (0.0, 2.0), (0.0, 0.5),
                      (2.0, 0.5), (0.0, 0.1), (3.0, 1.0)]:
        mu_v, lv_v = np.array([mu]), np.array([np.log(sigma ** 2)])
        cf = kl_standard_normal(mu_v, lv_v) + 0.0     # + 0.0 turns -0.0 into 0.0
        mc = kl_monte_carlo(mu_v, lv_v)
        print(f"  {mu:5.1f} {sigma:6.1f}   {cf:11.4f}   {mc:11.4f}")

    print("\nREAD THIS: the two columns agree, so the one-line formula really is")
    print("the KL. It is 0 only at (0, 1). Move the mean, or make the bell")
    print("curve wider OR narrower than 1, and the tax rises. Narrow is taxed")
    print("too -- a VAE cannot turn itself into a plain autoencoder for free.")


def demo_3_train(X):
    line("DEMO 3: train it. Watch reconstruction fall and the KL settle")

    model = train_vae(X, beta=1.0, log_every=250)
    ev = evaluate(model, X)
    d = ring_stats(X)
    s = ev["ring"]

    print(f"\n  reconstruction mse, z = mu (no noise):      {ev['mse']:.4f}")
    print(f"  reconstruction mse, z sampled from q(z|x): {ev['mse_z']:.4f}")
    print(f"  step 0 was 1.0640 -- the decoder predicting the origin for everything")
    print(f"  encoder's mean sigma {ev['sigma']:.3f}, mean |mu| {ev['mu_abs']:.3f}"
          f", KL per point {ev['kl']:.3f} nats")

    print("\n  THE GENERATIVE CLAIM. Draw z ~ N(0, I) -- never seen any data --")
    print("  decode 2000 of them, and measure whether they land on the ring:")
    print(f"  {'':22} {'radius mean':>11} {'radius std':>10} {'on ring':>8}")
    n = ring_stats(ev["noise"])
    print(f"  {'the data':22} {d[0]:11.3f} {d[1]:10.3f} {d[2]:8.3f}")
    print(f"  {'raw noise, no decoder':22} {n[0]:11.3f} {n[1]:10.3f} {n[2]:8.3f}")
    print(f"  {'decoded prior noise':22} {s[0]:11.3f} {s[1]:10.3f} {s[2]:8.3f}")
    print(f"\n  first 5 decoded noise points:")
    for p in ev["samples"][:5]:
        print(f"    z -> ({p[0]:6.3f}, {p[1]:6.3f})   radius {np.hypot(*p):.3f}")

    print("\nREAD THIS: the decoder has only ever seen z's that came from real")
    print("points. It still turns brand-new standard-normal noise into points")
    print("on the ring, because the KL forced the codes to look like that noise.")
    print("'on ring' = fraction within 0.1 of radius 1.")
    return ev


def demo_4_break_it(X, ev_good):
    line("DEMO 4: break it -- six edits, same measurements, side by side")

    configs = [("beta=0, no KL", dict(beta=0.0)),
               ("beta=4 (beta-VAE)", dict(beta=4.0)),
               ("beta=50, huge KL", dict(beta=50.0)),
               ("z = mu, no noise", dict(bug="no_noise")),
               ("forgot the exp", dict(bug="forgot_exp")),
               ("one eps, reused", dict(bug="fixed_eps"))]
    models = [(name, train_vae(X, **kw)) for name, kw in configs]
    rows = [("beta=1 (the paper)", ev_good)]
    rows += [(name, evaluate(m, X)) for name, m in models]

    print(f"  {'':20} {'mse':>7} {'kl':>7} {'sigma':>7} {'|mu|':>6} "
          f"{'recon std':>9} {'on ring':>8}")
    for name, ev in rows:
        print(f"  {name:20} {ev['mse']:7.4f} {ev['kl']:7.3f} {ev['sigma']:7.3f} "
              f"{ev['mu_abs']:6.2f} {ev['recon_std']:9.3f} {ev['ring'][2]:8.3f}")
    print(f"  {'(data itself)':20} {'':>7} {'':>7} {'':>7} {'':>6} "
          f"{X.std(axis=0).mean():9.3f} {ring_stats(X)[2]:8.3f}")
    print(f"  {'(raw noise, undecoded)':20} {'':>7} {'':>7} {'':>7} {'':>6} "
          f"{'':>9} {ring_stats(ev_good['noise'])[2]:8.3f}")
    print("  mse is with z = mu. 'recon std' = spread of the reconstructions.")

    # the fixed-eps model, scored with the eps it memorised vs fresh eps
    fixed = dict(models)["one eps, reused"]
    torch.manual_seed(0)
    _ = TinyVAE()                                     # burn the same RNG calls
    eps_memorised = torch.randn(X.shape[0], 2)
    with_own = evaluate(fixed, X, eps=eps_memorised)["mse_z"]
    with_new = evaluate(fixed, X)["mse_z"]
    print(f"\n  'one eps, reused': mse with the eps it trained on  {with_own:.4f}")
    print(f"  'one eps, reused': mse with fresh eps               {with_new:.4f}")
    print(f"  'beta=1 (the paper)': mse with fresh eps            {ev_good['mse_z']:.4f}")

    # the forgot-exp model: what is it actually multiplying eps by?
    with torch.no_grad():
        _, lv = dict(models)["forgot the exp"].encode(
            torch.tensor(X, dtype=torch.float32))
    print(f"\n  'forgot the exp': the number multiplying eps is now the raw output:"
          f"\n     min {lv.min().item():.4f}   mean {lv.mean().item():.4f}"
          f"   max {lv.max().item():.4f}   (the KL column read it as log_var=0)")

    print("\nREAD THIS, row by row:")
    print("  beta=0     best mse in the table, worst 'on ring'. Nothing taxes a")
    print("             narrow bell curve, so sigma -> 0.005: every code is a")
    print("             needle on a thin curve. Prior noise misses that curve.")
    print("  beta=4     the beta-VAE dial. Samples still land (a touch fewer);")
    print("             the price is a blurrier reconstruction, mse up ~40%.")
    print("             Push beta past ~8 on this toy and the samples go too.")
    print("  beta=50    posterior collapse. sigma -> 1, mu -> 0, KL -> 0: the")
    print("             encoder says nothing, the decoder ignores z, and every")
    print("             reconstruction is nearly the same point (recon std ~ 0).")
    print("  no noise   a plain autoencoder. sigma is unused, so the KL parks it")
    print("             at exactly 1.000 for free. Reconstructs fine; samples do")
    print("             not, because the decoder never saw a noisy code.")
    print("  forgot exp the multiplier is a raw network output, so the network")
    print("             drives it to ~0 (even negative) to kill the noise. The")
    print("             KL still thinks sigma=1. Two halves of the loss disagree.")
    print("  fixed eps  it fit the one dice roll: 3x better with that eps than")
    print("             with any other. The paper's estimator is unbiased only")
    print("             because eps is drawn fresh every step (section 2.3).")


if __name__ == "__main__":
    t0 = time.time()
    X = make_ring()
    demo_1_reparameterisation()
    demo_2_kl_closed_form()
    ev_good = demo_3_train(X)
    demo_4_break_it(X, ev_good)

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. encoder   x -> (mu, log_var)          a bell curve per input, not a point
  2. trick     z = mu + exp(log_var/2)*eps  eps ~ N(0,1) drawn OUTSIDE (sec 2.4)
  3. decoder   z -> x_hat                   an ordinary MLP
  4. loss      recon(x, x_hat) + KL(q(z|x) || N(0,I))    = -ELBO, sec 2.2
  5. KL        -1/2 sum(1 + log_var - mu^2 - exp(log_var))  closed form, App. B
  6. train     Adam on that loss. Backprop reaches the encoder because of 2.
  7. generate  z ~ N(0,I), decode. Works because 5 made the codes look like z.

  Everything else in the paper is the derivation of why 4 is a lower bound on
  log p(x). Real, but you can use the loss without reproving it.
""")
    print(f"total runtime: {time.time() - t0:.1f} s")
