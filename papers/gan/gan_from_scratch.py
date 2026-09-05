"""
Generative Adversarial Nets -- for programmers, not researchers.

Run it:      python3 gan_from_scratch.py
Debug it:    set a breakpoint in train_step and watch the two networks take
             turns. Then flip one keyword argument in a demo and watch it break.

NumPy for the parts the paper proves in closed form (the best possible
detective, the value of the game when the forger has won). A tiny PyTorch
model, CPU only, for the one part that needs gradients: the training loop.
No datasets -- the "real" data is three bumps on a line that we sample
ourselves, so mode collapse shows up as a count you can read.
"""

import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(1)     # determinism: two runs must print the same thing
np.set_printoptions(precision=3, suppress=True)

EPS = 1e-7                   # numerical hygiene: log(0) is -inf


# ---------------------------------------------------------------------------
# STAGE 0 -- the "real" data. Three bumps on a line. That is p_data.
#
# The bumps sit at -6, -2 and +2 so that an UNTRAINED forger, whose output
# is roughly 0, starts in the gap between two of them -- the way every real
# GAN starts: nowhere near the data. Three bumps, not one, because a forger
# that only learned one bump has "mode collapsed", and with a count per bump
# that shows up as a number rather than a picture.
# ---------------------------------------------------------------------------
MODES = np.array([-6.0, -2.0, 2.0])
MODE_STD = 0.5


def sample_real(n, rng):
    """n draws from the mixture: pick a bump, add Gaussian noise. Shape (n, 1)."""
    which = rng.integers(0, len(MODES), size=n)
    x = MODES[which] + MODE_STD * rng.normal(size=n)
    return x.astype(np.float32).reshape(n, 1)


def gaussian_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def mixture_pdf(x, modes, sd):
    """Closed-form density of an equal-weight mixture. Only demo 1 needs it."""
    return np.mean([gaussian_pdf(x, m, sd) for m in modes], axis=0)


# ---------------------------------------------------------------------------
# STAGE 1 -- what the paper PROVES (section 4), as NumPy on a grid. No nets.
#
# Proposition 1: for a fixed forger G, the best possible detective is a ratio,
#     D*(x) = p_data(x) / (p_data(x) + p_g(x))
# Theorem 1: with that D*, the value of the game is
#     V(D*, G) = -log 4 + 2 * JSD(p_data || p_g)
# so the global minimum is -log 4 = -1.386, reached exactly when p_g = p_data.
# ---------------------------------------------------------------------------
def optimal_discriminator(p_data_x, p_g_x):
    return p_data_x / (p_data_x + p_g_x)


def game_value(p_data_x, p_g_x, d_x, dx):
    """V(D, G) = E_data[log D(x)] + E_g[log(1 - D(x))], as a Riemann sum."""
    d_x = np.clip(d_x, 1e-12, 1 - 1e-12)          # log(0) hygiene, again
    real_term = np.sum(p_data_x * np.log(d_x) * dx)
    fake_term = np.sum(p_g_x * np.log(1 - d_x) * dx)
    return real_term + fake_term


def jsd(p, q, dx):
    """Jensen-Shannon divergence: how far apart two densities are. 0 if equal."""
    m = 0.5 * (p + q)
    def kl(a, b):
        safe_b = np.where(a > 0, b, 1)
        return np.sum(np.where(a > 0, a * np.log(a / safe_b), 0) * dx)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------------------------------------------------------------------------
# STAGE 2 -- the two loss functions. THE WHOLE PAPER is equation 1:
#
#     min_G max_D   E_x[ log D(x) ]  +  E_z[ log(1 - D(G(z))) ]
#
# D wants that big (real -> 1, fake -> 0). G wants the second term small.
# Optimisers minimise, so D's loss is the NEGATIVE of the thing D maximises.
# ---------------------------------------------------------------------------
def log(p):
    return torch.log(p.clamp(EPS, 1 - EPS))


def d_loss(d_real, d_fake):
    """Detective: maximise log D(x) + log(1 - D(G(z)))  ->  minimise minus that."""
    return -(log(d_real) + log(1 - d_fake)).mean()


def g_loss_saturating(d_fake):
    """Forger, as WRITTEN in equation 1: minimise log(1 - D(G(z))).
    When D is confident (D(G(z)) ~ 0) this has almost no gradient. Demo 2."""
    return log(1 - d_fake).mean()


def g_loss(d_fake):
    """Forger, as the paper RECOMMENDS (section 3, the paragraph after eq. 1):
    maximise log D(G(z)) instead. Same fixed point, far bigger gradient early."""
    return -log(d_fake).mean()


# ---------------------------------------------------------------------------
# STAGE 3 -- the two networks. Tiny MLPs; the idea does not need more.
#
#   G: noise (noise_dim)  ->  a point on the line (1)     the forger
#   D: a point on the line (1)  ->  P(it is real) (1)     the detective
# ---------------------------------------------------------------------------
def make_generator(noise_dim, hidden=32):
    return nn.Sequential(nn.Linear(noise_dim, hidden), nn.LeakyReLU(0.2),
                         nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
                         nn.Linear(hidden, 1))


def make_discriminator(hidden=32):
    return nn.Sequential(nn.Linear(1, hidden), nn.LeakyReLU(0.2),
                         nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
                         nn.Linear(hidden, 1), nn.Sigmoid())


def make_opt(params, lr, optim="adam"):
    if optim == "sgd":    # what the paper used: "we used momentum" (Algorithm 1)
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    return torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))   # what DCGAN uses


def grad_norm(net):
    """||gradient|| across every parameter of a network. 0.0 if none flowed."""
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.sqrt(sum((g ** 2).sum() for g in grads)))


# ---------------------------------------------------------------------------
# STAGE 4 -- one round of the game. Algorithm 1 in the paper.
#
# The alternation IS the method. Two losses, two optimisers, taking turns.
# The detective is a learned loss function for the forger: G never sees the
# data. It only ever sees D's opinion of its output.
# ---------------------------------------------------------------------------
def train_step(G, D, opt_G, opt_D, rng, batch, noise_dim,
               k_d=1, k_g=1, g_loss_fn=g_loss, detach=True):
    # Clear both gradient buffers once, at the top of the round.
    opt_D.zero_grad()
    opt_G.zero_grad()

    # (1) THE DETECTIVE'S TURN, k_d times. Real should score 1, fake 0.
    #     fake.detach() cuts the graph: D's loss must not flow back into G.
    for _ in range(k_d):
        real = torch.from_numpy(sample_real(batch, rng))
        fake = G(torch.randn(batch, noise_dim))
        if detach:
            fake = fake.detach()
        loss_d = d_loss(D(real), D(fake))
        loss_d.backward()
        opt_D.step()

    # (2) THE FORGER'S TURN, k_g times. Fresh noise. Push D(G(z)) towards 1.
    #     D's weights get a gradient too, but only opt_G.step() is called,
    #     so D stays put. That is what "taking turns" means in code.
    for _ in range(k_g):
        loss_g = g_loss_fn(D(G(torch.randn(batch, noise_dim))))
        loss_g.backward()
        opt_G.step()

    return -loss_d.item(), grad_norm(G)


def not_a_game_step(G, D, opt_both, rng, batch, noise_dim):
    """THE BUG in demo 7: one loss, one optimiser, no turns, no detach."""
    opt_both.zero_grad()
    real = torch.from_numpy(sample_real(batch, rng))
    fake = G(torch.randn(batch, noise_dim))
    loss_d = d_loss(D(real), D(fake))
    loss = loss_d + g_loss(D(fake))
    loss.backward()
    opt_both.step()
    return -loss_d.item(), grad_norm(G)


# ---------------------------------------------------------------------------
# STAGE 5 -- the loop, plus the numbers printed every few hundred rounds.
# ---------------------------------------------------------------------------
EVAL_NOISE = torch.randn(1000, 8, generator=torch.Generator().manual_seed(123))


def report(G, D, noise_dim, n=1000):
    """D's accuracy on real/fake, and how many fakes land near each bump."""
    rng = np.random.default_rng(123)               # same real batch every time
    with torch.no_grad():
        real = torch.from_numpy(sample_real(n, rng))
        fake = G(EVAL_NOISE[:, :noise_dim])
        acc_real = (D(real) > 0.5).float().mean().item()
        acc_fake = (D(fake) < 0.5).float().mean().item()
    x = fake.numpy().ravel()
    near = [int(np.sum(np.abs(x - m) < 1.0)) for m in MODES]   # within 2 std
    return acc_real, acc_fake, near, x.mean(), x.std()


def train(steps, noise_dim=8, batch=256, lr_d=5e-4, lr_g=2e-3, optim="adam",
          every=500, combined=False, seed=0, **step_kwargs):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    G, D = make_generator(noise_dim), make_discriminator()
    opt_G = make_opt(G.parameters(), lr_g, optim)
    opt_D = make_opt(D.parameters(), lr_d, optim)
    opt_both = make_opt(list(G.parameters()) + list(D.parameters()), lr_g, optim)
    print("   step  D acc real/fake     V    |gradG|  fake std   near -6 / -2 / +2")
    for step in range(1, steps + 1):
        if combined:
            v, gnorm = not_a_game_step(G, D, opt_both, rng, batch, noise_dim)
        else:
            v, gnorm = train_step(G, D, opt_G, opt_D, rng, batch, noise_dim,
                                  **step_kwargs)
        if step % every == 0:
            acc_r, acc_f, near, mean, std = report(G, D, noise_dim)
            print(f"  {step:5d}     {acc_r:.2f} / {acc_f:.2f}   {v:6.3f}   "
                  f"{gnorm:6.3f}     {std:5.2f}     "
                  f"{near[0]:4d} / {near[1]:4d} / {near[2]:4d}")
    acc_r, acc_f, near, mean, std = report(G, D, noise_dim)
    print(f"  final fakes: mean {mean:+.2f}  std {std:.2f}  "
          f"off every bump: {1000 - sum(near)} of 1000")
    return G, D


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def demo_1_the_theory():
    """Section 4 of the paper, as arithmetic on a grid. No networks at all."""
    line("DEMO 1: the best detective is a ratio, and -log 4 is the finish line")

    x = np.linspace(-12, 8, 8001)
    dx = x[1] - x[0]
    pd = mixture_pdf(x, MODES, MODE_STD)
    print(f"-log 4 = {-np.log(4):.3f}   <- the value of the game if the forger wins\n")
    print(f"{'forger':28s} {'D*(-6)':>7s} {'D*(-2)':>7s} {'V(D*,G)':>8s}   -log4 + 2*JSD")
    forgers = {
        "only the bump at -2":  mixture_pdf(x, [-2.0], MODE_STD),
        "all three, twice as wide": mixture_pdf(x, MODES, 2 * MODE_STD),
        "p_data exactly":       pd,
    }
    for name, pg in forgers.items():
        d_star = optimal_discriminator(pd, pg)
        v = game_value(pd, pg, d_star, dx)
        at = lambda pos: d_star[np.searchsorted(x, pos)]
        print(f"{name:28s} {at(-6):7.3f} {at(-2):7.3f} {v:8.3f}   "
              f"{-np.log(4) + 2 * jsd(pd, pg, dx):8.3f}")

    print("\nREAD THIS: D* = p_data / (p_data + p_g). Where the forger never goes")
    print("(x = -6, first row) the detective is certain: 1.000. Where the forger")
    print("matches the data exactly, D* = 0.5 everywhere -- a coin flip -- and")
    print("the value bottoms out at -log 4. The last two columns agree to three")
    print("decimals: that is Theorem 1, checked numerically.")


def demo_2_two_losses():
    """Why the paper swaps G's loss: the gradient, closed form then measured."""
    line("DEMO 2: the forger's two losses -- why the non-saturating trick exists")

    print("Closed form. D = sigmoid(logit). Gradient of G's loss w.r.t. the logit:")
    print("   saturating      log(1 - D)  ->  d/dlogit = -D        -> 0 as D -> 0")
    print("   non-saturating  -log D      ->  d/dlogit = -(1 - D)  -> 1 as D -> 0")
    print(f"\n   {'D(G(z))':>8s}   {'|grad| saturating':>18s}   {'|grad| non-sat':>15s}")
    for d in (0.5, 0.1, 0.01, 0.001):
        print(f"   {d:8.3f}   {d:18.3f}   {1 - d:15.3f}")

    print("\nMeasured. Give the detective a 1000-round head start against a frozen,")
    print("untrained forger, so D is confident. Then ask for G's gradient:")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    G, D = make_generator(8), make_discriminator()
    opt_G, opt_D = make_opt(G.parameters(), 2e-3), make_opt(D.parameters(), 5e-4)
    for _ in range(1000):
        train_step(G, D, opt_G, opt_D, rng, 256, 8, k_g=0)     # D only
    z = torch.randn(256, 8)
    with torch.no_grad():
        print(f"   D(G(z)) after the head start: {D(G(z)).mean().item():.4f}")
    for name, fn in (("saturating (eq. 1)", g_loss_saturating),
                     ("non-saturating (the trick)", g_loss)):
        opt_G.zero_grad()
        fn(D(G(z))).backward()
        print(f"   |gradG| with {name:27s}: {grad_norm(G):.4f}")

    print("\nREAD THIS: same networks, same noise, same D. The loss the paper")
    print("WRITES gives the forger a hundredth of the signal that the loss the")
    print("paper RECOMMENDS gives. Everyone implements the second one. Demo 5")
    print("shows what happens when you don't.")


def demo_3_healthy_run():
    line("DEMO 3: the game, played properly -- 1 D step, 1 G step, 4000 rounds")
    print("near -6 / -2 / +2 = how many of 1000 fakes land within 1.0 of each bump")
    print("V = log D(x) + log(1 - D(G(z))), the paper's value; -1.386 = forger wins\n")
    train(4000, every=500)
    print("\nREAD THIS: fakes spread across all three bumps -- not perfectly evenly,")
    print("but all three, and still evening out. D's accuracy sinks towards a coin")
    print("flip and V sits at -log 4. That is the paper's claim, in six columns.")


def demo_4_mode_collapse():
    line("DEMO 4: mode collapse -- the forger's learning rate x10 (lr_g=2e-2)")
    train(1000, every=100, lr_g=2e-2)
    print("\nREAD THIS: the forger out-runs the detective. It piles all 1000")
    print("samples onto whichever bump D currently rates 'real' (fake std ~0.3,")
    print("one column ~1000, the others 0), D catches up, and it hops to the")
    print("next bump. It never learns to be in three places at once.")


def demo_5_detective_too_strong():
    line("DEMO 5: the detective wins -- momentum SGD, k=5, G's loss as written")
    print("Momentum SGD (Algorithm 1's caption: what the paper used), k=5 D steps")
    print("per G step (Algorithm 1's k; the paper used k=1), and G's loss exactly")
    print("as written in equation 1:\n")
    train(400, every=100, optim="sgd", lr_d=1e-3, lr_g=1e-3, k_d=5,
          g_loss_fn=g_loss_saturating)
    print("\nSame thing, one change: G uses the non-saturating loss instead.\n")
    train(400, every=100, optim="sgd", lr_d=1e-3, lr_g=1e-3, k_d=5)
    print("\nREAD THIS: first run -- D is perfect (1.00 / 1.00), V is ~0 instead of")
    print("-1.386, and the forger has not moved: fake std 0.1, zero near any bump,")
    print("|gradG| ~0.05. A perfect detective is a useless teacher. Second run --")
    print("G moves in the first 100 rounds (onto one bump: SGD plus a strong D is")
    print("still a bad game, which is why demo 3 uses Adam at a gentler D rate).")
    print("Adam would hide the freeze even with the written loss, because Adam")
    print("rescales tiny gradients back up to a full step.")


def demo_6_forgotten_detach():
    line("DEMO 6: the forgotten detach -- D's loss leaking into G")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    G, D = make_generator(8), make_discriminator()
    opt_G = make_opt(G.parameters(), 2e-3)
    real = torch.from_numpy(sample_real(256, rng))
    z = torch.randn(256, 8)

    for detach in (True, False):
        opt_G.zero_grad()
        fake = G(z).detach() if detach else G(z)
        d_loss(D(real), D(fake)).backward()
        print(f"   detach={str(detach):5s}  |gradG| after D's step: {grad_norm(G):.4f}")
        if not detach:
            leaked = torch.cat([p.grad.ravel() for p in G.parameters()])

    opt_G.zero_grad()
    g_loss(D(G(z))).backward()
    own = torch.cat([p.grad.ravel() for p in G.parameters()])
    cos = torch.dot(leaked, own) / (leaked.norm() * own.norm())
    print(f"   cosine(leaked gradient, G's own gradient): {cos.item():+.3f}")

    print("\nNow train 3000 rounds with detach=False and the leak applied:\n")
    train(3000, every=500, detach=False)
    print("\nREAD THIS: without detach, D's loss writes a gradient into G that")
    print("points the exact opposite way to G's own (D's loss wants fakes MORE")
    print("detectable). Applied every round, it costs a whole bump: compare the")
    print("last column with demo 3. Whether it is applied at all depends on where")
    print("your zero_grad sits -- here it is at the top of the round, so it is.")


def demo_7_not_a_game():
    line("DEMO 7: forgetting to alternate -- one combined loss, one optimiser")
    train(1000, every=200, combined=True)
    print("\nREAD THIS: D is now trained to call fakes 0.5 (G's term pulls D(G(z))")
    print("up while D's own term pulls it down; they cancel at a coin flip), and")
    print("G's gradient is zero wherever D says 0.5. So the forger never leaves")
    print("its starting point: fake std 0.00, nothing near any bump, D 1.00 on")
    print("real. Nobody is being pushed. It is not a game any more.")


if __name__ == "__main__":
    demo_1_the_theory()
    demo_2_two_losses()
    demo_3_healthy_run()
    demo_4_mode_collapse()
    demo_5_detective_too_strong()
    demo_6_forgotten_detach()
    demo_7_not_a_game()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. two nets    G: noise -> sample.   D: sample -> P(real).
  2. one game    min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]        (eq. 1)
  3. two turns   D up on real, down on fake; then G up on D(G(z)). Alternate.
  4. detach      D's turn must not reach into G. One word.
  5. the trick   G maximises log D(G(z)), not minimises log(1 - D(G(z))).
  6. the proof   D* = p_data / (p_data + p_g); optimum -log 4 at p_g = p_data.
  7. the catch   unstable: mode collapse, a detective that wins. Not in the
                 paper's theory; very much in its practice, and in your logs.
""")
