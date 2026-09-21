## Privacy Accounting and the RDP Order Grid

This page explains how the testbed arrives at the `epsilon_spent` number it reports, and
what the Opacus warning about "the largest alpha" means when you see it. It is aimed at
readers who are new to differential privacy (DP) accounting.

### Why accounting is needed at all

Training a DP model is not a single noisy operation. Every optimizer step clips each
per-sample gradient to `max_grad_norm` and then adds Gaussian noise, so a run with 200
steps performs 200 private operations on the same dataset. Privacy loss accumulates across
them. An *accountant* is the bookkeeping that turns "200 noisy steps" into a single
`(epsilon, delta)` guarantee.

The naive way to add up privacy loss is far too pessimistic. Modern accountants get a much
tighter answer by doing the arithmetic in an intermediate currency and only converting to
`(epsilon, delta)` at the very end. In this testbed that currency is **Rényi differential
privacy (RDP)**.

### What an "order" (alpha) is

RDP is not one guarantee but a family of them, indexed by a parameter called the *order*,
written as alpha. An RDP guarantee at order alpha is a statement about how much the output
distribution can shift when one record changes, measured by the Rényi divergence of that
order.

Two properties make this useful:

- **Composition is addition.** The RDP cost of 200 steps at a given alpha is just 200 times
  the per-step cost at that alpha. No approximation, no loss.
- **Any order converts to `(epsilon, delta)`.** For each alpha there is a formula giving a
  valid `(epsilon, delta)` pair. Different orders give different epsilon values for the
  same mechanism.

So the recipe is: compose cheaply across many orders at once, then convert each one and
keep the best answer.

### Why there is a grid, and why the minimum is safe

Opacus does not solve for the single best alpha analytically. It evaluates a fixed list of
candidate orders — `RDPAccountant.DEFAULT_ALPHAS`, which covers 1.1 through 10.9 in steps
of 0.1, then the integers 12 through 63 — and reports the **smallest** epsilon any of them
produces.

This is the key safety property, and it is worth being precise about:

> Every order in the grid yields a *valid* `(epsilon, delta)` upper bound on its own.
> Taking the minimum picks the tightest of several correct answers. Evaluating fewer orders
> can therefore only make the reported epsilon **larger** than necessary — never smaller.

A narrow grid is conservative, not optimistic. It can cause you to *understate your privacy*
(claim a weaker guarantee than you actually have), which is the safe direction to err. It
cannot cause you to overstate it.

### The "optimal order is the largest alpha" warning

At the shipped configuration (`configs/config.json`: epsilon 0.2, delta 1e-5, 100 epochs,
batch size 8) you will see:

```
UserWarning: Optimal order is the largest alpha. Please consider expanding the range of
alphas to get a tighter privacy bound.
```

Opacus emits this when the winning order is at either end of the grid. Here it is the top
end: the best of the 151 candidates is alpha=63, the largest one available. That is Opacus
saying *"I ran out of grid before I ran out of improvement — there may be a better order
past 63 that I never checked."*

Very strong privacy settings push the optimum upward, so this warning is characteristic of
a small target epsilon rather than a sign that something is wrong.

### What we measured

The obvious question is how much the missing orders are actually costing. Measured directly
against the shipped parameters:

| Order grid | Reported epsilon | Optimal alpha |
| --- | --- | --- |
| `DEFAULT_ALPHAS` (max 63) | 0.196149 | 63 (at the edge) |
| extended to 128 | 0.194530 | 71 |
| extended to 1024 | 0.194530 | 71 |
| extended to 8192 | 0.194530 | 71 |

Three things follow:

1. **The true optimum is alpha=71, and the bound is loose by about 0.83%.** The optimum does
   not run away — extending the grid to 8192 finds nothing that extending it to 128 did not.
2. **The looseness costs no accuracy.** Opacus calibrates the noise multiplier by binary
   search with a default `epsilon_tolerance` of 0.01. The 0.0016 gap is roughly six times
   smaller than that tolerance, so the search lands on the same noise multiplier
   (sigma = 130.0) either way. No extra noise is being added because of the narrow grid.
3. **Nothing is unsound.** Per the minimum property above, 0.196149 is a true upper bound.
   It is simply not the tightest one obtainable.

### Why the testbed leaves the grid alone

Widening the grid would remove the warning and improve the reported number in the fourth
decimal place, while changing nothing about the trained model. That is not worth a behavior
change in privacy-critical code, so `model.py` deliberately uses the Opacus default and
documents why.

The warning is left visible rather than silenced. It is an accurate description of the
situation, and for a testbed that doubles as a teaching tool, an explained warning is more
useful than a hidden one.

`PETs_Testbed/tests/test_dp_privacy.py` turns this reasoning into tests, including one that
fails if the slack ever exceeds 5% — so if a future change to the DP parameters makes the
default grid genuinely inadequate, it will be caught rather than assumed away.

### If you do decide to widen the grid

There is one trap, and this repository has already hit a version of it once (commit
`b44d789`). Opacus takes an alpha grid in two separate places:

- **Calibration** — `make_private_with_epsilon(..., alphas=...)`, which forwards through
  `get_noise_multiplier` to the accountant and determines how much noise is added.
- **Reporting** — `accountant.get_privacy_spent(delta=..., alphas=...)`, which determines
  the epsilon you print.

**Both must receive the same grid, from a single shared constant.** If you calibrate the
noise against one grid and report epsilon from another, the number you publish is not the
guarantee you actually enforced.

Note also that assigning `accountant.alphas = [...]` does nothing. Opacus 1.x ignores that
instance attribute entirely and uses the class-level `DEFAULT_ALPHAS`, so the assignment
looks like it is configuring the accountant while having no effect. A grid built that way
also started at alpha=1.0, which is degenerate — the RDP-to-DP conversion divides by
`(alpha - 1)`. Both hazards are pinned by tests in `test_dp_privacy.py`.

### Where to look in the code

| Concern | Location |
| --- | --- |
| Engine setup, noise calibration, grid rationale | `PETs_Testbed/model.py`, `DPCNNModel.attach_privacy_engine` |
| Per-epoch epsilon reporting | `PETs_Testbed/model.py`, `DPCNNModel._after_epoch` |
| DP parameters (epsilon, delta, max_grad_norm, epochs, batch size) | `configs/` |
| Privacy-mechanic tests | `PETs_Testbed/tests/test_dp_privacy.py` |

Full measurements, method, and the alternatives considered are recorded in
`reports/2026-09-21-rdp-alpha-grid-tightness.md`.

---

This document was written with the assistance of Claude Code (Anthropic, model Claude Opus
5). The assistant measured the Opacus RDP order-grid behavior described above and drafted
this explanation in accordance with the author's instructions. All content, scientific
claims, and conclusions have been reviewed and verified by the authors to ensure accuracy
and originality.
