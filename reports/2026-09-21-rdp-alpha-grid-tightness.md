# Tightness of the Opacus default RDP order grid

**Date:** 2026-09-21
**Branch:** `alpha-grid-tightness`
**Status:** Resolved — documented, not changed (Option A)
**Scope:** Measurement and analysis only. No change to noise calibration, clipping bounds,
the accountant, or budget bookkeeping.

---

## 1. Question

Running `test_dp_privacy.py` emits a warning from Opacus 1.4.1:

```
opacus/accountants/analysis/rdp.py:332: UserWarning: Optimal order is the largest alpha.
Please consider expanding the range of alphas to get a tighter privacy bound.
```

The optimal Rényi order lands on the top edge of `RDPAccountant.DEFAULT_ALPHAS` (orders
1.1–10.9 in steps of 0.1, then integers 12–63), so Opacus cannot rule out a better order
beyond its grid. The warning has been present but unquantified since the `b44d789` work,
which established that Opacus ignores an instance-level `accountant.alphas` assignment.

Two things needed answering:

1. Does this threaten the `(epsilon, delta)` guarantee?
2. If not, how much tightness — and how much model accuracy — is it costing?

## 2. Method

All measurements used the project virtual environment (Python 3.12.8, Opacus 1.4.1,
torch 2.13.0, MacOS 27.0) and called Opacus directly. No repository file was modified during
measurement and no project data was read.

Parameters were taken from `configs/config.json`: `epsilon=0.2`, `delta=1e-5`,
`epochs=100`, `batch_size=8`, `num_clients=5`, `test_fraction=0.3`. The SCC training array
(`data/SCC/SCC_tt_vcf.npy`) has 84 rows; split across 5 clients and after the test holdout
that is roughly 12 training rows per client, so a batch size of 8 gives about 2 batches per
epoch — a **sample rate near 0.5** and **200 optimizer steps**. Calibration at those
settings yields a noise multiplier of **sigma = 130.0**.

Epsilon was recomputed from a synthetic accountant history, `[(sigma, sample_rate, steps)]`,
which is exactly what `RDPAccountant` accumulates during a real run. This makes the
measurement deterministic and independent of the dataset.

## 3. Results

### 3.1 Effect of widening the grid — shipped configuration

sigma = 130.0, sample rate 0.5, 200 steps, delta = 1e-5:

| Order grid | Reported epsilon | Optimal alpha | Warning |
| --- | --- | --- | --- |
| `DEFAULT_ALPHAS` (max 63) | 0.196149 | 63 (at edge) | yes |
| extended to 128 | 0.194530 | 71 | no |
| extended to 1024 | 0.194530 | 71 | no |
| extended to 8192 | 0.194530 | 71 | no |

**Relative slack: 0.83%.** The optimum converges at alpha = 71; extending the grid to 8192
finds nothing beyond what extending to 128 finds.

### 3.2 Effect at a lower sample rate

A larger-cohort regime (sigma = 18.75, sample rate 0.01, 10 000 steps), representing what a
bigger federation would look like:

| Order grid | Reported epsilon | Optimal alpha |
| --- | --- | --- |
| `DEFAULT_ALPHAS` | 0.192750 | 63 (at edge) |
| extended to 128 | 0.190682 | 72 |

**Relative slack: 1.08%.** Same shape, same magnitude.

### 3.3 Effect on calibrated noise — none

| Target epsilon | sigma, `DEFAULT_ALPHAS` | sigma, extended to 128 | Change |
| --- | --- | --- | --- |
| 0.2 | 130.0000 | 130.0000 | 0.00% |
| 1.0 | 28.7500 | 28.7500 | 0.00% |
| 8.0 | 4.6045 | 4.6045 | 0.00% |

`get_noise_multiplier` binary-searches sigma with a default `epsilon_tolerance` of 0.01. The
0.0016 epsilon gap from §3.1 is roughly six times smaller than that tolerance, so the search
terminates at the same sigma under either grid. **The narrow grid adds no noise and costs no
model accuracy.**

### 3.4 The warning is regime-specific

At a moderate budget (sigma = 1.5, sample rate 0.05, 50 steps) the optimal order is
alpha = 10.7 — comfortably interior — reported epsilon is 1.400557, widening the grid
changes it by exactly 0.0%, and no warning fires. The boundary condition is a consequence of
the very strong `epsilon=0.2` target, not a general property of `DEFAULT_ALPHAS`.

### 3.5 Mechanism check: a custom grid does reach calibration

Relevant if the grid is ever widened. Source reading gives the path
`make_private_with_epsilon(**kwargs)` → `get_noise_multiplier(**kwargs)` →
`accountant.get_epsilon(delta, **kwargs)`, meaning `alphas=` passed to the engine reaches
noise calibration.

Because passing a *widened* grid changes nothing observable (§3.3), that alone does not
prove the kwarg is honored rather than silently discarded. Passing a deliberately crippled
grid does:

```
make_private_with_epsilon(..., alphas=[2.0, 3.0])
  -> ValueError: The privacy budget is too low.
```

Two orders are too few to reach epsilon = 0.2 within `MAX_SIGMA`, so the failure confirms
the grid is genuinely consumed by the calibration search.

## 4. Analysis

**No soundness issue exists.** The RDP-to-DP conversion reports the *minimum* epsilon over
the grid. Each order independently yields a valid upper bound, so a minimum over a subset of
orders can only overstate epsilon, never understate it. `DEFAULT_ALPHAS` is conservative:
the reported 0.196149 is a true `(epsilon, delta)` guarantee, merely not the tightest one
available. Erring in this direction means claiming *weaker* privacy than is actually held.

**The cost is confined to the reported number.** Slack is under 1.1% in both tested regimes,
and because it sits inside the calibration tolerance, sigma — and therefore model utility —
is unaffected.

## 5. Options considered

| Option | Effect | Cost |
| --- | --- | --- |
| **A. Document and test** | Warning stays visible and explained; slack bounded by a test | None; no behavior change |
| **B. Widen the grid to 128** | Warning disappears; reported epsilon 0.196149 → 0.194530 | Behavior change in the privacy-critical protected zone; two call sites must be kept in lockstep |
| **C. Make the grid config-driven** | Grid becomes an experiment parameter | Exposes a DP knob to user input; needs guards against `alpha <= 1` and boundary optima |

**Option A was selected.** Option B buys a fourth decimal place in a reported number and
changes nothing about the trained model, which does not justify editing privacy-critical
code. Option C is a reasonable future enhancement but should not be built before the
behavior is documented and pinned.

Option B additionally carries the hazard that produced commit `b44d789`: Opacus accepts an
alpha grid at **two** sites — `make_private_with_epsilon(alphas=...)` for calibration and
`accountant.get_privacy_spent(delta=..., alphas=...)` for reporting. If those diverge, the
published epsilon is not the guarantee enforced. Any future implementation must drive both
from one shared constant.

## 6. What was changed

Documentation and tests only.

- `PETs_Testbed/model.py` — comment block in `attach_privacy_engine` expanded to record the
  finding and the lockstep requirement. **Comment text only; no executable line changed.**
- `PETs_Testbed/tests/test_dp_privacy.py` — four tests added:
  - `test_shipped_config_regime_pins_optimum_to_grid_upper_bound` — asserts the warning and
    the boundary optimum, so the behavior is characterized rather than incidental.
  - `test_widening_alpha_grid_leaves_epsilon_bound_nearly_unchanged` — fails if slack exceeds
    5% in either regime. This is the alarm that makes Option A safe over time.
  - `test_widening_alpha_grid_does_not_change_calibrated_noise` — pins the zero-utility-cost
    result of §3.3.
  - `test_moderate_privacy_regime_has_interior_optimum` — pins the contrast case of §3.4.
- `docs/privacy_accounting.md` — new explainer covering RDP orders, why the minimum over a
  grid is safe, and what the warning means.
- `.gitignore` — narrowed so Markdown reports under `reports/` are tracked while generated
  run output stays ignored.

### Privacy impact statement

**The `(epsilon, delta)` guarantee is unchanged, and unchanged by construction rather than
by argument.** No executable code was modified. Noise scale (sigma = 130.0 for the shipped
config), the clipping bound (`max_grad_norm = 1.0`, which sets the per-sample sensitivity the
noise is calibrated against), the accountant (`rdp`), the order grid (`DEFAULT_ALPHAS`), and
composition over 200 steps are all byte-for-byte as before. The relationship between noise
scale and sensitivity is untouched: Opacus continues to clip each per-sample gradient to
`max_grad_norm` and add Gaussian noise of scale `sigma * max_grad_norm`, and continues to
compose and report over the same grid used to calibrate it. The sole `model.py` edit is
comment text.

## 7. Verification

- `pytest PETs_Testbed/tests/test_dp_privacy.py -v` — 8 passed (4 pre-existing, 4 new).
- Full suite — 213 passed, 3 skipped, 2 deselected.
- `git diff PETs_Testbed/model.py` — comment lines only.

## 8. Follow-up worth separating

The shipped configuration puts **sample rate at roughly 0.5** — each "batch" is about half a
client's data, because 84 rows across 5 clients leaves ~12 training rows each against a batch
size of 8. Reaching epsilon = 0.2 in that regime requires sigma = 130.0, which is an enormous
amount of noise relative to the signal.

This is a far larger lever on the privacy/utility curve than the alpha grid and is worth
characterizing on its own: a sweep over `batch_size`, `num_clients`, and target epsilon,
reporting sigma and achieved accuracy, would say more about the trade-off this testbed exists
to study than anything in this report. Filed here as an observation, not acted on.

---

This report was written with the assistance of Claude Code (Anthropic, model Claude Opus 5).
The assistant designed and ran the measurements described above, analyzed the results, and
drafted the report in accordance with the author's instructions. All content, scientific
claims, and conclusions have been reviewed and verified by the authors to ensure accuracy
and originality.
