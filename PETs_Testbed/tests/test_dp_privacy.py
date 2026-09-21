# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software
#
# This file was created with the assistance of Claude Code (Anthropic, models
# Claude Opus 4.8 and Claude Opus 5). The assistant wrote privacy-mechanic
# characterization tests for the Opacus RDP accountant behavior that model.py
# relies on, in accordance with the author's instructions. All content has been
# reviewed and verified by the authors.
#
# Purpose: guard the differential-privacy invariants around the DP-CNN training
# path. model.py deliberately does NOT customize the RDP order grid; it lets the
# Opacus accountant calibrate noise and report (epsilon, delta) over its built-in
# DEFAULT_ALPHAS. These tests pin that contract so a future change cannot
# silently reintroduce a mismatched or degenerate alpha grid, and they bound how
# much tightness that choice gives up. See docs/privacy_accounting.md.

import warnings

import numpy as np
import pytest
import torch
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader, TensorDataset

from model import DPCNNModel

# The shipped configuration (configs/config.json: epsilon=0.2, delta=1e-5,
# epochs=100, batch_size=8) resolves to roughly this accountant history once the
# ~84-row training set is split across 5 clients: a sample rate near 0.5 and 200
# optimizer steps, calibrated to a noise multiplier of 130.0. Expressing it as a
# history keeps these tests fast and free of any dependency on the real dataset.
SHIPPED_SIGMA = 130.0
SHIPPED_SAMPLE_RATE = 0.5
SHIPPED_STEPS = 200
DELTA = 1e-5

# An order grid extended past the Opacus default ceiling of 63. The optimum for
# this testbed's regimes settles around alpha=71, so 128 is comfortably wide
# enough to locate it; see test_widening_alpha_grid_leaves_epsilon_bound_nearly_unchanged.
WIDENED_ALPHAS = RDPAccountant.DEFAULT_ALPHAS + list(range(64, 129))

# How much looseness we are willing to accept from the default grid before this
# becomes a real problem rather than a documented curiosity. Measured slack today
# is 0.83% (shipped regime) and 1.08% (a lower sample rate).
MAX_ACCEPTABLE_SLACK = 0.05


def _spend(sigma, sample_rate, steps, alphas=None):
    """Report (epsilon, best_alpha) for a synthetic accountant history."""
    accountant = RDPAccountant()
    accountant.history = [(sigma, sample_rate, steps)]
    if alphas is None:
        return accountant.get_privacy_spent(delta=DELTA)
    return accountant.get_privacy_spent(delta=DELTA, alphas=alphas)


# --------------------------------------------------------------------------- #
# The RDP order grid must never contain a degenerate order (alpha <= 1).       #
# The RDP-to-DP conversion divides by (alpha - 1), so alpha == 1 is a          #
# ZeroDivisionError and alpha < 1 is not a valid Renyi order. A previous       #
# no-op override in model.py built a grid starting at alpha == 1.0; this test  #
# documents why any such grid is invalid.                                      #
# --------------------------------------------------------------------------- #
def test_default_rdp_alphas_exclude_degenerate_orders():
    assert all(alpha > 1 for alpha in RDPAccountant.DEFAULT_ALPHAS)
    assert 1.0 not in RDPAccountant.DEFAULT_ALPHAS


def test_degenerate_alpha_grid_would_break_accounting():
    # Characterization: the exact grid the deleted line built (orders starting
    # at 1.0) cannot be used for accounting -- it raises rather than reporting a
    # privacy budget. This is why it must never be wired into get_privacy_spent.
    acc = RDPAccountant()
    acc.step(noise_multiplier=1.5, sample_rate=0.05)
    bad_grid = [1 + x / 10.0 for x in range(1000)]  # starts at alpha == 1.0
    assert bad_grid[0] == 1.0
    with np.testing.assert_raises(ZeroDivisionError):
        acc.get_privacy_spent(delta=1e-5, alphas=bad_grid)


# --------------------------------------------------------------------------- #
# Reported epsilon depends only on the default (or explicitly passed) grid --  #
# never on an instance `alphas` attribute. This is the property that made the  #
# removed model.py line a no-op, and it must stay true for the removal to be   #
# behavior-preserving.                                                         #
# --------------------------------------------------------------------------- #
def test_reported_epsilon_ignores_instance_alphas_attribute():
    def spend_after_setting_attribute(attr_value):
        acc = RDPAccountant()
        for _ in range(50):
            acc.step(noise_multiplier=1.5, sample_rate=0.05)
        if attr_value is not None:
            acc.alphas = attr_value  # mimic the old model.py override
        return acc.get_privacy_spent(delta=1e-5)

    baseline_eps, baseline_alpha = spend_after_setting_attribute(None)
    # Even a wildly different (and invalid) attribute must not change the result.
    override_eps, override_alpha = spend_after_setting_attribute(
        [1 + x / 10.0 for x in range(1000)]
    )
    assert override_eps == baseline_eps
    assert override_alpha == baseline_alpha


# --------------------------------------------------------------------------- #
# The production path: attach_privacy_engine must not inject a custom alpha     #
# grid onto the accountant. If someone reintroduces the override, this fails.   #
# --------------------------------------------------------------------------- #
def test_attach_privacy_engine_does_not_customize_alpha_grid():
    torch.manual_seed(0)
    num_features = 64
    num_samples = 32
    batch_size = 8

    features = torch.randn(num_samples, num_features)
    labels = torch.randn(num_samples)
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)

    model = DPCNNModel(model_id=0, output_dir=None, problem_type="regression")
    model.build_model(num_features=num_features, output_dim=1)
    optimizer = model.build_optimizer("sgd", learning_rate=0.01, weight_decay=0.0)

    opacus_params = {
        "secure_mode": False,
        "epochs": 1,
        "epsilon": 1.0,
        "delta": 1e-5,
        "max_grad_norm": 1.0,
    }

    _, _, privacy_engine = model.attach_privacy_engine(optimizer, loader, opacus_params)

    # No per-instance alpha grid should have been injected: the accountant must
    # rely on its DEFAULT_ALPHAS for both calibration and reporting.
    assert not hasattr(privacy_engine.accountant, "alphas")
    assert privacy_engine.accountant.mechanism() == "rdp"


# --------------------------------------------------------------------------- #
# How tight is the bound DEFAULT_ALPHAS gives us?                              #
#                                                                              #
# At the shipped configuration Opacus warns that the optimal order is the       #
# largest alpha, meaning the best Renyi order sits on the top edge of the grid  #
# and a better one may lie beyond it. That is a *tightness* question, not a     #
# soundness one: the RDP-to-DP conversion minimizes over the grid, so a narrow  #
# grid can only overstate epsilon. The tests below pin how much is given up.    #
# --------------------------------------------------------------------------- #
def test_shipped_config_regime_pins_optimum_to_grid_upper_bound():
    # Characterization of the warning's cause. Asserted deliberately rather than
    # suppressed, so that the day it stops happening is a visible event.
    with pytest.warns(UserWarning, match="largest alpha"):
        _, best_alpha = _spend(SHIPPED_SIGMA, SHIPPED_SAMPLE_RATE, SHIPPED_STEPS)

    assert best_alpha == max(RDPAccountant.DEFAULT_ALPHAS) == 63


def test_widening_alpha_grid_leaves_epsilon_bound_nearly_unchanged():
    # The alarm. Widening the grid can only lower the reported epsilon; this
    # bounds how much lower. If a future change to the DP parameters makes the
    # default grid genuinely inadequate, the slack grows and this test fails.
    regimes = [
        (SHIPPED_SIGMA, SHIPPED_SAMPLE_RATE, SHIPPED_STEPS),  # shipped config
        (18.75, 0.01, 10000),                                 # larger cohort
    ]

    for sigma, sample_rate, steps in regimes:
        default_eps, _ = _spend(sigma, sample_rate, steps)
        widened_eps, widened_alpha = _spend(sigma, sample_rate, steps, WIDENED_ALPHAS)

        # A superset grid must never report a looser bound.
        assert widened_eps <= default_eps

        # ...and the optimum must actually be interior to the widened grid,
        # otherwise the comparison says nothing about the true optimum.
        assert widened_alpha < max(WIDENED_ALPHAS)

        slack = (default_eps - widened_eps) / widened_eps
        assert slack < MAX_ACCEPTABLE_SLACK, (
            f"default alpha grid is {slack:.2%} loose at sigma={sigma}, "
            f"sample_rate={sample_rate}, steps={steps}; the grid may need widening"
        )


def test_widening_alpha_grid_does_not_change_calibrated_noise():
    # The reason the slack above costs no utility: get_noise_multiplier searches
    # for sigma with epsilon_tolerance=0.01 by default, and the slack is far
    # inside that tolerance, so the same sigma is selected either way.
    calibration = dict(
        target_epsilon=0.2, target_delta=DELTA,
        sample_rate=SHIPPED_SAMPLE_RATE, epochs=100,
    )

    default_sigma = get_noise_multiplier(**calibration)
    widened_sigma = get_noise_multiplier(**calibration, alphas=WIDENED_ALPHAS)

    assert default_sigma == widened_sigma == SHIPPED_SIGMA


def test_moderate_privacy_regime_has_interior_optimum():
    # Contrast case: the boundary warning is specific to very strong privacy
    # settings, not a universal property of DEFAULT_ALPHAS. At a moderate budget
    # the optimum sits comfortably inside the grid and nothing is warned about.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        epsilon, best_alpha = _spend(1.5, 0.05, 50)

    assert RDPAccountant.DEFAULT_ALPHAS[0] < best_alpha < max(RDPAccountant.DEFAULT_ALPHAS)
    assert epsilon > 0
