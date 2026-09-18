# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software
#
# This file was created with the assistance of Claude Code (Anthropic, model
# Claude Opus 4.8). The assistant wrote privacy-mechanic characterization tests
# for the Opacus RDP accountant behavior that model.py relies on, in accordance
# with the author's instructions. All content has been reviewed and verified by
# the authors.
#
# Purpose: guard the differential-privacy invariants around the DP-CNN training
# path. model.py deliberately does NOT customize the RDP order grid; it lets the
# Opacus accountant calibrate noise and report (epsilon, delta) over its built-in
# DEFAULT_ALPHAS. These tests pin that contract so a future change cannot
# silently reintroduce a mismatched or degenerate alpha grid.

import numpy as np
import torch
from opacus.accountants import RDPAccountant
from torch.utils.data import DataLoader, TensorDataset

from model import DPCNNModel


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
