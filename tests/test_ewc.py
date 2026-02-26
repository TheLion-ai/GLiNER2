"""
Tests for Elastic Weight Consolidation (EWC) implementation.

These tests use a tiny synthetic model and a small in-memory dataset so that
no network access or large checkpoints are required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from gliner2.training.ewc import EWC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal model that mimics the Extractor forward interface."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, batch):
        x = batch["x"].float()
        logits = self.linear(x)
        # Simple MSE loss so that the tensor always has requires_grad=True
        loss = ((logits - batch["y"].float()) ** 2).mean()
        return {"total_loss": loss}

    def named_parameters(self, *args, **kwargs):
        return super().named_parameters(*args, **kwargs)


class _SyntheticDataset(Dataset):
    """Tiny in-memory dataset of random tensors."""

    def __init__(self, n: int = 8, seed: int = 0):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.x = torch.randn(n, 4, generator=rng)
        self.y = torch.randn(n, 2, generator=rng)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return {"x": self.x[idx], "y": self.y[idx]}


def _collate(batch):
    """Simple collator that stacks tensors."""
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


def _make_ewc(n_samples: int = 8, ewc_lambda: float = 10.0, normalize: bool = True) -> tuple:
    """Return (model, dataset, ewc) for use in tests."""
    model = _TinyModel()
    dataset = _SyntheticDataset(n=n_samples)
    ewc = EWC(
        model=model,
        dataset=dataset,
        data_collator=_collate,
        device=torch.device("cpu"),
        ewc_lambda=ewc_lambda,
        batch_size=2,
        num_samples=n_samples,
        normalize_fisher=normalize,
    )
    return model, dataset, ewc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ewc_initialisation():
    """EWC can be initialized with a model and synthetic dataset."""
    model, dataset, ewc = _make_ewc()

    assert ewc.ewc_lambda == 10.0
    assert len(ewc.old_params) > 0
    assert len(ewc.fisher_info) > 0

    # old_params should contain all trainable parameters
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(ewc.old_params.keys()) == trainable_names
    assert set(ewc.fisher_info.keys()) == trainable_names


def test_ewc_loss_is_non_negative():
    """ewc_loss() returns a non-negative scalar tensor."""
    model, _, ewc = _make_ewc()

    penalty = ewc.ewc_loss()
    assert isinstance(penalty, torch.Tensor)
    assert penalty.ndim == 0, "EWC loss should be a scalar"
    assert penalty.item() >= 0.0


def test_ewc_loss_zero_when_params_unchanged():
    """EWC penalty is zero when model parameters have not changed."""
    model, _, ewc = _make_ewc()

    # Parameters have not changed since initialisation – penalty should be 0
    penalty = ewc.ewc_loss()
    assert abs(penalty.item()) < 1e-6


def test_ewc_loss_positive_after_param_change():
    """EWC penalty is positive after modifying model parameters."""
    model, _, ewc = _make_ewc()

    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.5)

    penalty = ewc.ewc_loss()
    assert penalty.item() > 0.0


def test_ewc_loss_scales_with_lambda():
    """EWC penalty scales linearly with ewc_lambda."""
    model, _, ewc = _make_ewc(ewc_lambda=1.0)

    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.1)

    loss1 = ewc.ewc_loss().item()
    ewc.update_lambda(10.0)
    loss10 = ewc.ewc_loss().item()

    assert abs(loss10 - 10.0 * loss1) < 1e-4


def test_consolidate_updates_fisher_and_params():
    """consolidate() blends Fisher estimates and refreshes old_params."""
    model, dataset, ewc = _make_ewc()

    old_fisher = {k: v.clone() for k, v in ewc.fisher_info.items()}
    old_params = {k: v.clone() for k, v in ewc.old_params.items()}

    # Modify parameters so that old_params must change after consolidation
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.2)

    ewc.consolidate(dataset, alpha=0.5)

    # Fisher values should have been blended (may differ from original)
    for name in ewc.fisher_info:
        # After blending with new Fisher, values may change
        assert ewc.fisher_info[name].shape == old_fisher[name].shape

    # old_params should now reflect the updated model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert torch.allclose(ewc.old_params[name], param.data)


def test_get_importance_scores_returns_float_dict():
    """get_importance_scores() returns a dict with float values."""
    _, _, ewc = _make_ewc()

    scores = ewc.get_importance_scores()
    assert isinstance(scores, dict)
    assert len(scores) > 0
    for group, value in scores.items():
        assert isinstance(group, str)
        assert isinstance(value, float)


def test_update_lambda():
    """update_lambda() updates the ewc_lambda attribute."""
    _, _, ewc = _make_ewc(ewc_lambda=50.0)

    ewc.update_lambda(200.0)
    assert ewc.ewc_lambda == 200.0


def test_fisher_normalization():
    """When normalize_fisher=True, all Fisher values are in [0, 1]."""
    _, _, ewc = _make_ewc(normalize=True)

    for name, fisher in ewc.fisher_info.items():
        assert fisher.min().item() >= 0.0 - 1e-6, f"Fisher min < 0 for {name}"
        assert fisher.max().item() <= 1.0 + 1e-6, f"Fisher max > 1 for {name}"


def test_ewc_loss_requires_grad():
    """EWC loss should allow gradients to flow through model parameters."""
    model, _, ewc = _make_ewc()

    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.1)

    penalty = ewc.ewc_loss()
    # Penalty should be differentiable w.r.t. model parameters
    penalty.backward()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
