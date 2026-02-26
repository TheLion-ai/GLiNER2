"""
Elastic Weight Consolidation (EWC) for Continual Learning in GLiNER2
=====================================================================

Prevents catastrophic forgetting during fine-tuning by penalizing changes
to parameters important for previous tasks, weighted by the diagonal Fisher
information matrix.

Based on: "Overcoming catastrophic forgetting in neural networks"
Paper: https://arxiv.org/abs/1612.00796

Reference implementation:
https://github.com/Knowledgator/GLiClass/blob/7a7bc67e90362de11afdce3c730f382565a54881/gliclass/training.py
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class EWC:
    """
    Elastic Weight Consolidation for continual learning.

    Penalizes changes to parameters that were important for previous tasks,
    using the diagonal Fisher information matrix as importance weights.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be consolidated. Compatible with
        the ``Extractor`` class from ``gliner2.model``.
    dataset : Dataset
        Dataset from the previous task used to compute Fisher information.
    data_collator : Callable
        Collator function that converts dataset batches to model inputs
        (e.g., ``ExtractorCollator`` from ``gliner2.training.trainer``).
    device : torch.device
        Device to run computations on.
    ewc_lambda : float, optional
        Regularisation strength. Larger values mean stronger penalty.
        Default is ``100.0``.
    batch_size : int, optional
        Batch size to use when computing the Fisher matrix. Default is ``1``.
    num_samples : int, optional
        Maximum number of samples to use for Fisher estimation.
        ``None`` means all samples in the dataset. Default is ``None``.
    normalize_fisher : bool, optional
        If ``True``, normalise Fisher values to ``[0, 1]`` after computation.
        Default is ``True``.

    Examples
    --------
    >>> ewc = EWC(model, prev_dataset, collator, device, ewc_lambda=100.0)
    >>> # During training:
    >>> loss = model_loss + ewc.ewc_loss(batch_size=config.batch_size)
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        data_collator: Callable,
        device: torch.device,
        ewc_lambda: float = 100.0,
        batch_size: int = 1,
        num_samples: Optional[int] = None,
        normalize_fisher: bool = True,
    ) -> None:
        self.model = model
        self.data_collator = data_collator
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.normalize_fisher = normalize_fisher

        # Snapshot of model parameters before fine-tuning on new task
        self.old_params: Dict[str, torch.Tensor] = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        logger.info(
            "Computing Fisher information matrix on %d samples (batch_size=%d) ...",
            len(dataset) if num_samples is None else min(num_samples, len(dataset)),
            batch_size,
        )
        self.fisher_info: Dict[str, torch.Tensor] = self._compute_fisher(dataset)

        if normalize_fisher:
            self._normalize_fisher()

        num_params = sum(f.numel() for f in self.fisher_info.values())
        logger.info(
            "EWC initialized: lambda=%.1f, params=%d, normalize=%s",
            ewc_lambda,
            num_params,
            normalize_fisher,
        )

    # -------------------------------------------------------------------------
    # Fisher information
    # -------------------------------------------------------------------------

    def _compute_fisher(self, dataset: Dataset) -> Dict[str, torch.Tensor]:
        """
        Compute the empirical diagonal Fisher information matrix.

        Runs forward/backward passes on a subset of ``dataset`` and
        accumulates squared gradients.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to estimate Fisher information.

        Returns
        -------
        Dict[str, torch.Tensor]
            Per-parameter Fisher diagonal estimates (same shape as each
            parameter tensor).
        """
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Limit samples if requested
        if self.num_samples is not None and self.num_samples < len(dataset):
            indices = list(range(self.num_samples))
            subset = torch.utils.data.Subset(dataset, indices)
        else:
            subset = dataset

        dataloader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
        )

        was_training = self.model.training
        self.model.train()

        num_batches = 0
        for batch in dataloader:
            self.model.zero_grad()
            outputs = self.model(batch)
            loss = outputs["total_loss"]

            if not loss.requires_grad:
                logger.debug("Skipping batch: loss does not require grad")
                continue

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.clone().detach() ** 2

            num_batches += 1

        if was_training:
            self.model.train()
        else:
            self.model.eval()

        # Normalise by number of batches processed
        if num_batches > 0:
            for name in fisher:
                fisher[name] /= num_batches
        else:
            logger.warning("No valid batches processed during Fisher computation")

        return fisher

    def _normalize_fisher(self) -> None:
        """
        Normalise Fisher values to ``[0, 1]`` globally across all parameters.

        Uses the global maximum value across all parameter tensors so that
        relative importances are preserved.
        """
        max_val = max(
            (f.max().item() for f in self.fisher_info.values()),
            default=1.0,
        )
        if max_val > 0:
            for name in self.fisher_info:
                self.fisher_info[name] /= max_val
        else:
            logger.warning("Fisher max value is 0; skipping normalisation")

    # -------------------------------------------------------------------------
    # EWC loss
    # -------------------------------------------------------------------------

    def ewc_loss(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Compute the EWC regularisation penalty.

        .. math::

            \\mathcal{L}_{\\text{EWC}} = \\lambda \\sum_i F_i (\\theta_i - \\theta_i^*)^2

        Parameters
        ----------
        batch_size : int, optional
            Unused; kept for API compatibility. Default is ``None``.

        Returns
        -------
        torch.Tensor
            Scalar EWC penalty tensor (gradient-tracked, on ``self.device``).
        """
        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_info:
                param_diff = param - self.old_params[name].to(param.device)
                fisher = self.fisher_info[name].to(param.device)
                loss = loss + (fisher * param_diff ** 2).sum()
        return self.ewc_lambda * loss

    # -------------------------------------------------------------------------
    # Online EWC consolidation
    # -------------------------------------------------------------------------

    def consolidate(self, dataset: Dataset, alpha: float = 0.5) -> None:
        """
        Online EWC: blend new Fisher estimates into the accumulated ones.

        Computes Fisher on ``dataset``, mixes it with the existing
        ``fisher_info`` using ``alpha``, then updates ``old_params`` to
        the current model weights.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to compute the new Fisher matrix.
        alpha : float, optional
            Blending factor for the new Fisher estimate.
            ``fisher = (1 - alpha) * old + alpha * new``. Default is ``0.5``.
        """
        new_fisher = self._compute_fisher(dataset)

        for name in self.fisher_info:
            if name in new_fisher:
                self.fisher_info[name] = (
                    (1 - alpha) * self.fisher_info[name]
                    + alpha * new_fisher[name]
                )

        if self.normalize_fisher:
            self._normalize_fisher()

        # Snapshot current weights as the new reference point
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone().detach()

        logger.info("EWC consolidation complete (alpha=%.2f)", alpha)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_importance_scores(self) -> Dict[str, float]:
        """
        Return mean Fisher importance score per parameter group.

        Groups are the top-level module names extracted from parameter names
        (e.g. ``"encoder"``, ``"classifier"``).

        Returns
        -------
        Dict[str, float]
            Mapping from group name to mean importance value.
        """
        group_sums: Dict[str, float] = {}
        group_counts: Dict[str, int] = {}

        for name, fisher in self.fisher_info.items():
            group = name.split(".")[0]
            mean_val = fisher.mean().item()
            group_sums[group] = group_sums.get(group, 0.0) + mean_val
            group_counts[group] = group_counts.get(group, 0) + 1

        return {
            group: group_sums[group] / group_counts[group]
            for group in group_sums
        }

    def update_lambda(self, new_lambda: float) -> None:
        """
        Update the EWC regularisation strength.

        Parameters
        ----------
        new_lambda : float
            New value for ``ewc_lambda``.
        """
        logger.info("EWC lambda updated: %.4f -> %.4f", self.ewc_lambda, new_lambda)
        self.ewc_lambda = new_lambda
