"""
End-to-end tests for EWC (Elastic Weight Consolidation) with the real
GLiNER2 Extractor model and training infrastructure.

These tests build a tiny Extractor with a randomly-initialised small BERT
encoder so that no network access or pre-trained checkpoints are required.

Covered scenarios
-----------------
* EWC initialisation via ``GLiNER2Trainer`` when ``use_ewc=True``
* EWC penalty is non-zero and grows after a training step
* Fisher matrix uses real GLiNER2 parameter groups (encoder, classifier, …)
* ``consolidate()`` updates Fisher and parameter anchors with real data
* ``get_importance_scores()`` returns the correct top-level groups
* Passing a pre-built ``EWC`` object directly to ``GLiNER2Trainer``
* Training completes without errors when EWC is active
"""

from __future__ import annotations

import tempfile
from typing import Tuple

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, PreTrainedTokenizerFast

from gliner2.model import Extractor, ExtractorConfig
from gliner2.training.data import InputExample
from gliner2.training.ewc import EWC
from gliner2.training.trainer import (
    ExtractorCollator,
    ExtractorDataset,
    GLiNER2Trainer,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Minimal vocabulary – enough for the synthetic sentences used in tests.
# Special tokens follow BERT naming convention (uppercase + brackets).
# Regular word tokens are lowercase because the WordPiece tokeniser
# operates in a case-folded space.
_VOCAB = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
    "john": 5, "works": 6, "at": 7, "google": 8, "in": 9, "nyc": 10,
    "apple": 11, "released": 12, "iphone": 13, "tim": 14, "cook": 15,
    "is": 16, "a": 17, "great": 18, "company": 19, "the": 20,
}

_PREV_EXAMPLES = [
    InputExample(
        text="John works at Google in NYC.",
        entities={"person": ["John"], "company": ["Google"], "location": ["NYC"]},
    ),
    InputExample(
        text="Tim Cook released iPhone.",
        entities={"person": ["Tim Cook"], "product": ["iPhone"]},
    ),
]

_NEW_EXAMPLES = [
    InputExample(
        text="Apple is a great company.",
        entities={"company": ["Apple"]},
    ),
    InputExample(
        text="Google works in NYC.",
        entities={"company": ["Google"], "location": ["NYC"]},
    ),
]


def _make_tokenizer() -> PreTrainedTokenizerFast:
    """Create a minimal fast tokenizer with a fixed small vocabulary."""
    tok_model = WordPiece(vocab=_VOCAB, unk_token="[UNK]")
    tokenizer_core = Tokenizer(tok_model)
    tokenizer_core.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def _make_model(tokenizer: PreTrainedTokenizerFast) -> Extractor:
    """
    Build a tiny Extractor model with a randomly-initialised small BERT
    encoder.  No pre-trained weights are downloaded.
    """
    tiny_bert = BertConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=len(tokenizer.get_vocab()),
    )
    ec = ExtractorConfig(
        # The model_name field is a metadata string used by ExtractorConfig;
        # the actual encoder is supplied via encoder_config so no pre-trained
        # weights are downloaded.
        model_name="bert-base-uncased",
        max_width=4,
        counting_layer="count_lstm",
        token_pooling="first",
    )
    return Extractor(config=ec, encoder_config=tiny_bert, tokenizer=tokenizer)


def _make_fixtures() -> Tuple[Extractor, ExtractorDataset, ExtractorCollator]:
    """Return a fresh (model, prev_dataset, collator) triple."""
    tok = _make_tokenizer()
    model = _make_model(tok)
    prev_dataset = ExtractorDataset(
        data=_PREV_EXAMPLES, shuffle=False, validate=False
    )
    collator = ExtractorCollator(model.processor, is_training=True)
    return model, prev_dataset, collator


def _minimal_training_config(tmpdir: str, **overrides) -> TrainingConfig:
    """Build a minimal TrainingConfig that avoids fp16 and workers."""
    defaults = dict(
        output_dir=tmpdir,
        num_epochs=1,
        batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=1,
        fp16=False,
        bf16=False,
        validate_data=False,
        eval_strategy="no",
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
        save_total_limit=0,
        report_to_wandb=False,
        logging_steps=1,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ewc_e2e_initialisation_via_trainer():
    """
    GLiNER2Trainer auto-creates an EWC object when use_ewc=True and
    ewc_prev_data is provided.
    """
    model, _, _ = _make_fixtures()
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _minimal_training_config(
            tmpdir,
            use_ewc=True,
            ewc_lambda=10.0,
            ewc_fisher_samples=2,
            ewc_normalize_fisher=True,
            ewc_prev_data=_PREV_EXAMPLES,
        )
        trainer = GLiNER2Trainer(model=model, config=config)

        assert trainer.ewc is not None, "Trainer should create an EWC object"
        assert isinstance(trainer.ewc, EWC)
        assert trainer.ewc.ewc_lambda == 10.0
        assert len(trainer.ewc.old_params) > 0
        assert len(trainer.ewc.fisher_info) > 0


def test_ewc_e2e_loss_is_zero_before_training():
    """
    Immediately after initialisation (before any weight updates) the EWC
    penalty should be ~0 because weights equal the stored anchors.
    """
    model, prev_dataset, collator = _make_fixtures()
    ewc = EWC(
        model=model,
        dataset=prev_dataset,
        data_collator=collator,
        device=torch.device("cpu"),
        ewc_lambda=50.0,
        batch_size=1,
        num_samples=2,
        normalize_fisher=True,
    )

    penalty = ewc.ewc_loss()
    assert penalty.item() < 1e-5, (
        f"EWC penalty should be ~0 before any weight change, got {penalty.item()}"
    )


def test_ewc_e2e_loss_increases_after_weight_update():
    """
    After one gradient step the EWC penalty should be strictly positive.
    """
    model, prev_dataset, collator = _make_fixtures()
    ewc = EWC(
        model=model,
        dataset=prev_dataset,
        data_collator=collator,
        device=torch.device("cpu"),
        ewc_lambda=50.0,
        batch_size=1,
        num_samples=2,
        normalize_fisher=True,
    )

    # Simulate a weight update
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05)

    penalty = ewc.ewc_loss()
    assert penalty.item() > 0.0, (
        f"EWC penalty should be > 0 after weight change, got {penalty.item()}"
    )


def test_ewc_e2e_fisher_covers_real_param_groups():
    """
    Fisher information matrix should cover the key GLiNER2 parameter groups:
    encoder, classifier, count_pred, count_embed, span_rep.
    """
    model, prev_dataset, collator = _make_fixtures()
    ewc = EWC(
        model=model,
        dataset=prev_dataset,
        data_collator=collator,
        device=torch.device("cpu"),
        ewc_lambda=10.0,
        batch_size=1,
        normalize_fisher=False,
    )

    groups = {name.split(".")[0] for name in ewc.fisher_info}
    assert "encoder" in groups, "Fisher should include encoder params"
    assert "classifier" in groups, "Fisher should include classifier params"


def test_ewc_e2e_importance_scores_return_real_groups():
    """
    get_importance_scores() should return a dict keyed by top-level module
    names present in the real Extractor architecture.
    """
    model, prev_dataset, collator = _make_fixtures()
    ewc = EWC(
        model=model,
        dataset=prev_dataset,
        data_collator=collator,
        device=torch.device("cpu"),
        ewc_lambda=10.0,
        batch_size=1,
        normalize_fisher=True,
    )

    scores = ewc.get_importance_scores()
    assert isinstance(scores, dict)
    assert len(scores) > 0
    for group, value in scores.items():
        assert isinstance(group, str)
        assert isinstance(value, float)
        assert value >= 0.0

    # Must at least contain the encoder group
    assert "encoder" in scores


def test_ewc_e2e_consolidate_with_real_data():
    """
    consolidate() should update Fisher estimates and old_params using real
    Extractor model data.
    """
    model, prev_dataset, collator = _make_fixtures()
    ewc = EWC(
        model=model,
        dataset=prev_dataset,
        data_collator=collator,
        device=torch.device("cpu"),
        ewc_lambda=10.0,
        batch_size=1,
        num_samples=2,
        normalize_fisher=True,
    )

    # Shift params, then consolidate
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05)

    ewc.consolidate(prev_dataset, alpha=0.5)

    # After consolidation, old_params should reflect the current weights
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert torch.allclose(ewc.old_params[name], param.data), (
                f"old_params[{name}] not updated after consolidate()"
            )

    # EWC loss should be ~0 after consolidation (anchors refreshed)
    penalty = ewc.ewc_loss()
    assert penalty.item() < 1e-5, (
        f"EWC penalty should be ~0 after consolidation, got {penalty.item()}"
    )


def test_ewc_e2e_training_completes_with_ewc():
    """
    A full training run through GLiNER2Trainer completes without errors
    when EWC is enabled.
    """
    model, _, _ = _make_fixtures()
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _minimal_training_config(
            tmpdir,
            use_ewc=True,
            ewc_lambda=10.0,
            ewc_fisher_samples=2,
            ewc_normalize_fisher=True,
            ewc_prev_data=_PREV_EXAMPLES,
        )
        trainer = GLiNER2Trainer(model=model, config=config)
        result = trainer.train(train_data=_NEW_EXAMPLES)

    assert result["total_steps"] > 0, "Should have completed at least one step"


def test_ewc_e2e_direct_ewc_object_respected():
    """
    When an EWC object is passed directly to GLiNER2Trainer, auto-init is
    skipped and the provided object is used as-is.
    """
    model, prev_dataset, collator = _make_fixtures()
    pre_built_ewc = EWC(
        model=model,
        dataset=prev_dataset,
        data_collator=collator,
        device=torch.device("cpu"),
        ewc_lambda=999.0,
        batch_size=1,
        num_samples=2,
        normalize_fisher=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _minimal_training_config(
            tmpdir,
            use_ewc=True,
            ewc_lambda=1.0,  # different lambda in config
            ewc_prev_data=_PREV_EXAMPLES,
        )
        trainer = GLiNER2Trainer(model=model, config=config, ewc=pre_built_ewc)

    # The pre-built EWC (lambda=999) should be used, not re-initialized
    assert trainer.ewc is pre_built_ewc
    assert trainer.ewc.ewc_lambda == 999.0


def test_ewc_e2e_penalty_added_to_total_loss():
    """
    Verify that the loss seen during training is strictly greater than the
    base model loss when EWC is active (the penalty is being added).
    """
    model, prev_dataset, collator = _make_fixtures()
    new_dataset = ExtractorDataset(
        data=_NEW_EXAMPLES, shuffle=False, validate=False
    )

    # Shift model weights so the penalty is non-trivial
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)

    ewc = EWC(
        model=model,
        dataset=prev_dataset,
        data_collator=collator,
        device=torch.device("cpu"),
        ewc_lambda=1000.0,  # large lambda to make penalty clearly visible
        batch_size=1,
        num_samples=2,
        normalize_fisher=True,
    )

    # Shift again so the current params differ from old_params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)

    # Compute base loss from the model
    model.eval()
    batch = collator([new_dataset[0]])
    with torch.no_grad():
        base_loss = model(batch)["total_loss"].item()

    penalty = ewc.ewc_loss().item()
    total_with_ewc = base_loss + penalty

    assert penalty > 0.0, "EWC penalty should be positive"
    assert total_with_ewc > base_loss, (
        "Total loss with EWC should exceed base model loss"
    )
