from __future__ import annotations

from dataclasses import replace

import pytest

from training import (
    load_training_config,
    validate_training_config,
    validate_training_inputs,
)
from utils import PROJECT_ROOT


def test_both_model_configs_and_data_are_valid() -> None:
    for filename, family in (
        ("qwen2.5-lora.yaml", "qwen2.5"),
        ("qwen3-lora.yaml", "qwen3"),
    ):
        config = load_training_config(PROJECT_ROOT / "configs" / filename)
        assert config.model_family == family
        validate_training_inputs(config)


def test_precision_modes_are_mutually_exclusive() -> None:
    config = load_training_config(PROJECT_ROOT / "configs" / "qwen2.5-lora.yaml")

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_training_config(replace(config, fp16=True, bf16=True))
