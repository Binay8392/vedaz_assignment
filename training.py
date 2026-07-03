"""Shared LoRA training implementation for Qwen2.5 and Qwen3."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from dataset_pipeline import validate_dataset_files
from utils import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
SUPPORTED_MODEL_FAMILIES = {"qwen2.5", "qwen3"}
SUPPORTED_BACKENDS = {"transformers", "unsloth"}


@dataclass(frozen=True)
class TrainingConfig:
    """Serializable training configuration."""

    model_family: str
    model_name_or_path: str
    backend: str
    train_file: str
    validation_file: str
    output_dir: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 0.0002
    num_train_epochs: float = 3.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    eval_steps: int = 5
    save_steps: int = 5
    save_total_limit: int = 2
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    use_rslora: bool = False
    optimizer: str = "paged_adamw_8bit"
    seed: int = 42
    dataloader_num_workers: int = 0
    report_to: str = "none"
    run_name: str = "vedaz-lora"
    resume_from_checkpoint: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = "sdpa"

    @property
    def train_path(self) -> Path:
        """Return the absolute training dataset path."""
        return _project_path(self.train_file)

    @property
    def validation_path(self) -> Path:
        """Return the absolute validation dataset path."""
        return _project_path(self.validation_file)

    @property
    def output_path(self) -> Path:
        """Return the absolute output path."""
        return _project_path(self.output_dir)


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_training_config(path: Path) -> TrainingConfig:
    """Load a strict YAML training configuration."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read training configs") from exc

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    allowed = {item.name for item in fields(TrainingConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unsupported training config keys: {', '.join(unknown)}")
    if isinstance(payload.get("target_modules"), list):
        payload["target_modules"] = tuple(payload["target_modules"])
    try:
        config = TrainingConfig(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid training config: {exc}") from exc
    validate_training_config(config)
    return config


def apply_overrides(
    config: TrainingConfig,
    overrides: dict[str, Any],
) -> TrainingConfig:
    """Apply non-null command-line overrides to a config."""
    clean = {key: value for key, value in overrides.items() if value is not None}
    updated = replace(config, **clean)
    validate_training_config(updated)
    return updated


def validate_training_config(config: TrainingConfig) -> None:
    """Reject unsafe or internally inconsistent training settings."""
    if config.model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"model_family must be one of {', '.join(sorted(SUPPORTED_MODEL_FAMILIES))}"
        )
    if config.backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"backend must be one of {', '.join(sorted(SUPPORTED_BACKENDS))}"
        )
    if config.fp16 and config.bf16:
        raise ValueError("fp16 and bf16 are mutually exclusive")
    if config.max_seq_length < 128:
        raise ValueError("max_seq_length must be at least 128")
    if config.per_device_train_batch_size < 1:
        raise ValueError("per_device_train_batch_size must be positive")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.num_train_epochs <= 0 and config.max_steps <= 0:
        raise ValueError("num_train_epochs or max_steps must be positive")
    if config.lora_rank < 1 or config.lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= config.lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")
    if not config.target_modules:
        raise ValueError("target_modules cannot be empty")
    if config.report_to not in {"none", "tensorboard", "wandb"}:
        raise ValueError("report_to must be none, tensorboard, or wandb")


def validate_training_inputs(config: TrainingConfig) -> None:
    """Validate both dataset splits and their mutual uniqueness."""
    summary = validate_dataset_files(
        [config.train_path, config.validation_path],
        reject_duplicates=True,
    )
    if not summary.passed:
        errors = [
            f"{issue.source}:{issue.record or '-'} {issue.message}"
            for issue in summary.issues
            if issue.severity == "error"
        ]
        raise ValueError("training data validation failed: " + "; ".join(errors))
    if summary.total_records < 2:
        raise ValueError("training requires at least two conversations")


def config_as_dict(config: TrainingConfig) -> dict[str, Any]:
    """Convert a config into JSON-safe values."""
    payload = asdict(config)
    payload["target_modules"] = list(config.target_modules)
    return payload


def _resolve_torch_dtype(config: TrainingConfig) -> Any:
    import torch

    if config.bf16:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bf16 was requested but the active CUDA device does not "
                "support it; use fp16 or full precision"
            )
        return torch.bfloat16
    if config.fp16:
        if not torch.cuda.is_available():
            raise RuntimeError("fp16 training requires a CUDA device")
        return torch.float16
    return torch.float32


def _load_transformers_model(
    config: TrainingConfig,
    torch_dtype: Any,
) -> tuple[Any, Any]:
    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if config.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit training requires CUDA and bitsandbytes")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=list(config.target_modules),
        task_type="CAUSAL_LM",
        use_rslora=config.use_rslora,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def _load_unsloth_model(
    config: TrainingConfig,
    torch_dtype: Any,
) -> tuple[Any, Any]:
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth backend requested but unsloth is not installed. "
            "Install requirements-unsloth.txt on Linux or WSL."
        ) from exc

    dtype = torch_dtype if config.fp16 or config.bf16 else None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name_or_path,
        max_seq_length=config.max_seq_length,
        dtype=dtype,
        load_in_4bit=config.load_in_4bit,
        trust_remote_code=config.trust_remote_code,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=list(config.target_modules),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=(
            "unsloth" if config.gradient_checkpointing else False
        ),
        random_state=config.seed,
        use_rslora=config.use_rslora,
    )
    return model, tokenizer


def _tokenize_datasets(config: TrainingConfig, tokenizer: Any) -> tuple[Any, Any]:
    from datasets import load_dataset

    data = load_dataset(
        "json",
        data_files={
            "train": str(config.train_path),
            "validation": str(config.validation_path),
        },
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer defines neither a pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_record(record: dict[str, Any]) -> dict[str, Any]:
        template_kwargs: dict[str, Any] = {}
        if config.model_family == "qwen3":
            template_kwargs["enable_thinking"] = False
        rendered = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
            **template_kwargs,
        )
        encoded = tokenizer(
            rendered,
            truncation=True,
            max_length=config.max_seq_length,
            add_special_tokens=False,
        )
        encoded["length"] = len(encoded["input_ids"])
        return encoded

    train_columns = data["train"].column_names
    validation_columns = data["validation"].column_names
    tokenized_train = data["train"].map(
        tokenize_record,
        remove_columns=train_columns,
        desc="Tokenizing train split",
    )
    tokenized_validation = data["validation"].map(
        tokenize_record,
        remove_columns=validation_columns,
        desc="Tokenizing validation split",
    )
    return tokenized_train, tokenized_validation


class CausalLMDataCollator:
    """Right-pad causal-LM examples and mask padding labels."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        clean_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            clean_features,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels.to(dtype=torch.long)
        return batch


def _training_arguments(config: TrainingConfig) -> Any:
    from transformers import TrainingArguments

    optimizer = config.optimizer
    if not config.load_in_4bit and optimizer.startswith("paged_"):
        optimizer = "adamw_torch"
    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_path),
        "run_name": config.run_name,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": config.lr_scheduler_type,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "save_total_limit": config.save_total_limit,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "optim": optimizer,
        "seed": config.seed,
        "data_seed": config.seed,
        "dataloader_num_workers": config.dataloader_num_workers,
        "report_to": [] if config.report_to == "none" else [config.report_to],
        "remove_unused_columns": False,
        "load_best_model_at_end": False,
        "save_safetensors": True,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    strategy_name = (
        "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    )
    kwargs[strategy_name] = "steps"
    if "gradient_checkpointing_kwargs" in parameters:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**kwargs)


def train_lora(config: TrainingConfig) -> Path:
    """Run LoRA SFT and save a portable PEFT adapter."""
    try:
        from transformers import Trainer, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing; install requirements.txt"
        ) from exc

    validate_training_inputs(config)
    set_seed(config.seed)
    torch_dtype = _resolve_torch_dtype(config)
    config.output_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Loading %s with the %s backend",
        config.model_name_or_path,
        config.backend,
    )
    if config.backend == "unsloth":
        model, tokenizer = _load_unsloth_model(config, torch_dtype)
    else:
        model, tokenizer = _load_transformers_model(config, torch_dtype)

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_dataset, validation_dataset = _tokenize_datasets(config, tokenizer)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": _training_arguments(config),
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": CausalLMDataCollator(tokenizer),
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(
        **trainer_kwargs,
    )

    resume: bool | str | None = config.resume_from_checkpoint
    if resume == "latest":
        resume = True
    elif isinstance(resume, str):
        checkpoint = _project_path(resume)
        if not checkpoint.is_dir():
            raise ValueError(f"resume checkpoint does not exist: {checkpoint}")
        resume = str(checkpoint)

    LOGGER.info("Starting supervised fine-tuning")
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    adapter_path = config.output_path / "adapter"
    model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(adapter_path)
    (config.output_path / "training_config.json").write_text(
        json.dumps(config_as_dict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved PEFT adapter to %s", adapter_path)
    return adapter_path
