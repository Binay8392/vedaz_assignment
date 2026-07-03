#Binay 
# Vedaz Responsible Astrology LLM

An end-to-end data, fine-tuning, evaluation, and deployment pipeline for a
multilingual astrology assistant. The repository was built for the Vedaz AI
Engineer internship assignment and is designed around one constraint: the model
must remain useful without presenting astrology as certainty or replacing
medical, legal, financial, or emergency support.

The project supports LoRA and QLoRA fine-tuning of Qwen2.5 and Qwen3 with either
Hugging Face Transformers + PEFT or Unsloth. It also includes deterministic data
curation, local and API inference, a behavioral evaluation suite, adapter
merging, and production serving through vLLM.

## Project Overview

The repository covers the complete model lifecycle:

1. Normalize OpenAI-style conversation data.
2. Validate schema, role order, encoding, safety, and uniqueness.
3. Remove duplicate and ungrounded examples before splitting.
4. Fine-tune Qwen2.5 or Qwen3 with configurable LoRA/QLoRA.
5. Save portable PEFT adapters and resume from checkpoints.
6. Run local chat inference or merge adapters into the base model.
7. Evaluate behavior locally or through an OpenAI-compatible endpoint.
8. Serve the adapter on an NVIDIA VPS with vLLM, systemd, Nginx, and HTTPS.

No API key is required for dataset preparation, validation, tests, or the
offline evaluation smoke test.

## Architecture

```text
Source JSONL
   |
   v
schema + role order + UTF-8 + safety + duplicate checks
   |
   v
ungrounded chart-claim filter
   |
   +------> data/train.jsonl
   `------> data/validation.jsonl
                |
                v
        Qwen chat template
                |
       +--------+---------+
       |                  |
Transformers + PEFT    Unsloth + PEFT
       |                  |
       +--------+---------+
                |
          LoRA adapter
          /     |      \
 local inference  evaluation  vLLM serving
```

The data and safety rules are shared by preparation, synthetic generation, and
evaluation. Heavy GPU libraries are imported only when a model is loaded, so
reviewers can run validation and CLI dry-runs on a CPU machine.

## Dataset

Every training record uses the conversational shape accepted by Hugging Face:

```json
{
  "messages": [
    {"role": "system", "content": "Behavior and safety policy"},
    {"role": "user", "content": "User question"},
    {"role": "assistant", "content": "Responsible answer"}
  ]
}
```

The first message must be `system`; user and assistant turns must alternate; the
last message must be `assistant`; and every content field must be a non-empty
string.

The committed sources contain 75 conversations before curation. The additional
document supplied with the assignment contained 55 pasted root objects; it was
normalized to JSONL and reduced to 50 unique records. Preparation excludes 17
examples that assert chart placements, dashas, or transits without calculation
provenance. The final deterministic split is:

| Split | Conversations |
|---|---:|
| Train | 46 |
| Validation | 12 |
| Total eligible | 58 |

Sensitive-topic warnings are retained when the assistant sets an appropriate
boundary and redirects the user. Unsafe records, malformed records, exact
duplicates, and ungrounded chart assertions are excluded. The manifest records
source and output SHA-256 hashes, split seed, counts, and curation settings in
`data/dataset_manifest.json`.

Rebuild and validate the data:

```bash
python prepare_dataset.py
python validate_dataset.py
```

Validation exits non-zero for a missing file, invalid JSON, schema or role
errors, encoding artifacts, exact duplicates across splits, or unsafe assistant
content. It writes machine-readable and Markdown reports under `results/`.

`generator.py` can create reviewed offline scenarios or use OpenAI/Together:

```bash
python generator.py --provider offline --count 10
python generator.py \
  --provider openai \
  --topic "Career change" \
  --language Hinglish \
  --difficulty hard \
  --count 1
```

Generated responses are parsed, schema-checked, and safety-filtered before they
are saved.

## Training Pipeline

`train.py` reads a strict YAML config, validates both splits together to prevent
leakage, renders each conversation through the model's own chat template,
tokenizes to the configured sequence length, and trains a causal language model
with PEFT adapters.

Two backends share the same pipeline:

- `transformers`: loads the base model with Transformers, prepares 4-bit models
  with PEFT, and attaches LoRA directly.
- `unsloth`: uses `FastLanguageModel` for optimized loading and gradient
  checkpointing, then trains the PEFT model with the Transformers trainer.

Both backends support fp16, bf16, full precision, NF4 4-bit loading, gradient
checkpointing, gradient accumulation, checkpoint rotation, evaluation during
training, resume-from-checkpoint, and safetensors adapter output.

Run a configuration and data check without downloading a model:

```bash
python train.py --config configs/qwen2.5-lora.yaml --dry-run
python train.py --config configs/qwen3-lora.yaml --dry-run
```

Start training:

```bash
# Qwen2.5 with Transformers + PEFT
python train.py --config configs/qwen2.5-lora.yaml

# Qwen3 with Unsloth + PEFT
python train.py --config configs/qwen3-lora.yaml
```

Override common settings from the CLI:

```bash
python train.py \
  --config configs/qwen2.5-lora.yaml \
  --fp16 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 0.0001 \
  --epochs 4
```

Resume the latest checkpoint in the output directory, or pass a specific
checkpoint path:

```bash
python train.py \
  --config configs/qwen2.5-lora.yaml \
  --resume-from-checkpoint
```

Adapters and tokenizer files are saved to `<output_dir>/adapter`. Trainer state,
metrics, checkpoints, and the resolved JSON configuration remain in the parent
output directory.

## Model

The default configurations target:

- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen3-4B`

Qwen2.5 is the smaller default for development and single-GPU deployment.
Qwen3 uses non-thinking mode during dataset rendering and local inference
because this assistant needs concise user-facing guidance rather than exposed
reasoning traces.

The PEFT target modules cover attention and MLP projections:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`. A LoRA adapter is tied to its exact base model family and should
not be moved between Qwen2.5 and Qwen3.

## Hyperparameters

Both committed configs use the following baseline:

| Parameter | Value |
|---|---:|
| Sequence length | 2,048 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Train batch size / device | 2 |
| Gradient accumulation | 8 |
| Effective batch size / device | 16 |
| Scheduler | Cosine |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Quantization | NF4 4-bit |
| Precision | bf16 |
| Gradient checkpointing | Enabled |

These are defensible starting values, not universal optima. On pre-Ampere GPUs,
use `--fp16`; for a CPU-only functional check, disable 4-bit loading and both
half-precision modes. Dataset size is small enough that validation loss and
qualitative behavior should be reviewed for overfitting after every run.

## Inference

Load an adapter without manually repeating the base model name:

```bash
python inference.py \
  --adapter artifacts/qwen2.5-3b-vedaz-lora/adapter \
  --load-in-4bit \
  --prompt "Can astrology guarantee that I will get a promotion?"
```

Omit `--prompt` for an interactive conversation. A merged model can be loaded
with `--model` and no adapter.

Merge in bf16 for a standalone deployment artifact:

```bash
python merge_adapter.py \
  --adapter artifacts/qwen2.5-3b-vedaz-lora/adapter \
  --output artifacts/qwen2.5-3b-vedaz-merged \
  --dtype bf16
```

Merging deliberately does not use a 4-bit base model. It performs PEFT's safe
merge check and saves sharded safetensors plus tokenizer files.

## Evaluation

The 16-case suite covers:

- instruction following;
- safety;
- English, Hindi, and Hinglish consistency;
- refusal quality;
- ungrounded or fabricated chart claims;
- overall response quality.

It includes adversarial self-harm, death, urgent health, legal, financial,
pregnancy, gemstone pressure, visa, relationship, and exact-prediction prompts.

Run the deterministic reference policy to verify the evaluation machinery:

```bash
python evaluator.py --backend offline --judge heuristic
```

Evaluate a trained adapter:

```bash
python evaluator.py \
  --backend local \
  --adapter artifacts/qwen2.5-3b-vedaz-lora/adapter \
  --load-in-4bit
```

Evaluate the vLLM adapter:

```bash
python evaluator.py \
  --backend endpoint \
  --base-url http://127.0.0.1:8000/v1 \
  --model vedaz
```

Set `VLLM_API_KEY` for endpoint evaluation. A separate OpenAI-compatible judge
can be enabled with `--judge llm --judge-model <model-id>` and
`OPENAI_API_KEY`.

The committed `results/evaluation_report.md` is explicitly labeled as an
offline reference-policy smoke test. It scores 5.00/5 for safety, 5.00/5 for
language consistency, and 0% on the heuristic hallucination flag. It is not
presented as evidence for an untrained adapter; the report must be rerun after
fine-tuning and before deployment.

## Deployment

The Docker path serves a LoRA adapter through vLLM:

```bash
cp .env.example .env
# Set VLLM_API_KEY and ADAPTER_PATH in .env.
docker compose build
docker compose up -d
```

The API binds to `127.0.0.1:8000`, exposes the adapter as model `vedaz`, mounts
the adapter read-only, persists the Hugging Face cache, reserves the NVIDIA GPU,
and includes a health check.

The full VPS runbook is in
[`deployment/vllm_guide.md`](deployment/vllm_guide.md). It covers Ubuntu and
NVIDIA setup, CUDA compatibility, Python virtual environments, vLLM
installation, model downloads, direct LoRA serving, merged serving, the
OpenAI-compatible endpoint, systemd hardening, Nginx, HTTPS, firewall rules,
monitoring, GPU memory tuning, and failure diagnosis.

## Repository Structure

```text
.
|-- configs/
|   |-- qwen2.5-lora.yaml
|   `-- qwen3-lora.yaml
|-- data/
|   |-- vedaz_astrologer_finetune.jsonl
|   |-- generated_chats.jsonl
|   |-- additional_conversations.jsonl
|   |-- train.jsonl
|   |-- validation.jsonl
|   `-- dataset_manifest.json
|-- deployment/
|   `-- vllm_guide.md
|-- evaluation/
|   `-- test_cases.jsonl
|-- results/
|   |-- dataset_validation.json
|   |-- dataset_validation.md
|   |-- evaluation_results.jsonl
|   `-- evaluation_report.md
|-- scripts/
|   |-- train.sh / train.ps1
|   |-- evaluate.sh / evaluate.ps1
|   |-- merge_adapter.sh / merge_adapter.ps1
|   `-- launch_vllm.sh
|-- tests/
|-- checker.py
|-- dataset_pipeline.py
|-- evaluator.py
|-- generator.py
|-- inference.py
|-- merge_adapter.py
|-- prepare_dataset.py
|-- train.py
|-- training.py
|-- validate_dataset.py
|-- Dockerfile
|-- docker-compose.yml
|-- environment.yml
|-- requirements.txt
`-- requirements-unsloth.txt
```

## How to Run

Python 3.11 is the tested version.

```bash
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Linux or WSL with Unsloth:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-unsloth.txt
```

Conda is also supported:

```bash
conda env create -f environment.yml
conda activate vedaz-lora
```

Run the CPU-safe verification path:

```bash
python prepare_dataset.py
python validate_dataset.py
python train.py --config configs/qwen2.5-lora.yaml --dry-run
python train.py --config configs/qwen3-lora.yaml --dry-run
python evaluator.py --backend offline
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Environment variables are documented in `.env.example`. Secrets and model
artifacts are excluded from git.

## Limitations

- The curated corpus is only 58 conversations; it is suitable for a hiring
  assignment and pipeline validation, not broad production coverage.
- Regex safety checks are interpretable but can miss indirect phrasing and
  multilingual edge cases.
- Sensitive-topic warnings still need human review even when they pass the
  blocking rules.
- The training objective covers the full rendered conversation, not only
  assistant spans.
- No astrology calculation engine is included. Examples that assert computed
  placements without provenance are filtered, and model outputs are checked for
  the same failure mode.
- Heuristic evaluation is reproducible but not a substitute for blinded human
  ratings or a labeled safety benchmark.
- This project evaluates responsible communication, not the factual validity of
  astrology.

## Future Work

- Add a multilingual, labeled safety classifier and measure per-category
  precision and recall.
- Integrate a deterministic ephemeris service with provenance if chart-specific
  claims become a product requirement.
- Expand the evaluation set and add blinded ratings from Hindi and Hinglish
  speakers.
- Add assistant-token loss masking after pinning a chat template that exposes a
  reliable assistant mask.
- Track experiments and dataset/model lineage in MLflow or Weights & Biases.
- Add GPU integration tests and automated endpoint evaluation to CI.
- Measure latency, throughput, VRAM use, and quality across bf16, AWQ, GPTQ, and
  FP8 deployments.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Built with Qwen, Hugging Face Transformers, PEFT, Unsloth, PyTorch, and vLLM.
The safety design is informed by the practical need to separate reflective
astrology from medical, legal, financial, and crisis decisions.
