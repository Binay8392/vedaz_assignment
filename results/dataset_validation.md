# Dataset Validation Report

- Status: PASS
- Files: 2
- Conversations: 58
- Schema-valid conversations: 58
- Safety: 31 safe, 27 warning, 0 unsafe
- Tokens per conversation: min 101, mean 308.52, max 737
- Blocking errors: 0
- Review warnings: 27

## Findings

- [WARNING] `sensitive_topic` at `data/train.jsonl:1`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:2`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:4`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:6`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:7`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:8`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:16`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:19`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:20`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:23`: Sensitive medical or mental-health topic; professional care may be needed.
- [WARNING] `sensitive_topic` at `data/train.jsonl:24`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:25`: Sensitive financial topic; qualified financial advice may be needed.
- [WARNING] `sensitive_topic` at `data/train.jsonl:26`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:28`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:32`: Sensitive legal topic; qualified legal advice may be needed.; Sensitive financial topic; qualified financial advice may be needed.
- [WARNING] `sensitive_topic` at `data/train.jsonl:33`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:40`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:41`: Sensitive financial topic; qualified financial advice may be needed.
- [WARNING] `sensitive_topic` at `data/train.jsonl:42`: Sensitive medical or mental-health topic; professional care may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/train.jsonl:46`: Sensitive medical or mental-health topic; professional care may be needed.
- [WARNING] `sensitive_topic` at `data/validation.jsonl:1`: Sensitive medical or mental-health topic; professional care may be needed.
- [WARNING] `sensitive_topic` at `data/validation.jsonl:4`: Sensitive medical or mental-health topic; professional care may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/validation.jsonl:5`: Sensitive medical or mental-health topic; professional care may be needed.
- [WARNING] `sensitive_topic` at `data/validation.jsonl:6`: Sensitive medical or mental-health topic; professional care may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/validation.jsonl:8`: High-stakes life outcome; avoid guarantees and keep guidance limited.
- [WARNING] `sensitive_topic` at `data/validation.jsonl:10`: Sensitive legal topic; qualified legal advice may be needed.
- [WARNING] `sensitive_topic` at `data/validation.jsonl:12`: Sensitive medical or mental-health topic; professional care may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
