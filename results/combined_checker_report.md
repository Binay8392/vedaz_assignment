# Vedaz Dataset Checker Report

## Summary

- Total conversations: 25
- Valid conversations: 25
- Malformed conversations: 0
- Safety counts: Safe=13, Warning=12, Unsafe=0
- Exact duplicate groups: 0
- Near-duplicate pairs: 0
- Train split: 22
- Test split: 3

## Dataset Statistics

- Average word count: 240.1
- Average token count: 323.6
- Average conversation length: 4.8 messages

## Safety Findings

- Record 2 (vedaz_astrologer_finetune.jsonl): Warning. Sensitive medical or mental-health topic; professional care may be needed.; Sensitive legal topic; qualified legal advice may be needed.
- Record 7 (vedaz_astrologer_finetune.jsonl): Warning. High-stakes life outcome; avoid guarantees and keep guidance limited.
- Record 15 (generated_chats.jsonl): Warning. Sensitive medical or mental-health topic; professional care may be needed.; Sensitive legal topic; qualified legal advice may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- Record 16 (generated_chats.jsonl): Warning. Sensitive medical or mental-health topic; professional care may be needed.; Sensitive legal topic; qualified legal advice may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- Record 17 (generated_chats.jsonl): Warning. Sensitive legal topic; qualified legal advice may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- Record 18 (generated_chats.jsonl): Warning. Sensitive legal topic; qualified legal advice may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- Record 19 (generated_chats.jsonl): Warning. Sensitive legal topic; qualified legal advice may be needed.; Sensitive financial topic; qualified financial advice may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- Record 20 (generated_chats.jsonl): Warning. Sensitive medical or mental-health topic; professional care may be needed.; Sensitive legal topic; qualified legal advice may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- Record 21 (generated_chats.jsonl): Warning. Sensitive medical or mental-health topic; professional care may be needed.; Sensitive legal topic; qualified legal advice may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- Record 22 (generated_chats.jsonl): Warning. Sensitive legal topic; qualified legal advice may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- Record 23 (generated_chats.jsonl): Warning. Sensitive legal topic; qualified legal advice may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.
- Record 24 (generated_chats.jsonl): Warning. Sensitive medical or mental-health topic; professional care may be needed.; Sensitive legal topic; qualified legal advice may be needed.; High-stakes life outcome; avoid guarantees and keep guidance limited.

## Exact Duplicates

- No exact duplicates detected.

## Near Duplicates

- No near duplicates above threshold detected.

## Method Notes

- Structure validation requires a system message first, then alternating user and assistant messages.
- Unsafe detection is rule-based by default, with optional LLM judging through `ENABLE_LLM_JUDGE=true`.
- Near-duplicate detection uses sentence embeddings when available; otherwise it falls back to TF-IDF cosine similarity.
- Token counts use `tiktoken` when installed; otherwise they use a conservative word-based approximation.
