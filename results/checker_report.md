# Vedaz Dataset Checker Report

## Summary

- Total conversations: 15
- Valid conversations: 15
- Malformed conversations: 0
- Safety counts: Safe=14, Warning=1, Unsafe=0
- Exact duplicate groups: 0
- Near-duplicate pairs: 0
- Train split: 13
- Test split: 2

## Dataset Statistics

- Average word count: 231.9
- Average token count: 312.6
- Average conversation length: 3.3 messages

## Safety Findings

- Record 7 (vedaz_astrologer_finetune.jsonl): Warning. High-stakes life outcome; avoid guarantees and keep guidance limited.

## Exact Duplicates

- No exact duplicates detected.

## Near Duplicates

- No near duplicates above threshold detected.

## Method Notes

- Structure validation requires a system message first, then alternating user and assistant messages.
- Unsafe detection is rule-based by default, with optional LLM judging through `ENABLE_LLM_JUDGE=true`.
- Near-duplicate detection uses sentence embeddings when available; otherwise it falls back to TF-IDF cosine similarity.
- Token counts use `tiktoken` when installed; otherwise they use a conservative word-based approximation.
