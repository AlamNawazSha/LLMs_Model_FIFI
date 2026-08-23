# FIFI 2026 — Scientific Title Retrieval & Rewriting

Solution for the **FIFI@FIRE 2026** shared task: trace an LLM-rewritten scientific title back to its original.
Task site: https://fifi-2026.github.io/FIFI-2026

## Task
LLMs rewrite titles into 3 styles: **Technical**, **Accessible**, **Catchy**.
- **Subtask 1 (Find It):** rewritten title → ranked top-10 original titles. Metric: **MRR@10**
- **Subtask 2 (Fix It):** rewritten title + style label → reconstructed original title. Metric: **F1 + BERTScore**

## Pipeline
Bi-encoder and reranker are fine-tuned (not zero-shot) using **hard negatives**, since Catchy/Accessible titles drift furthest from the original and are hardest to match. Candidates are oversampled for those categories, fused by rank (not raw score), and the style label is fed into Subtask 2 as allowed.

## Files
| File | Purpose |
|---|---|
| `finetuned_retrieval_pipeline_v2.py` | Main script |
| `train.tsv` / `val.tsv` | Data (columns: `id`, `category`, `generated_title`, `original_title`) |
| `subtask1_submission.tsv` / `subtask2_submission.tsv` | Outputs |

## Setup & Run
```bash
pip install -q sentence-transformers rank_bm25 faiss-gpu transformers accelerate bert-score evaluate
python finetuned_retrieval_pipeline_v2.py
```
GPU recommended. First run trains + caches everything (bi-encoder, reranker, hard negatives); later runs reuse the cache. Prints MRR@10 overall and **per category** so you can check Catchy/Accessible improvement.

## Key Settings
```python
HARD_CATEGORIES = {"Catchy", "Accessible"}
HARD_CATEGORY_OVERSAMPLE_FACTOR = 2
N_HARD_NEGATIVES = 4
TOP_K_PER_RETRIEVER = 50
RRF_K = 60
```

## Troubleshooting
`AttributeError: 'DataParallel' object has no attribute 'preprocess'` → caused by multiple visible GPUs; fixed by pinning `CUDA_VISIBLE_DEVICES=0` at the top of the script.

## Credit
Task & dataset by the FIFI Shared Task, FIRE 2026.
