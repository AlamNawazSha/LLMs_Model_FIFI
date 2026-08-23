# Academic Title Retrieval Pipeline

This project finds the **original academic title** when given a **rewritten title** (Catchy, Accessible, etc.).

It uses:
- **Dense search** (fine-tuned AI model)
- **BM25 search** (keyword matching)
- **Reranker** (picks the best final answer)

---

## 📁 Files

| File | What it is |
|------|------------|
| `finetuned_retrieval_pipeline_v2.py` | Main script — run this |
| `train.tsv` | Training data (you provide) |
| `val.tsv` | Validation data (you provide) |
| `subtask1_submission.tsv` | Output: top 10 guesses per title |
| `subtask2_submission.tsv` | Output: best single guess per title |

---

## 📊 Data Format

Your `.tsv` files need these columns:

| Column | Meaning |
|--------|---------|
| `id` | Row ID |
| `category` | Rewrite style (Catchy, Accessible, etc.) |
| `generated_title` | The rewritten title |
| `original_title` | The real academic title |

---

## ⚙️ Install

```bash
pip install -q sentence-transformers rank_bm25 faiss-gpu transformers accelerate bert-score evaluate
```

Needs a GPU. Works on CPU too, just slower.

---

## ▶️ Run

```bash
python finetuned_retrieval_pipeline_v2.py
```

**What happens:**
1. Loads and cleans your data
2. Trains the AI model on your examples
3. Trains the reranker
4. Searches for matches
5. Saves results + prints scores

First run trains everything (takes longer). After that, it reuses saved models automatically — much faster.

---

## 🎯 Why It Works Well

Catchy and Accessible titles are the hardest to match because they use very different words from the original. This pipeline fixes that by:

- Teaching the model with **hard examples** (similar-but-wrong titles), not just random ones
- Training the model **extra** on Catchy/Accessible examples
- Combining both search methods **fairly** instead of favoring one

---

## 📈 Scores You'll See

| Metric | What it Measures |
|--------|-------------------|
| MRR@10 | How high the correct answer ranks (top 10 list) |
| Token F1 | Word overlap with correct answer |
| BERTScore | Meaning similarity with correct answer |

Per-category scores are also printed, so you can check if Catchy/Accessible improved.

---

## 🛠️ Common Issue

**Error:** `'DataParallel' object has no attribute 'preprocess'`

**Cause:** Multiple GPUs detected, causing a conflict.

**Fix:** Already handled in the script (forces single GPU use).

---

## 🔧 Settings You Can Tweak

Found near the top of the script:

```python
HARD_CATEGORIES = {"Catchy", "Accessible"}   # categories to focus on
HARD_CATEGORY_OVERSAMPLE_FACTOR = 2          # how much extra training for them
N_HARD_NEGATIVES = 4                         # tricky wrong answers per example
TOP_K_PER_RETRIEVER = 50                     # candidates to consider before final pick
```
