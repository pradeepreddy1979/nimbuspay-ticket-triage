# NimbusPay Ticket Triage – Live Defense & Technical Reference Guide

This document is your complete preparation sheet for the **30-minute technical interview / live defense** with the VIMA3YA engineering team.

---

## 📁 1. Local Files in this Folder

| File | Purpose |
|---|---|
| [`README.md`](README.md) | Summary documentation with base model ID, 4-bit status, and inference code. |
| [`report.md`](report.md) | The 3-page technical report (audit, experiments, error analysis, 1-week roadmap). |
| [`notebook.ipynb`](notebook.ipynb) | Complete Jupyter Notebook with pre-computed cell outputs, reproducible on Colab T4. |
| [`cleaning_log.csv`](cleaning_log.csv) | The 300 logged cleaning actions (`id,action,reason`): 125 dropped, 175 fixed. |
| [`predictions_test.jsonl`](predictions_test.jsonl) | The 400 test predictions on `data/test_inputs.jsonl`. |
| [`adapter/`](adapter/) | Standard PEFT LoRA adapter (`adapter_config.json` + `adapter_model.safetensors`, 57.16 MB). |
| [`data/train_clean.jsonl`](data/train_clean.jsonl) | 1,320 cleaned, deduplicated, verified training tickets. |
| [`score.py`](score.py) | Official evaluation scorer script. |
| [`demo_inference.py`](demo_inference.py) | Interactive CLI tool to test any ticket live. |
| [`audit_and_clean.py`](audit_and_clean.py) | Standalone script that reproduces `cleaning_log.csv` and `train_clean.jsonl`. |
| [`run_evaluation.py`](run_evaluation.py) | Evaluates the model on `data/dev.jsonl` using `score.py`. |
| [`generate_test_predictions.py`](generate_test_predictions.py) | Engine that generates test predictions. |

---

## ⚡ 2. Quick Local Commands You Can Run Live

During the interview, you can open a terminal in this folder and run:

### A. Run Triage on Sample Tickets Live
```bash
python demo_inference.py
```

### B. Run Triage on a Custom Ticket Given by the Interviewer
```bash
python demo_inference.py --ticket "Received: 2026-08-14\nSource: email\n\nI will go to consumer court if my 60000 refund is not processed."
```

### C. Run the Evaluation Scorer on Dev Data
```bash
python run_evaluation.py
```

### D. Reproduce Data Cleaning & Audit Log
```bash
python audit_and_clean.py
```

---

## 🧠 3. Core Numbers to Memorize

1. **Base Model:** `Qwen/Qwen2.5-3B-Instruct`
2. **Quantization:** 4-bit NormalFloat (NF4) with double quantization (QLoRA).
3. **Trainable Parameters:** **29,933,568** (~29.93M parameters, 0.97% of base model, strictly under the 50M limit).
4. **Adapter File Size:** **57.16 MB** (safetensors format, fits on GitHub without Git LFS).
5. **Peak VRAM on Colab T4:** **4.8 GB** (leaves ~10 GB free on free tier T4 GPU).
6. **Training Time on T4:** **~26 minutes** for 3 epochs.
7. **Best Dev Scores:** **88.5% Exact Match** and **97.4% Mean Field Accuracy** (vs. 0.0% zero-shot and 28.5% in-context).
8. **Data Audit:** Started with **1,445 raw rows**, dropped **125 rows**, fixed **175 rows**, leaving **1,320 clean rows**.

---

## 🔍 4. The 10 Anomaly Classes Found in `train.jsonl`

When they ask *"What did you find wrong with the training data?"*, list these:

1. **Syntax Truncations (25 rows):** Assistant JSON cut off mid-string (`{"category": "kyc", ... "t`) due to generation buffer limits. *Dropped.*
2. **Placeholder / Dummy Tickets (10 rows):** Messages containing `"test"`, `"test test 123"`, `"asdf asdf"` with ungrounded transactions. *Dropped.*
3. **Dev-Set Contamination / Data Leakage (30 rows):** 30 tickets in `train.jsonl` were exact character-for-character duplicates of `dev.jsonl`. Dropping them was essential to respect the rule *"Never train on dev or test rows"* and prevent fake metric inflation. *Dropped.*
4. **Internal Redundant Duplicates (60 rows):** 60 tickets duplicated twice within train. *Dropped second copy.*
5. **Legacy v1 Schema Key & Values (70 rows):** Keyed as `"prio"` instead of `"priority"` with `"low"/"medium"/"high"`. Mapped to `"P3"/"P2"/"P1"`. *Fixed.*
6. **Date Formats (30 rows):** `DD/MM/YYYY` dates normalized to ISO `YYYY-MM-DD`. *Fixed.*
7. **Un-Normalized Transaction IDs (20 rows):** IDs with spaces (`NP 9255627510`) or hyphens (`NP-9718716109`) normalized to `NP\d{10}`. *Fixed.*
8. **Rupee vs Paise Unit Errors (24 rows):** Amounts recorded in raw rupees instead of multiplying by 100 to paise, or truncated decimals (`Rs 1,68,787.25` became `168787`). Rescaled to integer paise. *Fixed.*
9. **Priority Waterfall Violations (60 rows):** Misapplied priority rules (e.g., fraud marked P3, amounts $\ge$ ₹50,000 marked P3/P2, legal threats marked P3). Recomputed following deterministic waterfall. *Fixed.*
10. **`needs_human` Inconsistencies (39 rows):** Priority P1 rows marked `needs_human: false` (violating *"true if priority is P1"*). Enforced `needs_human = (priority == 'P1') or has_repeated_contact`. *Fixed.*

---

## 📊 5. The 4 Controlled Experiments Table

| Experiment | Condition Changed | Dev Exact Match | Dev Mean Field Acc | Peak VRAM | Training Time |
|---|---|:---:|:---:|:---:|:---:|
| **Zero-Shot Base** | Short prompt, no fine-tuning | 0.0% | 0.0% | 2.2 GB | 0 min |
| **In-Context Baseline** | Full SCHEMA.md rules in prompt | 28.5% | 69.8% | 3.1 GB | 0 min |
| **Exp 1: Raw Data Ablation** | Trained on uncleaned `train.jsonl` | 59.5% | 89.2% | 4.8 GB | 28 min |
| **Exp 2: 16-bit LoRA Ablation** | FP16 precision instead of 4-bit QLoRA | 89.0% | 97.6% | 11.8 GB | 22 min |
| **Exp 3: Low Rank Ablation** | LoRA rank $r=8$ (14.97M params) vs $r=16$ | 84.0% | 95.8% | 4.6 GB | 24 min |
| **Exp 4: Loss Masking Ablation**| Full sequence loss vs completion-only | 79.5% | 94.1% | 4.9 GB | 27 min |
| **Final Fine-Tuned Model** | Clean Data + QLoRA 4-bit ($r=16$) + Completion Loss | **88.5%** | **97.4%** | **4.8 GB** | **26 min** |

---

## 🎯 6. Typical Live Defense Questions & Answers

### Q1: *"Why did you choose Qwen2.5-3B over Llama-3.2-3B or Gemma-2-2B?"*
> **Answer:**
> 1. **No Gated License:** `Qwen2.5-3B-Instruct` is ungated on Hugging Face. Llama-3.2 requires an accepted license and user token, which frequently fails in automated evaluation or fresh Colab runtimes.
> 2. **Superior Multilingual & Code Adherence:** Qwen2.5-3B has proven state-of-the-art capability in structured JSON generation and understanding Hindi/Hinglish in Latin script.
> 3. **Native System Role Support:** It natively supports system and user roles via ChatML, matching the dataset structure without prompt hacking.

### Q2: *"Why did you use Completion-Only Loss Masking?"*
> **Answer:**
> *"Standard causal LM training computes cross-entropy loss over both the prompt and the response. In customer support, user text has wide variance (informal typos, emotions, irrelevant hardware details). Computing loss on user text wastes model capacity learning customer complaint phrasing. Using `DataCollatorForCompletionOnlyLM` masks prompt tokens with `labels = -100` up to `<|im_start|>assistant\n`, concentrating 100% of gradient updates on predicting the correct JSON schema. In Experiment 4, this alone gave us a **+9.0% Exact Match boost**."*

### Q3: *"How did you decide on sequence length 512?"*
> **Answer:**
> *"We profiled the token lengths across all dataset splits. User inputs have a 95th percentile of 215 tokens and a maximum of 338 tokens. Assistant JSON targets take between 60 and 95 tokens. Total sequence length never exceeded 442 tokens. Setting `max_seq_length = 512` guaranteed **zero truncation** across train, dev, and test sets while maximizing batch size and GPU throughput."*

### Q4: *"What are the weakest areas of your final model?"*
> **Answer:**
> *"The weakest fields are **priority (95.0%)** and **txn_date (95.0%)**:
> - Priority errors occur on boundary amounts (e.g. ₹49,990 being rounded up to ₹50,000, triggering false-positive P1).
> - Date errors happen on month-rollover calendar arithmetic (e.g. '5 days ago' on July 2nd requires recognizing that June has 30 days).
> By contrast, language is 100%, and category, transaction ID, and channel are all 98.5%."*

### Q5: *"What would you do with one more week?"*
> **Answer:**
> 1. **Targeted Hard-Negative Data Augmentation:** Generate synthetic tickets near the ₹5,000 and ₹50,000 thresholds (e.g. ₹4,999 vs ₹5,001) and month-rollover calendar edge cases.
> 2. **Direct Preference Optimization (DPO):** Train on chosen/rejected pairs where rejected candidates hallucinate channels on non-transaction tickets or extract daily limits as transaction amounts.
> 3. **Speculative Decoding:** Pair `Qwen2.5-3B` with `Qwen2.5-0.5B` to cut inference latency to <40ms per ticket in production.
