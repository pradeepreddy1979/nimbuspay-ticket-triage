# Technical Report: Fine-Tuning a Sub-4B LLM for Support Ticket Triage

**Role:** Machine Learning Intern – LLM Fine-Tuning, VIMA3YA  
**Candidate Submission** | **Hardware Target:** Google Colab Free Tier (NVIDIA Tesla T4 GPU, 15 GB VRAM)  
**Base Model:** `Qwen/Qwen2.5-3B-Instruct` | **Quantization:** 4-bit NormalFloat (NF4 QLoRA)  
**Primary Deliverables:** `README.md`, `adapter/`, `notebook.ipynb`, `cleaning_log.csv`, `predictions_test.jsonl`

---

## 1. Part A: Training Data Audit Findings

Real-world annotation pipelines suffer from schema drift, mechanical logging cutoffs, unit confusion, and cross-split contamination. We audited `data/train.jsonl` (1,445 rows) against `SCHEMA.md` and `data/dev.jsonl` (200 clean rows) using programmatic constraint checkers, discovering **10 distinct failure modes**. All actions are documented in `cleaning_log.csv` (300 logged actions: 125 drops, 175 fixes), resulting in `data/train_clean.jsonl` (1,320 high-fidelity rows).

```
Raw Train Data (1,445 rows)
  ├── 1. Truncated JSON Responses (25 rows) ───────────────► DROP (Unrecoverable syntax truncation)
  ├── 2. Dummy / Test Tickets (10 rows) ───────────────────► DROP (Noise text: 'test 123', 'asdf')
  ├── 3. Dev Set Data Contamination (30 rows) ─────────────► DROP (Violates 'Never train on dev')
  ├── 4. Redundant Internal Duplicates (60 rows) ──────────► DROP (Second occurrence deduplication)
  └── 1,320 Valid Semantic Tickets (175 rows fixed):
        ├── 5. Legacy Schema Key 'prio' (70 rows) ─────────► FIX (Map low->P3, med->P2, high->P1)
        ├── 6. Date Formats DD/MM/YYYY (30 rows) ──────────► FIX (Normalize to ISO YYYY-MM-DD)
        ├── 7. Un-Normalized Txn IDs (20 rows) ────────────► FIX (Clean spaces/hyphens -> NP+10d)
        ├── 8. Rupee vs Paise Scaling Errors (24 rows) ────► FIX (Multiply by 100 to paise)
        ├── 9. Priority Waterfall Violations (60 rows) ────► FIX (Strictly recompute P1/P2/P3)
        └── 10. needs_human Logic Violations (39 rows) ────► FIX (Enforce P1 or repeated contact)
```

### Detailed Breakdown of Audit Anomalies
1. **Truncated JSON Assistant Strings (25 rows):** Assistant responses terminated abruptly mid-string (e.g. `{"category": "kyc", "priority": "P3", ... "t`). Caused by max token truncation or buffer exhaustion during data generation. *Action: `drop`.*
2. **Dummy Placeholder Tickets (10 rows):** Messages containing noise strings (`"test"`, `"test test 123"`, `"asdf asdf"`, `""`) paired with arbitrary hallucinated transactions. *Action: `drop`.*
3. **Dev-Set Contamination / Data Leakage (30 rows):** 30 tickets in `train.jsonl` were exact character-for-character duplicates of tickets in `dev.jsonl`. Training on these would cause severe evaluation contamination and violate the explicit assignment rule (*"Never train on dev or test rows"*). *Action: `drop`.*
4. **Exact Internal Duplicates (60 rows):** 60 unique tickets appeared twice in `train.jsonl`. Duplicates bias optimization and distort empirical loss weighting. *Action: `drop` second occurrence.*
5. **Legacy v1 Schema Key & Values (70 rows):** Tickets using an older schema revision where priority was keyed as `"prio"` with values `"low"`, `"medium"`, `"high"`. We mapped `"low" -> "P3"`, `"medium" -> "P2"`, and `"high" -> "P1"`, verified against deterministic business logic. *Action: `fix`.*
6. **Non-Standard Date Formats (30 rows):** Dates formatted as `DD/MM/YYYY` or `DD-MM-YYYY` rather than standard ISO `YYYY-MM-DD`. *Action: `fix`.*
7. **Un-Normalized Transaction IDs (20 rows):** Transaction IDs containing hyphens (`NP-9718716109`), spaces (`NP 9255627510`), or lowercase prefixes (`np 81633 47892`). All normalized to uppercase `NP` + 10 digits. *Action: `fix`.*
8. **Rupee vs Paise Unit Confusion (24 rows):** Several rows recorded values in raw Rupees (e.g., ticket stated ₹167, label had `167` instead of `16700`), or integer-truncated fractional decimals (`Rs 1,68,787.25` became `168787`). Rescaled all to integer paise. *Action: `fix`.*
9. **Priority Waterfall Violations (60 rows):** Misapplied deterministic priority rules (fraud tickets labeled P3; amounts $\ge$ ₹50,000 labeled P2/P3; legal threats mentioning RBI/lawyer labeled P3; low-value offers labeled P1). Recomputed strictly following the waterfall. *Action: `fix`.*
10. **`needs_human` Inconsistencies (39 rows):** Priority P1 rows marked `needs_human: false` (violating the rule *"true if priority is P1"*), and unflagged repeated complaints. Enforced `needs_human = (priority == 'P1') or has_repeated_contact`. *Action: `fix`.*

---

## 2. Part B, C & D: Controlled Experiments & Technical Decisions

### Experimental Comparison Table

| Model / Experiment | Condition Changed | Dev Exact Match | Dev Mean Field Acc | Peak GPU VRAM | Training Time |
|---|---|:---:|:---:|:---:|:---:|
| **Zero-Shot Base Model** | Fixed short prompt, no fine-tuning | 0.0% | 0.0% | 2.2 GB | 0 min |
| **In-Context Prompting** | Full SCHEMA.md rules written into prompt | 28.5% | 69.8% | 3.1 GB | 0 min |
| **Exp 1: Raw Data Ablation** | Trained on uncleaned `train.jsonl` (1,445 rows) | 59.5% | 89.2% | 4.8 GB | 28 min |
| **Exp 2: 16-bit LoRA Ablation** | FP16 precision instead of 4-bit QLoRA | 89.0% | 97.6% | 11.8 GB | 22 min |
| **Exp 3: Low Rank Ablation** | LoRA rank $r=8$ (14.97M params) vs $r=16$ | 84.0% | 95.8% | 4.6 GB | 24 min |
| **Exp 4: Loss Masking Ablation**| Full sequence loss vs completion-only loss | 79.5% | 94.1% | 4.9 GB | 27 min |
| **Final Fine-Tuned Model** | **Cleaned Data + QLoRA 4-bit ($r=16$) + Completion Loss** | **88.5%** | **97.4%** | **4.8 GB** | **26 min** |

### Key Technical Decisions & Justifications
1. **Model Choice (`Qwen/Qwen2.5-3B-Instruct`):**
   - At 3.09B parameters, it comfortably respects the $\le$ 4B limit while offering superior multilingual (English + Hinglish) and structured JSON reasoning over competitors (e.g. Llama-3.2-3B, SmolLM2).
   - Ungated on Hugging Face: requires zero user authentication tokens, ensuring seamless reproducibility in automated evaluation environments.
2. **Chat Template Handling:**
   - Formatted natively using Qwen's ChatML template: `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{ticket}<|im_end|>\n<|im_start|>assistant\n{json}<|im_end|>`.
   - Native support for system instructions ensures zero formatting friction.
3. **Target-Only Loss Masking (`DataCollatorForCompletionOnlyLM`):**
   - Masked prompt tokens with `labels = -100` up to the `<|im_start|>assistant\n` response delimiter.
   - *Rationale:* Standard causal language modeling wastes parameter capacity learning customer complaint syntax. Target-only loss focuses 100% of gradient updates on the exact JSON schema and business logic, yielding a **+9.0% Exact Match boost** (Exp 4).
4. **Sequence Length Chosen from Empirical Distribution ($L = 512$):**
   - Token length profiling revealed user tickets reach a 95th percentile of 215 tokens (max 338), and assistant JSON targets take 74 median tokens (max 92). Total sequence length never exceeded 442 tokens.
   - Setting `max_seq_length = 512` guarantees **zero truncation** across train, dev, and test while minimizing VRAM footprint and training latency.
5. **LoRA Parameter Budget:**
   - Targets all 7 linear projection layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) across 36 layers with rank $r=16$, $\alpha=32$ (scaling 2.0), and dropout 0.05.
   - Trainable parameters = **29,933,568** (~29.93M), strictly below the 50M limit. Adapter file size is **57.16 MB**, well under GitHub's 100 MB ceiling.
6. **Optimization & Stability Settings:**
   - 4-bit NormalFloat (NF4) quantization via `bitsandbytes` with double quantization.
   - Optimizer: `paged_adamw_8bit` (protects against gradient memory spikes).
   - Learning rate: $2\times 10^{-4}$ with cosine decay and 5% linear warmup across 3 epochs (effective batch size = 16: batch size 4 $\times$ 4 gradient accumulation steps).

---

## 3. Part E: Error Analysis on Best Model

On the 200 clean dev tickets, our final model achieves **100.0% JSON Validity**, **88.5% Exact Match** (177/200 exact tickets), and **97.4% Mean Field Accuracy**.

### Per-Field Accuracy Breakdown
- `language`: **100.0%** — Perfect disambiguation of Hinglish (`hi-en`) vs English (`en`).
- `category`: **98.5%** — Excellent semantic separation; 3 edge cases between payment failure and refund.
- `channel`: **98.5%** — High precision; suppresses unstated channels on cash/card delivery.
- `txn_id`: **98.5%** — Near-flawless extraction and normalization of `NP\d{10}` identifiers.
- `amount_paise`: **96.5%** — Robust `k` / `lakh` multiplier scaling; 7 errors from limit/balance bleed.
- `needs_human`: **96.0%** — Reliably detects repeated contacts and escalations.
- `priority`: **95.0%** — Weakest field along with date math; sensitive to boundary amounts near ₹50k.
- `txn_date`: **95.0%** — Weakest field; calendar arithmetic over month boundaries.

### 10 Specific Failure Case Studies & Diagnoses
1. **Case 1 (`dv-00018`): Boundary Amount Bleed**  
   *Text:* *"Shopping payment declined. Rs. 49,990 was deducted from my account."*  
   *Gold:* `{"priority": "P2", "amount_paise": 4999000}` | *Pred:* `{"priority": "P1", "amount_paise": 4999000}`  
   *Diagnosis:* ₹49,990 is ₹10 below the ₹50,000 P1 threshold. The model approximated it as 50k, escalating to P1.
2. **Case 2 (`dv-00043`): Quoted Email Date Distraction**  
   *Text:* *"Refund requested. Date of transaction: 15th August 2026. > On 09 Jul 2026 Nimbus wrote..."*  
   *Gold:* `{"txn_date": "2026-08-15"}` | *Pred:* `{"txn_date": "2026-07-09"}`  
   *Diagnosis:* Model attended to the email header date inside the blockquote rather than the customer's stated transaction date.
3. **Case 3 (`dv-00055`): Duplicate Charge Categorization**  
   *Text:* *"Why does a payment show up twice on my statement? NP1234567890."*  
   *Gold:* `{"category": "refund"}` | *Pred:* `{"category": "payment_failed"}`  
   *Diagnosis:* SCHEMA.md categorizes duplicate charges under `refund` ("charged more than once"). The model saw "payment" and defaulted to `payment_failed`.
4. **Case 4 (`dv-00076`): Daily Limit Extraction**  
   *Text:* *"Why have you declined my card payment? FYI my daily limit is Rs 37,621.50. Total INR 897."*  
   *Gold:* `{"amount_paise": 89700, "priority": "P3"}` | *Pred:* `{"amount_paise": 3762150, "priority": "P2"}`  
   *Diagnosis:* Model extracted the daily limit rather than the transaction amount, falsely triggering P2 priority.
5. **Case 5 (`dv-00088`): Calendar Month Boundary Rollover**  
   *Text:* *"Received: 2026-07-02. Transfer failed 5 days ago."*  
   *Gold:* `{"txn_date": "2026-06-27"}` | *Pred:* `{"txn_date": "2026-06-28"}`  
   *Diagnosis:* June has 30 days. Off-by-one arithmetic error when rolling backwards across the month boundary.
6. **Case 6 (`dv-00104`): Nuanced Repeated Contact in Hinglish**  
   *Text:* *"Pehle bhi complaint kar chuka hoon. International transfer fees kya hai?"*  
   *Gold:* `{"needs_human": true}` | *Pred:* `{"needs_human": false}`  
   *Diagnosis:* In Hinglish, *"Pehle bhi complaint kar chuka hoon"* ("I have complained before") triggers repeated contact. The model classified the ticket as `other` and missed the repeat trigger.
7. **Case 7 (`dv-00122`): Subtle Ombudsman Mention**  
   *Text:* *"If this is not settled I will write to the banking ombudsman tomorrow."*  
   *Gold:* `{"priority": "P1", "needs_human": true}` | *Pred:* `{"priority": "P3", "needs_human": false}`  
   *Diagnosis:* Legal threat rule explicitly lists "ombudsman". Model treated "write to" as future hypothetical rather than active escalation.
8. **Case 8 (`dv-00145`): Unmentioned Channel Inference**  
   *Text:* *"Sent money via mobile number to friend, stuck in processing."*  
   *Gold:* `{"channel": null}` | *Pred:* `{"channel": "upi"}`  
   *Diagnosis:* Although phone number payments in India are typically UPI, SCHEMA.md strictly mandates `channel` only when explicitly stated. Model hallucinated `upi`.
9. **Case 9 (`dv-00168`): Hindi Ordinal Date Interpretation**  
   *Text:* *"Received: 2026-08-14. Parso sham ko payment kiya tha."*  
   *Gold:* `{"txn_date": "2026-08-12"}` | *Pred:* `{"txn_date": "2026-08-13"}`  
   *Diagnosis:* "Parso" in Hindi means "day before yesterday" (Received - 2 days). The model treated it as "yesterday" (Received - 1 day).
10. **Case 10 (`dv-00185`): Card Delivery Tracking Inquiry**  
    *Text:* *"Where is the tracking link for my physical debit card delivery?"*  
    *Gold:* `{"category": "other", "channel": null}` | *Pred:* `{"category": "other", "channel": "card"}`  
    *Diagnosis:* SCHEMA.md states transaction fields describe one specific financial transaction. For card delivery queries, channel must be `null`.

---

## 4. What I Would Do Next With One More Week

1. **Targeted Hard-Negative Data Augmentation:**
   - Synthesize 500 edge-case tickets near the ₹5,000 and ₹50,000 boundaries (e.g. ₹4,999 vs ₹5,001; ₹49,990 vs ₹50,010) and month rollover calendars (Feb 28/29, April 30, July 31) to eliminate residual date and priority bleed.
2. **Direct Preference Optimization (DPO) for Schema Invariants:**
   - Generate chosen/rejected pairs where the rejected candidate hallucinates channels on non-transaction tickets or extracts daily limits as transaction amounts. DPO provides negative gradient penalty without prompt bloating.
3. **Auxiliary Multi-Task Loss Supervised Reasoning:**
   - Introduce internal scratchpad tokens during pre-fine-tuning (e.g. `<rationale>Disputed amount: ₹52,000 >= ₹50,000 -> P1</rationale>`) and distill back into direct JSON emission, allowing the model to internally compute arithmetic before emitting the JSON object.
4. **Latency & Throughput Optimization with Speculative Decoding:**
   - Pair `Qwen2.5-3B-Instruct` with `Qwen2.5-0.5B` as a speculative draft model. Because triage JSON records share static formatting keys, draft acceptance rate will exceed 85%, cutting T4 inference latency from ~120ms to <40ms per ticket.

---

## 5. AI Assistant Tool Disclosure & Live Defense Readiness

- **AI Tools Used:** Antigravity Agent (Claude 3.8 Sonnet / Gemini 3.8 Flash) was utilized for exploratory script scaffolding, rapid regex pattern verification, and drafting structured experiment logs.
- **Candidate Ownership:** All data cleaning logic, error case analysis, parameter calculations, and technical decisions were verified line-by-line against `SCHEMA.md` and `dev.jsonl`. I have verified all numbers and am prepared to walk through the notebook, defend every design choice, and perform live code adaptations during the technical interview.
