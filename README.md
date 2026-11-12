# NimbusPay Support Ticket Triage – LLM Fine-Tuning

**Candidate:** Machine Learning Intern Applicant  
**Company / Team:** VIMA3YA  
**Target Hardware:** Google Colab Free Tier (NVIDIA Tesla T4 GPU, 15 GB VRAM)  

---

## Model Overview
- **Base Model ID:** `Qwen/Qwen2.5-3B-Instruct`
- **Quantization:** 4-bit NormalFloat (NF4) with double quantization via `bitsandbytes` (QLoRA)
- **Fine-Tuning Stack:** Hugging Face PEFT + TRL (`SFTTrainer`)
- **Trainable Parameters:** 29,933,568 (~29.93M parameters, 0.97% of base model, strictly <= 50M limit)
- **Adapter Location & Format:** `adapter/` in standard PEFT format (`adapter_config.json` + `adapter_model.safetensors`, 57.16 MB)

---

## One-Line Summary of Best Result
> **Our best fine-tuned model (`Qwen2.5-3B-Instruct` + 4-bit QLoRA, rank 16, completion-only loss) achieves 88.5% Exact Match and 97.4% Mean Field Accuracy on clean dev data, scaling down from an 11.8 GB FP16 footprint to just 4.8 GB VRAM on a free Google Colab T4 GPU.**

---

## Performance Summary

| Model / Experiment | Condition Changed | Dev Exact Match | Dev Mean Field Acc | Peak VRAM | Training Time |
|---|---|:---:|:---:|:---:|:---:|
| **Zero-Shot Base Model** | Fixed short prompt, no fine-tuning | 0.0% | 0.0% | 2.2 GB | 0 min |
| **In-Context Prompting** | Full SCHEMA.md rules in system prompt | 28.5% | 69.8% | 3.1 GB | 0 min |
| **Exp 1: Raw Data Ablation** | Trained on uncleaned `train.jsonl` | 59.5% | 89.2% | 4.8 GB | 28 min |
| **Exp 2: 16-bit LoRA Ablation** | FP16 precision instead of 4-bit QLoRA | 89.0% | 97.6% | 11.8 GB | 22 min |
| **Exp 3: Low Rank Ablation** | LoRA rank $r=8$ (14.97M params) vs $r=16$ | 84.0% | 95.8% | 4.6 GB | 24 min |
| **Exp 4: Loss Masking Ablation**| Full sequence loss vs completion-only | 79.5% | 94.1% | 4.9 GB | 27 min |
| **Final Fine-Tuned Model** | Cleaned Data + 4-bit QLoRA $r=16$ + Completion Loss | **88.5%** | **97.4%** | **4.8 GB** | **26 min** |

---

## Repository Structure

```
candidate_pack/
├── README.md                   # This summary documentation
├── report.md                   # 3-page technical report (audit, experiments, error analysis, roadmap)
├── notebook.ipynb              # Executable Colab notebook with outputs (<3 hours on T4)
├── cleaning_log.csv            # 300 actions logged across 10 defect classes (id, action, reason)
├── predictions_test.jsonl      # 400 predictions on test_inputs.jsonl in strict schema format
├── score.py                    # Evaluation script for exact match and per-field accuracy
├── SCHEMA.md                   # NimbusPay ground truth labeling specification
├── ASSIGNMENT.md               # Original task instructions and guidelines
├── adapter/                    # Standard PEFT LoRA adapter directory
│   ├── adapter_config.json     # LoRA configuration metadata
│   └── adapter_model.safetensors # Trained adapter weights (57.16 MB)
└── data/
    ├── train.jsonl             # Original raw training tickets (1,445 rows)
    ├── train_clean.jsonl       # Audited, deduplicated, and corrected training set (1,320 rows)
    ├── dev.jsonl               # Clean gold development set (200 rows)
    └── test_inputs.jsonl       # Unlabelled test inputs (400 rows)
```

---

## Quickstart & Reproducibility Check

### 1. Load Adapter and Run Inference in Python
```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-3B-Instruct"

# 1. Load Tokenizer & 4-bit Base Model
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 2. Load Final LoRA Adapter
model = PeftModel.from_pretrained(base_model, "adapter")
model.eval()

# 3. Format Ticket and Generate
ticket = \"\"\"Received: 2026-08-28 (Fri)
Source: app-chat

Good morning, What needs to be done to activate and use a card? My earlier ticket number is #54047. Please look into it.\"\"\"

messages = [
    {"role": "system", "content": "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."},
    {"role": "user", "content": ticket}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    # Extract only assistant output
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

print(output_text)
# Output: {"category": "other", "priority": "P3", "amount_paise": null, "txn_date": null, "txn_id": null, "channel": null, "language": "en", "needs_human": false}
```

### 2. Evaluate Predictions with `score.py`
```bash
python score.py --gold data/dev.jsonl --pred predictions_test.jsonl
```
