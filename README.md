# StealthText Fine-Tuning: Custom AI Text Humanization 🕵️‍♂️

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA%204--bit-purple)](https://github.com/huggingface/peft)

This repository contains the complete training pipeline, dataset preprocessing tools, and inference scripts for fine-tuning open-source Large Language Models (LLMs) to transform AI-generated text into natural, human-like writing that bypasses AI detectors.

---

## 🎯 Purpose & Scope

While zero-shot prompting on frontier commercial models (e.g. Groq Llama 3.3 70B, GPT-4o) provides strong baseline performance, custom fine-tuning enables:
- **Low Inference Latency**: Deploy small models (3B–8B parameters) on dedicated hardware.
- **Cost Reduction**: Avoid per-token API charges for high-volume text humanization.
- **Model Ownership**: Maintain proprietary model weights optimized specifically for anti-detection tasks.

---

## 🏗 Repository Structure

```
StealthText-FineTuning/
├── README.md              # Documentation & Production Roadmap
├── download_dataset.py    # Preprocessing script for dataset formatting
├── train.py               # Hugging Face SFTTrainer pipeline with QLoRA & MPS support
├── test_inference.py      # Inference testing script for base model + LoRA adapter
├── requirements.txt       # Dependencies for local and GPU environments
└── results/               # Target directory for saved adapter weights
```

---

## 🚀 Quick Start

### 1. Cloud Execution (Google Colab T4 GPU)

For zero local storage requirements, run the pipeline on a free T4 GPU in Google Colab:

```bash
# Install required libraries
!pip install -q transformers peft trl accelerate bitsandbytes datasets

# Download and format dataset (5,000 samples)
!python download_dataset.py --limit 5000

# Train Qwen 2.5 3B with QLoRA (4-bit quantization)
!python train.py \
    --base_model "Qwen/Qwen2.5-3B-Instruct" \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 4 \
    --learning_rate 2e-4 \
    --use_qlora
```

*(Optional)* Push trained adapter weights to Hugging Face Hub:
```python
from huggingface_hub import notebook_login
notebook_login()

!python train.py \
    --base_model "Qwen/Qwen2.5-3B-Instruct" \
    --epochs 3 \
    --use_qlora \
    --push_to_hub \
    --hub_model_id "your-username/qwen-3b-humanizer"
```

---

### 2. Local Setup (Mac MPS / Linux CUDA / Windows)

```bash
# Clone repository
git clone https://github.com/sandhya-bdb/finetuned-stealthtext.git
cd finetuned-stealthtext

# Install dependencies
pip install -r requirements.txt

# Download and preprocess dataset
python download_dataset.py --limit 5000

# Run training (Automatically detects Apple Silicon MPS or CUDA)
python train.py --base_model "Qwen/Qwen2.5-3B-Instruct" --epochs 3 --batch_size 1 --grad_accum 8

# Run test inference
python test_inference.py \
    --base_model "Qwen/Qwen2.5-3B-Instruct" \
    --adapter_dir "./results" \
    --input_text "Artificial intelligence is changing the way people work by introducing automation to routine tasks."
```

---

## 📊 Baseline Model Evaluation & Findings

### Experimental Setup
* **Base Model**: `Qwen/Qwen2.5-3B-Instruct`
* **Dataset**: `Hello-SimpleAI/HC3` (Human-ChatGPT Comparison Corpus)
* **Methodology**: Supervised Fine-Tuning (SFT) using QLoRA 4-bit quantization on all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Training Metric**: Final evaluation loss reached **1.262**.

### Empirical Insights & Technical Bottlenecks
While training loss converged cleanly, initial output evaluation highlighted key structural challenges with standard SFT on unaligned QA corpora:
1. **Un-aligned QA Pair Limitation**: The HC3 dataset pairs AI answers with independently written human answers. Because the human target text is not an *edit* of the input AI text, the fine-tuned model tends to generate completely new responses rather than preserving original factual content and flow.
2. **Template Alignment**: Using Alpaca prompt style instead of native ChatML (`<|im_start|>user...`) leads to suboptimal instruction-following in Qwen base models.

---

## 🗺 Production-Ready Roadmap

To evolve this repository into a commercial-grade, anti-detection fine-tuning pipeline, the following 4-phase upgrade strategy is planned:

```mermaid
graph LR
    A[Phase 1: Parallel Synthetic Data] --> B[Phase 2: DPO Alignment]
    B --> C[Phase 3: Architecture & Scaling]
    C --> D[Phase 4: vLLM Production API]
```

### 🔴 Phase 1: Synthetic Parallel Revision Dataset Generation
* **Goal**: Replace unaligned QA pairs with strict, parallel sentence-level revision datasets.
* **Implementation**:
  * Utilize frontier LLMs (GPT-4o / Llama 3.3 70B) to edit raw AI text line-by-line into human writing style.
  * Enforce strict semantic preservation constraints so output text retains 100% of facts, entities, and intent.
  * Target dataset size: 15,000–25,000 parallel revision pairs.

### 🟡 Phase 2: Direct Preference Optimization (DPO) & RLHF
* **Goal**: Train the model on human preference pairs to explicitly penalize AI writing markers.
* **Implementation**:
  * Construct triplets: `(Input_AI_Text, Chosen_Humanized_Revision, Rejected_AI_Draft)`.
  * Train using `TRL`'s `DPOTrainer` to directly align model probabilities with anti-detection criteria (elevating perplexity variance and burstiness).

### 🟢 Phase 3: Model Scale & Native Template Alignment
* **Goal**: Upgrade base model architecture for superior linguistic fluidity and stylistic nuance.
* **Implementation**:
  * Upgrade base model to **`Meta-Llama/Llama-3.1-8B-Instruct`** or **`Mistral-7B-Instruct-v0.3`**.
  * Format all dataset inputs using native model chat templates (e.g. ChatML for Qwen, Llama 3 header tokens for Llama).

### 🔵 Phase 4: High-Throughput Production Deployment
* **Goal**: Serve the fine-tuned model with sub-second latency for API integration.
* **Implementation**:
  * Merge LoRA adapter weights into base model weights.
  * Containerize inference using **vLLM** or **TensorRT-LLM** with continuous batching.
  * Integrate endpoint directly into the core **[StealthText API](https://github.com/sandhya-bdb/stealthtext_AI_bypass)** backend.

---

## 🤝 Contributing & License

Contributions, issue reports, and experimental feedback are welcome!
Licensed under the [MIT License](LICENSE).
