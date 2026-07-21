# StealthText Fine-Tuning: Custom AI Text Humanization 🕵️‍♂️

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA%204--bit-purple)](https://github.com/huggingface/peft)

This repository contains the complete training pipeline, dataset preprocessing tools, Google Colab notebook, and inference scripts for fine-tuning open-source LLMs (`Qwen/Qwen2.5-3B-Instruct`, `Llama 3.1 8B`) to transform AI-generated text into natural, human-written prose that bypasses AI detectors.

---

## 🚀 Key Improvements for Higher Output Quality

To achieve high-quality humanized output when fine-tuning in Colab:
1. **Native ChatML Prompt Formatting**: Uses model-native `<|im_start|>system...` ChatML templates via `tokenizer.apply_chat_template` rather than raw Alpaca templates, eliminating format confusion in instruction-following models.
2. **System Prompt Conditioning**: Introduces a dedicated text-humanization system prompt during both SFT training and inference.
3. **Data Quality Filtering**: Excludes low-word-count snippets (<20 words) to ensure the model learns rich paragraph-level perplexity and burstiness transformations.
4. **Optimized Cosine Learning Schedule**: Incorporates warmup and cosine decay (`lr_scheduler_type="cosine"`) for smooth LoRA weight convergence.

---

## 🏗 Repository Structure

```
StealthText-FineTuning/
├── README.md                          # Documentation & Production Roadmap
├── StealthText_FineTuning_Colab.ipynb # Ready-to-run 1-Click Google Colab Notebook
├── download_dataset.py                # Preprocessing script with ChatML & quality filtering
├── train.py                           # Hugging Face SFTTrainer pipeline with QLoRA & ChatML
├── test_inference.py                  # Inference script matching training prompt template
├── requirements.txt                   # Dependencies for local and Colab GPU environments
└── results/                           # Target directory for saved adapter weights
```

---

## ⚡ Quick Start (Google Colab Execution)

Open and run **[`StealthText_FineTuning_Colab.ipynb`](file:///Users/sandhyabantiduttaborah/Desktop/StealthText-FineTuning/StealthText_FineTuning_Colab.ipynb)** in Google Colab (with a free T4 GPU enabled):

```bash
# 1. Install required libraries
!pip install -q -U transformers peft trl accelerate bitsandbytes datasets

# 2. Preprocess 2,500 quality training samples with system prompt & length filter
!python download_dataset.py --limit 2500 --min_words 20

# 3. Fine-tune Qwen 2.5 3B with QLoRA (4-bit quantization)
!python train.py \
    --base_model "Qwen/Qwen2.5-3B-Instruct" \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 4 \
    --learning_rate 2e-4 \
    --use_qlora

# 4. Test inference output quality
!python test_inference.py \
    --base_model "Qwen/Qwen2.5-3B-Instruct" \
    --adapter_dir "./results" \
    --input_text "Artificial intelligence systems have automated routine tasks across many sectors..." \
    --temperature 0.7 \
    --top_p 0.9
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

## 💻 Local Setup (Mac MPS / Linux CUDA)

```bash
# Install dependencies
pip install -r requirements.txt

# Download and preprocess dataset
python download_dataset.py --limit 2500 --min_words 20

# Run training (Automatically detects Apple Silicon MPS or CUDA)
python train.py --base_model "Qwen/Qwen2.5-3B-Instruct" --epochs 3 --batch_size 1 --grad_accum 8

# Run test inference
python test_inference.py \
    --base_model "Qwen/Qwen2.5-3B-Instruct" \
    --adapter_dir "./results" \
    --input_text "Artificial intelligence is changing the way people work..."
```

---

## 🗺 Roadmap for Next-Level Output

1. **Parallel Synthetic Revision Dataset**: Replace QA dataset pairs with direct line-by-line LLM revisions (Llama 3.3 70B editing AI text into human prose).
2. **DPO (Direct Preference Optimization)**: Train using `TRL`'s `DPOTrainer` on `(AI_Text, Humanized_Text, Raw_AI)` triplets.
3. **8B Model Scale**: Scale up base model to `Meta-Llama/Llama-3.1-8B-Instruct`.
