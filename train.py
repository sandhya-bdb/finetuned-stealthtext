import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def train(args):
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 4-bit quantization configuration (Colab NVIDIA GPUs e.g. T4/L4/A100)
    quantization_config = None
    if device == "cuda" and args.use_qlora:
        print("Configuring 4-bit QLoRA quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        print("Quantization bypassed (running on CPU/MPS or qlora disabled).")

    # Load dataset
    print(f"Loading datasets from {args.train_file} and {args.val_file}...")
    data_files = {"train": args.train_file}
    if os.path.exists(args.val_file):
        data_files["validation"] = args.val_file
    
    dataset = load_dataset("json", data_files=data_files)
    
    # Load tokenizer and base model
    print(f"Loading tokenizer and base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {}
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32)
        model_kwargs["torch_dtype"] = torch_dtype
        if device == "mps":
            model_kwargs["device_map"] = {"": "mps"}
        elif device == "cuda":
            model_kwargs["device_map"] = "auto"
            
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        **model_kwargs
    )
    
    # Configure LoRA adapter
    print("Configuring LoRA parameters...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # Custom ChatML / Native Chat Template Formatter function
    def formatting_prompts_func(example):
        if "messages" in example and example["messages"]:
            # If batched input
            if isinstance(example["messages"][0], list):
                outputs = []
                for msgs in example["messages"]:
                    text = tokenizer.apply_chat_template(msgs, tokenize=False)
                    outputs.append(text)
                return outputs
            else:
                return tokenizer.apply_chat_template(example["messages"], tokenize=False)
        else:
            # Fallback to instruction format if messages column is absent
            if isinstance(example.get("instruction"), list):
                output_texts = []
                for i in range(len(example["instruction"])):
                    text = (
                        f"### Instruction:\n{example['instruction'][i]}\n\n"
                        f"### Input:\n{example['input'][i]}\n\n"
                        f"### Response:\n{example['output'][i]}"
                    )
                    output_texts.append(text)
                return output_texts
            else:
                return (
                    f"### Instruction:\n{example.get('instruction', '')}\n\n"
                    f"### Input:\n{example.get('input', '')}\n\n"
                    f"### Response:\n{example.get('output', '')}"
                )

    tokenizer.model_max_length = args.max_seq_length

    print("Setting up training arguments...")
    fp16 = device == "cuda" and not torch.cuda.is_bf16_supported()
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    
    try:
        from trl import SFTConfig
        sft_config = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if "validation" in dataset else "no",
            fp16=fp16,
            bf16=bf16,
            report_to="none",
            max_seq_length=args.max_seq_length,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id if args.push_to_hub else None
        )
    except Exception:
        sft_config = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if "validation" in dataset else "no",
            fp16=fp16,
            bf16=bf16,
            report_to="none",
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id if args.push_to_hub else None
        )
    
    from trl import SFTTrainer
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"] if "validation" in dataset else None,
            peft_config=lora_config,
            formatting_func=formatting_prompts_func,
            processing_class=tokenizer,
            args=sft_config
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"] if "validation" in dataset else None,
            peft_config=lora_config,
            formatting_func=formatting_prompts_func,
            tokenizer=tokenizer,
            args=sft_config
        )
    
    print("Starting SFT training...")
    trainer.train()
    
    print(f"Saving fine-tuned adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if args.push_to_hub:
        print(f"Pushing fine-tuned adapter to Hugging Face Hub: {args.hub_model_id}...")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        
    print("Training successfully completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on instruction pairs using QLoRA/SFT")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Hugging Face base model name")
    parser.add_argument("--train_file", type=str, default="train.jsonl", help="Path to training jsonl file")
    parser.add_argument("--val_file", type=str, default="val.jsonl", help="Path to validation jsonl file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save output adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length for SFT")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--use_qlora", action="store_true", default=True, help="Enable 4-bit QLoRA on CUDA")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default="", help="Hugging Face model ID to push to")
    
    args = parser.parse_args()
    
    if args.push_to_hub and not args.hub_model_id:
        parser.error("--hub_model_id is required if --push_to_hub is enabled")
        
    train(args)
