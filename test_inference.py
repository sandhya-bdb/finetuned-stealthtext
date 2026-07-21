import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert editor specializing in text humanization. "
    "Rewrite the input text to sound completely natural, human-written, and engaging "
    "with natural sentence variation, while strictly preserving all original facts and core meaning."
)

def generate_humanized_text(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    print(f"Loading tokenizer: {args.base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.base_model}...", flush=True)
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32)
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    
    if device == "mps":
        model_kwargs["device_map"] = {"": "mps"}
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"
        
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    print(f"Loading fine-tuned adapter from {args.adapter_dir}...", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    
    if args.merge:
        print("Merging weights for optimized inference...", flush=True)
        model = model.merge_and_unload()

    # Format the prompt using ChatML / native chat template matching training
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.input_text}
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = (
            f"### System:\n{args.system_prompt}\n\n"
            f"### Input:\n{args.input_text}\n\n"
            f"### Response:\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\nGenerating humanized response...", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode generation
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
    print("\n" + "="*40 + " INPUT AI TEXT " + "="*40, flush=True)
    print(args.input_text, flush=True)
    print("="*40 + " HUMANIZED OUTPUT " + "="*40, flush=True)
    print(response, flush=True)
    print("="*95, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using fine-tuned humanizer adapter")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Hugging Face base model name")
    parser.add_argument("--adapter_dir", type=str, default="./results", help="Directory where adapter is saved")
    parser.add_argument("--input_text", type=str, default="Artificial intelligence has revolutionized many industries by automating complex tasks. However, it is essential to ensure that AI output is checked for accuracy and quality before publication.", help="Text to humanize")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Generation top_p")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA weights into base model")
    
    args = parser.parse_args()
    generate_humanized_text(args)
