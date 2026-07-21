import os
import argparse
from huggingface_hub import HfApi, login

def publish_adapter(args):
    adapter_dir = args.adapter_dir
    if not os.path.exists(adapter_dir):
        print(f"❌ Error: Adapter directory '{adapter_dir}' does not exist.")
        return
        
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print(f"❌ Error: Could not find valid adapter files in '{adapter_dir}'. Expected 'adapter_config.json' and 'adapter_model.safetensors'.")
        return

    print(f"📦 Found fine-tuned adapter files in '{adapter_dir}' (~60MB).")

    if args.token:
        print("Authenticating with Hugging Face...")
        login(token=args.token)
    
    api = HfApi()
    
    print(f"🚀 Uploading fine-tuned adapter to Hugging Face Hub: {args.repo_id}...")
    try:
        api.create_repo(repo_id=args.repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=adapter_dir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Publish fine-tuned StealthText QLoRA adapter weights"
        )
        print(f"✅ Success! Your fine-tuned model adapter is published and live at:")
        print(f"👉 https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\n💡 Tip: Make sure you have logged in using `huggingface-cli login` or passed `--token YOUR_HF_TOKEN`.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish already fine-tuned adapter model weights to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Model ID (e.g., username/stealthtext-qwen3b-humanizer)")
    parser.add_argument("--adapter_dir", type=str, default="./results", help="Directory containing fine-tuned adapter weights")
    parser.add_argument("--token", type=str, default="", help="Optional Hugging Face Write Token")
    
    args = parser.parse_args()
    publish_adapter(args)
