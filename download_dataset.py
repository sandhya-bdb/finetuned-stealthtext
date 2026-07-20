import json
import argparse
from datasets import load_dataset

def prepare_hc3_dataset(output_dir=".", subset="all", limit=5000, test_size=0.1):
    print(f"Loading Hello-SimpleAI/HC3 dataset (subset: {subset})...")
    # Load dataset
    try:
        # Load the English JSONL subset directly using json loader
        data_url = f"hf://datasets/Hello-SimpleAI/HC3/{subset}.jsonl"
        print(f"Loading direct JSONL data from: {data_url}")
        dataset = load_dataset("json", data_files=data_url)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Extract all QA pairs
    pairs = []
    
    # HC3 dataset only has a 'train' split by default
    split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
    data_split = dataset[split_name]
    
    print(f"Extracting and formatting text pairs from {len(data_split)} records...")
    
    for item in data_split:
        question = item.get("question", "")
        human_answers = item.get("human_answers", [])
        gpt_answers = item.get("chatgpt_answers", [])
        
        # Make sure we have both human and GPT answers
        if not human_answers or not gpt_answers:
            continue
            
        # Pair each GPT answer with the human answers
        # To keep dataset diverse but compact, pair each GPT answer with up to 2 human answers
        for gpt_ans in gpt_answers:
            for human_ans in human_answers[:2]:
                gpt_ans = gpt_ans.strip()
                human_ans = human_ans.strip()
                if gpt_ans and human_ans:
                    pairs.append({
                        "instruction": "Humanize the following text to bypass AI detectors. Preserve the original facts, core meaning, and tone.",
                        "input": gpt_ans,
                        "output": human_ans
                    })
                    
        if len(pairs) >= limit:
            print(f"Reached targeted limit of {limit} samples.")
            pairs = pairs[:limit]
            break

    print(f"Total formatted pairs collected: {len(pairs)}")
    
    # Split into train and validation sets
    train_count = int(len(pairs) * (1 - test_size))
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]
    
    # Save as JSON Lines (jsonl)
    train_path = f"{output_dir}/train.jsonl"
    val_path = f"{output_dir}/val.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")
            
    with open(val_path, "w", encoding="utf-8") as f:
        for p in val_pairs:
            f.write(json.dumps(p) + "\n")
            
    print(f"Saved {len(train_pairs)} training samples to {train_path}")
    print(f"Saved {len(val_pairs)} validation samples to {val_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess Hello-SimpleAI/HC3 dataset for SFT")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the train/val files")
    parser.add_argument("--subset", type=str, default="all", help="Subset of HC3 to download (e.g., all, wiki_csai, reddit_eli5)")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum number of samples to process")
    parser.add_argument("--test_size", type=float, default=0.1, help="Validation set split ratio")
    
    args = parser.parse_args()
    prepare_hc3_dataset(args.output_dir, args.subset, args.limit, args.test_size)
