import torch
import os
from huggingface_hub import HfFolder, login
from datasets import load_dataset
import json
from dotenv import load_dotenv
import argparse
import torch.nn.functional as F
from tqdm import tqdm

# Environment variables from .env file
load_dotenv()

# Hugging Face log-in
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

login(token=hf_token)

# Command-line arguments parse
parser = argparse.ArgumentParser("explainability")
parser.add_argument("--func", help="Influence estimate method: 'dot', 'cosine', or 'both'.", choices=["dot", "cosine", "both"], required=True)
parser.add_argument("--dataset", help="Dataset to load from Huggingface Hub.", required=True)
parser.add_argument("--train_data_path", help="Path to training gradients.", required=True)
parser.add_argument("--test_data_path", help="Path to test gradients.", required=True)
parser.add_argument("--where", required=False)
parser.add_argument("--mapped", help="Whether to include the sample information in the output.", required=False, default="no")
args = parser.parse_args()

# Dataset and gradients loading
dataset = load_dataset(args.dataset, split="train")
train_grads = torch.load(args.train_data_path)
test_grads = torch.load(args.test_data_path)

# Dimension compatibility check
train_grads_size = train_grads.size()
test_grads_size = test_grads.size()

print(f"Dimensions of train gradients: {train_grads_size}")
print(f"Dimensions of test gradients: {test_grads_size} \n")

if train_grads_size[-1] != test_grads_size[-1]:
    raise ValueError(f"Incompatible gradient dimensions.")

include_mapping = args.mapped.lower() == "yes"

if include_mapping:
    print("Including full test sample information in the output. \n")
else:
    print("Storing scores only. \n")

# A method to apply
methods = ["dot", "cosine"] if args.func == "both" else [args.func]

# Normalized gradients for cosine
train_grads_normalized = F.normalize(train_grads, p=2, dim=-1) if "cosine" in methods else None

# Compute influence scores
for method in methods:
    output_dir = f"./explainability/{train_grads_size[-1]}/{args.where}/{method}"
    os.makedirs(output_dir, exist_ok=True)

    for idx, single_test_grad in tqdm(enumerate(test_grads), total=len(test_grads), desc=f"Computing {method} scores for each test sample"):

        if method == "dot":
            scores = torch.sum(train_grads * single_test_grad, dim=-1)

        elif method == "cosine":
            test_grad_normalized = F.normalize(single_test_grad, p=2, dim=-1)
            scores = torch.sum(train_grads_normalized * test_grad_normalized, dim=-1)

        # Combining scores with corresponding samples
        structured_data = []
        for i in range(len(scores)):
            
            if include_mapping:
                structured_data.append({
                    "score": float(scores[i].item()),
                    **dataset[i]
                })
            else:
                structured_data.append({
                    "score": float(scores[i].item())
                })

        # Sorting in descending order if include_mapping and saving
        if include_mapping:
            structured_data.sort(key=lambda x: x["score"], reverse=True)

        output_file = os.path.join(output_dir, f"{args.dataset.replace('/', '_')}_test_{idx}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2)

    print(f"Saved {method} results to {output_dir}.")
