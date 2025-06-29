import os
import sys
import math
import time
import logging
import argparse
import traceback
import random
import numpy as np
import torch
from tqdm import tqdm
import wandb
from dotenv import load_dotenv
load_dotenv()

# Ensure required tokens are set
if os.getenv("HUGGINGFACE_TOKEN") is None:
    raise EnvironmentError("Missing Hugging Face token: Please set the HF_TOKEN environment variable.")

# HuggingFace Transformers
from transformers import (
    RobertaConfig,
    AutoConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

# HuggingFace Datasets
from datasets import load_dataset

# TRL and TRAK
from trl import apply_chat_template, is_conversational
from trak.projectors import CudaProjector, NoOpProjector, ProjectionType

# Custom utilities
import util
from util import get_checkpoints_hub, DeterministicDataCollatorForLanguageModeling
from olmo_training_utils import sft_tulu_tokenize_and_truncate_v1

# Multiprocessing
from multiprocessing import Pool, Queue, Manager


os.environ["TOKENIZERS_PARALLELISM"] = "True"

parser = argparse.ArgumentParser("gradient_extraction")
parser.add_argument("model", help="A model on the hf hub. Format: username/name_curriculum")
parser.add_argument("dataset", help="A dataset on the hf hub. Format: username/name")
parser.add_argument("--dataset_split", help="The split to access", default="train")
parser.add_argument("checkpoint_nr", help="Id of the checkpoint to extract gradients for (starting at 0)",type=int)
parser.add_argument("--gradients_per_file", help="Number of gradients per output file", type=int, nargs="?", const=1, default=1000) # 10000 = ~7.4 GB per file for BERT
parser.add_argument("--paradigm", help="Eiter 'pre', 'mlm', or 'sft'", default="mlm")
parser.add_argument("--gradients_output_path", help="The path where to store gradients at", default="./gradients")
parser.add_argument("--mode", help="Eiter 'store', or 'store_mean'", default="store")
parser.add_argument("--skip_if_gradient_folder_exists", default=False, action='store_true')
parser.add_argument("--random_projection", default=False, action='store_true')
parser.add_argument("--proj_dim", type=int, nargs="?", const=1, default=2**14)
parser.add_argument("--proj_type", type=str, default="rademacher", help="Type of projection to use: normal or rademacher.")
args = parser.parse_args()

# print("Args", args, flush=True)
# print("Cuda version:", torch.version.cuda)


model_name = args.model.split("/")[-1]
dataset_name = args.dataset.split("/")[-1]
dataset_split_name = args.dataset_split

# create output dirs
if args.random_projection:
    proj_folder_name = f"{args.proj_type}_{args.proj_dim}"
else:
    proj_folder_name = "full"

gradient_output_dir = os.path.join(args.gradients_output_path, proj_folder_name, model_name, dataset_name, dataset_split_name)


if not os.path.exists(gradient_output_dir):
    os.makedirs(gradient_output_dir, exist_ok=True)



logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')



device = "cuda:0"


def get_loss_gradient(model, example,device):
    """Computes gradient of the loss function irt to the input embeddings.

    Args:
        model: A model with a `forward` method wich returns `loss`
        example: An instance from the training data
        device: What GPU to use (e.g., cuda:0)

    Returns:
        A 1D tensor: gradient of the loss function irt to the input embeddings
    """
    # uncomment this to debug out of memory errors:

    # free_mem, total_mem = torch.cuda.mem_get_info(device=device)
    # print(f"Free memory {device}:  {(free_mem/total_mem)} {free_mem / (1024 ** 2):.2f}/{total_mem / (1024 ** 2):.2f}",flush=True)


    model.zero_grad()

    # we process one example at a time 
    example = data_collator(
        [ # <- but the dataloader expects a batch
            {
            key: value  
            for key, value in example.items() if key in ["input_ids", "attention_mask", "labels"] # only keep relevant cols
            }
        ]) 
    
    # n.b.: according to the documentaiton: "Shifting the inputs and labels to align them happens inside the model, so the data collator just copies the inputs to create the labels." https://huggingface.co/learn/nlp-course/chapter7/6#initializing-a-new-model


    # obtain input embeddings
    inputs_embeds=model.get_input_embeddings().weight[example["input_ids"]].to(device)
    inputs_embeds.retain_grad()


    outputs = model.forward(
            inputs_embeds=inputs_embeds,
            labels=example["labels"].to(device)
        )


    loss = outputs.loss 
    loss.retain_grad()
    return torch.autograd.grad(loss, inputs_embeds, retain_graph=False)[0].squeeze()
    

def get_for_checkpoint(model, projector, checkpoint_path, i_start, i_end):
    """Calculates gradients at a given checkpoint for a given subset and stores it to disk

    Args:
        checkpoint_path: Path to the checkpoint folder
        i_start: Start id from the dataset
        i_end: Stop id from the dataset (non-inclusive)

    Raises:
        e: Any error but most likely OOM
    """                
    log_prefix = f"[batch {checkpoint_path}_{i_start}_{i_end}]"
    try:      
        logging.debug(f"{log_prefix} is starting...")
        if not any([a in args.model for a in ["llama", "OLMo"]]):
            data_collator.set_epoch(util.get_epoch(checkpoint_path)) # to ensure the same masking as during training


        out_dir = os.path.join(gradient_output_dir, os.path.basename(checkpoint_path))
        out_path = os.path.join(gradient_output_dir, os.path.basename(checkpoint_path), "mean") if args.mode == "store_mean" else os.path.join(gradient_output_dir, os.path.basename(checkpoint_path),str(i_start) + "_" + str(i_end))
        os.makedirs(out_dir,exist_ok=True)
        if os.path.isfile(out_path):
            logging.info(f"{log_prefix} skipping {out_path}, already stored")
            return 
      
        
    
        gradients = None
        
        logging.debug(f"{log_prefix} is getting gradients...")


        def project(x):
            p = projector.project(x, model_id=0)
            return p

              
        gradients = torch.stack([ 
                                    project(
                                        get_loss_gradient(
                                            model, 
                                            dataset[i],
                                            device
                                        ).detach().flatten().unsqueeze(0).half()
                                    ).cpu() 
                                    for i in tqdm(range(i_start, i_end), desc=f"{log_prefix} is getting gradients...")
                                ])
        logging.debug(f"{log_prefix} ... got gradients")
        if args.mode == "store":
            torch.save( gradients, out_path)
            logging.info(f"{log_prefix} stored gradients to {out_path}")
        else:
            logging.info(f"{log_prefix} partial sum")
            return torch.sum(gradients, axis=0)
        del gradients
        return
    except:
        print(f"Exception during {checkpoint_path}_{i_start}_{i_end}", traceback.format_exc(),flush=True)



tokenizer = None


if "llama" in args.model:
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model, max_len=512)
elif "OLMo" in args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
else: # RoBERTa
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model, max_len=512)
   


dataset = None

paradigm = args.paradigm

if "alpaca" in args.dataset:
    # https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy#sampling-from-the-model-during-training
    def prompt_no_input(output,_,instruction):
        return ("Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{output}").format(instruction=instruction, output=output)


    def prompt_input(output,input,instruction):
        return ("Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}").format(instruction=instruction, input=input, output=output)


    def create_alpaca_prompt(rows):

        return [prompt_no_input(*row) if row[1] == "" else prompt_input(*row) for row in zip(*rows.values())]
    
    dataset = load_dataset(args.dataset, split=args.dataset_split) 
    dataset.set_transform(lambda x : tokenizer(create_alpaca_prompt(x), return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=4096))
    
    paradigm = "pre"
    logging.info(f"dataset format: alpaca (pre)")

elif paradigm in ["pre", "mlm"]:
    dataset = load_dataset(args.dataset, split=args.dataset_split)

    if "completion" in dataset.column_names: # hotfix for custom tulu dataset format used during olmo fine-tuning
        def transform_example(x):
            # re format...
            chat_template = \
                {"messages":  [
                    {"role": "user", "content": x["prompt"][0]},
                    {"role": "assistant", "content": x["completion"][0]}
                    ]
                }
            return sft_tulu_tokenize_and_truncate_v1(chat_template, tokenizer=tokenizer)  # ... so we can use Olmo sft code
        dataset.set_transform(transform_example)

        logging.info(f"dataset format: chat format (pre)")
    elif "text" in dataset.column_names:   
        dataset.set_transform(lambda x : tokenizer(x["text"], return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=4096 if "OLMo" in args.model else 512))

        logging.info(f"dataset format: pre")
    else:
        raise NotImplementedError
elif paradigm == "sft":
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    dataset.set_transform(lambda x : sft_tulu_tokenize_and_truncate_v1(x, tokenizer=tokenizer))
    logging.info(f"dataset format: tulu (sft)")

else:
    raise NotImplementedError

def get_data_collator(paradigm):
    if paradigm in ["mlm"]:
        return DeterministicDataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        ) 
    if paradigm in ["pre"]:
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
    if paradigm in ["sft"]:
            return DataCollatorForSeq2Seq(
                tokenizer=tokenizer
            )

data_collator = get_data_collator(paradigm)


checkpoints =  get_checkpoints_hub(args.model)


if __name__ == '__main__':
    
    checkpoint = checkpoints[args.checkpoint_nr]
    
    out_path = os.path.join(gradient_output_dir, os.path.basename(checkpoint))
    

    if args.mode == "store_mean" and os.path.isfile(os.path.join(out_path,"mean")):
        logging.info("Skipping {}, already calculated".format(out_path) )
    elif args.mode == "store" and os.path.isdir(out_path) and len(os.listdir(out_path)) == 0 and args.skip_if_gradient_folder_exists:
        logging.info("Skipping {}, because folder exists (--skip_if_gradient_folder_exists) and is empty".format(out_path) )
    else:
        
        logging.info(f"writing results to {out_path}")

        run = None
        if os.getenv("WANDB_API_KEY") is not None:
            run = wandb.init(project="gradient_extraction")
            run.name = os.path.join(args.model, os.getenv("SLURM_JOB_NAME", "?"))

      
        model = None
 
        logging.info(f"loading model {args.model}...")

        if "llama" in args.model:
            model_config = AutoConfig.from_pretrained(checkpoint)
            model = LlamaForCausalLM(config=model_config).to(device)
        elif "OLMo" in args.model:
            model = AutoModelForCausalLM.from_pretrained(args.model, revision=checkpoint, torch_dtype=torch.float16).to(device)
        else:
            model_config = AutoConfig.from_pretrained(checkpoint)
            model = RobertaForMaskedLM(config=model_config).to(device)
        model.train()
        logging.info(f"... loading done")

        
        
        projector = None
        if args.random_projection and args.mode != "store_mean":
            grad_dim = None
            logging.debug(f"inferring projection parameters ...")
            grad_dim = get_loss_gradient(model, dataset[0], device).detach().flatten().unsqueeze(0).shape[-1]
            logging.debug(f"... using grad_dim={grad_dim} proj_dim={args.proj_dim} ...")

            proj_type = ProjectionType[args.proj_type]

            logging.info(f"Projection type: {proj_type}")

            projector = CudaProjector(grad_dim=grad_dim, proj_dim=args.proj_dim,seed=42, proj_type=proj_type,device=device, max_batch_size=8)
            logging.info(f"... set up projector done")
        else:
            logging.info(f"storing full gradients")
            projector = NoOpProjector()
       
        
        results = []
        for i in range(0, len(dataset), args.gradients_per_file): 
            start_time = time.time()

            results.append(get_for_checkpoint(model, projector, checkpoint,i, i + args.gradients_per_file))   

            if run is not None:
                run.log({"gradients/time_per_chunk": time.time()-start_time},commit=False)
                run.log({"gradients/time_per_example": (time.time()-start_time)/args.gradients_per_file},commit=False)
        
        if args.mode == "store_mean":
            logging.info("aggregating mean gradients")
            t = torch.stack(results).sum(axis=0) / len(dataset)
            torch.save(t, os.path.join(out_path, "mean"))
            logging.info(f"stored mean gradients")

        logging.info(f"task complete!")
        if run is not None:
            run.finish()