# Context

The codes above are the results for the Data Analysis Project on *Gradient dimensionality reduction for instance-based
explainability of Large Language Models*.

Many thanks to `loris3` for providing a big part of these codes. My contributions, file contents and usage are shortly explained below.

# Main results

The end results are contained in `DAP_main_results.ipynb` file. These will be explained in more detail in the report and presentation.

`Data_sets_sample_and_analysis.ipynb` contains codes that were used to sample and push the train and test data sets to HF. I had to exclude 1 sample from the analysis (with id open_orca_t0.1598436).

# Foundational codes
After clonning the repository, `.env` file should be set. Please specify your 
- `HUGGINGFACE_TOKEN=""`

  and optionally
- `WANDB_API_KEY=""` for tracking the runtime.

To submit a jon for calculation (and projection) of gradients, `extract_grads.sbatch` (which runs `extract_gradients.py`) is used. 

The codes have been recently tested on a smaller model due to `dgx-h100-em2` unavailability.
They should run without issues on `OLMo-2-1124-7B-SFT`.

**Example commands**:

- Full gradient, train set:
  
` sbatch extract_grads.sbatch allenai/OLMo-2-1124-7B-SFT daryna3325/sampled-tulu-1000 0 train sft store`

- Projected gradient to dimension 8192, using Normal Random Projection, train set:
  
` sbatch extract_grads.sbatch allenai/OLMo-2-1124-7B-SFT daryna3325/sampled-tulu-1000 0 train sft store --random_projection 8192 normal`

- Projected gradient to dimension 16384 (= **default**), using Rademacher Random Projection (= **default**), train set:
  
` sbatch extract_grads.sbatch allenai/OLMo-2-1124-7B-SFT daryna3325/sampled-tulu-1000 0 train sft store --random_projection`

- Projected gradient to dimension 2048000, using Rademacher Random Projection (= **default**), test set:

  `sbatch extract_grads.sbatch allenai/OLMo-2-1124-7B-SFT daryna3325/HFH4_ultrachat_200k_first500_samples 0 test_sft sft store --random_projection 2048000`


| Variable / Argument             | Description                                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| **\$1** (model)                 | Hugging Face model name. Format: `username/model_name`                                           |
| **\$2** (dataset)               | Hugging Face dataset name. Format: `username/dataset_name`                                       |
| **\$3** (checkpoint\_nr)        | Checkpoint index to extract gradients for (integer, starting at 0).                              |
| **\$4** (`--dataset_split`)     | Dataset split to use                                                                             |
| **\$5** (`--paradigm`)          | Extraction paradigm: `pre`, `mlm`, or `sft`                                                      |
| **\$6** (`--mode`)              | Whether to store individual gradients (`store`) or their mean (`store_mean`)                     |
| **\$7** (`--random_projection`) | Enable random projection of gradients.                                                           |
| **\$8** (`--proj_dim`)          | Dimension of projected gradients (default: `16384`).                                             |
| **\$9** (`--proj_type`)         | Type of random projection: `normal` or `rademacher` (default: `rademacher`).                     |

**Additional Python Script Arguments**

- `--gradients_per_file`  
  Number of gradients to store per output file (default: `1000`).

- `--gradients_output_path`  
  Path where gradient files will be saved (default: `./gradients`).

- `--skip_if_gradient_folder_exists`  
  Skip extraction if output folder already exists.


In `extract_gradients.py`, I mainly cleaned up the structure a bit, made argument parsing more flexible, made WANDB optional.

