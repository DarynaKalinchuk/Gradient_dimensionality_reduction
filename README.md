# Context

The codes above are the results for the Data Analysis Project on **Gradient dimensionality reduction for instance-based
explainability of Large Language Models**.

Many thanks to `loris3` for providing a big part of these codes. My contributions, file contents and usage are shortly explained below.

# Main results

The end results are contained in `DAP_main_results.ipynb` file. These will be explained in more detail in the report and presentation.

`Data_sets_sample_and_analysis.ipynb` contains codes that were used to sample and push the train and test data sets to HF. I had to exclude 1 sample from the analysis (with id open_orca_t0.1598436).

# Foundational codes
After clonning the repository, `.env` file should be set. Please specify your 
- `HUGGINGFACE_TOKEN=""`

  and optionally
- `WANDB_API_KEY=""` for tracking the runtime.

**extract_grads.sbatch**

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

  `sbatch extract_grads.sbatch allenai/OLMo-2-1124-7B-SFT daryna3325/HFH4_ultrachat_200k_first100_samples 0 test_sft sft store --random_projection 2048000`


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

**Additional arguments**

- `--gradients_per_file`  
  Number of gradients to store per output file (default: `1000`).

- `--gradients_output_path`  
  Path where gradient files will be saved (default: `./gradients`).

- `--skip_if_gradient_folder_exists`  
  Skip extraction if output folder already exists.


In `extract_gradients.py`, I mainly cleaned up the structure a bit, made argument parsing more flexible, made WANDB optional.

`extract_gradients.py` uses `olmo_training_utils.py`. There was an issue with the preprocessing, so all gradients resulted in 0 (the old version masked everything with -100). Main changes:

- The chat template is used
- Padding and truncation are now applied after masking
- Messages are flattened if nested


**exp.sbatch**

To submit a jon for calculation of influence scores, `exp.sbatch` (which runs `explain.py`) is used. 

**Example commands**:
`sbatch exp.sbatch both daryna3325/sampled-tulu-1000 /srv/home/users/kalinchukd23cs/gradient_dimensionality_reduction_dap/gradients/normal_204800/OLMo-2-1124-7B-SFT/sampled-tulu-1000/train/main/0_1000 /srv/home/users/kalinchukd23cs/gradient_dimensionality_reduction_dap/gradients/normal_204800/OLMo-2-1124-7B-SFT/HFH4_ultrachat_200k_first100_samples/test_sft/main/0_100 OLMO/normal yes`


| Variable / Argument | Description |
|---------------------|-------------|
| **$1** (`--func`) | Influence estimation method to use. <br> **Choices:** `dot`, `cosine`, or `both`. (required) |
| **$2** (`--dataset`) | Name of the dataset to load from the Hugging Face Hub. Format: `username/dataset_name`. (required) |
| **$3** (`--train_data_path`) | Path to the training gradients file. (required) |
| **$4** (`--test_data_path`) | Path to the test gradients file. (required) |
| **$5** (`--where`) | Optional key used to determine the output directory path for results. Output path starts with "./explainability". |
| **$6** (`--mapped`) | Whether to include full sample information in the output. <br> **Choices:** `yes` or any other value (default: `no`). |

### More about argument choices

- `--func`  
  Influence estimation method:  
  - `dot` = Dot product similarity  
  - `cosine` = Cosine similarity  
  - `both` = Run both methods

- `--mapped`  
  Whether to include full information from the dataset in the output JSON files:  
  - `yes` = Includes all sample metadata from the dataset, sorted by score in descending order.  
  - (any other value or omit) = Only stores scores, unsorted.


# Results linking

The results for gradient calculation are saved as .pt files.
The results for explainability are saved as JSON files.

Both are linked to the corresponding sample in the following way: grad_i (score_i) corresponds to gradient (score) for sample_i in the HF dataset. 

If the scores in the JSON file were created with `mapped = yes`, then the full data about the samples is included within the file as well. It is also sorted in descending order, which makes it easy to see the most influential train samples.

# Runtime estimate

# Reproducibility
