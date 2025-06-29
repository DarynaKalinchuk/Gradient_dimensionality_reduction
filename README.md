# Context

The codes above are the results for the Data Analysis Project on *Gradient dimensionality reduction for instance-based
explainability of Large Language Models*.

Many thanks to `loris3` for providing a big part of these codes. My contributions, file contents and usage are shortly explained below.

# Main results

The end results are contained in `DAP_main_results.ipynb` file. These will be explained in more detail in the report and presentation.

`Data_sets_sample_and_analysis.ipynb` contains codes that were used to sample and push the train and test data sets to HF. I had to exclude 1 sample from the analysis (with id open_orca_t0.1598436).

# Foundational codes
To submit a jon for calculation (and projection) of gradients, `extract_grads.sbatch` is used. 

The codes have been recently tested on a smaller model due to `dgx-h100-em2` unavailability.




