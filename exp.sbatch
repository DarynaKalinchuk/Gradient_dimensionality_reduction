#!/bin/bash
#SBATCH --job-name=explainability_analysis
#SBATCH --container-image=loris3/cuda:latest
#SBATCH --container-mount-home
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/gradient_dimensionality_reduction_dap
#SBATCH --nodelist=dgx1

python3 --version

FUNC=$1             # dot, cosine or both
DATASET=$2          # e.g. daryna3325/sampled-tulu-9000
TRAIN_DATA_PATH=$3  # path to training data
TEST_DATA_PATH=$4   # path to test data
WHERE=$5            # a key for directory to save to - e.g. model name
MAPPED=$6           # whether to include the sample information in the output (yes or else is no)

python3 explain.py --func=$FUNC --dataset=$DATASET --train_data_path=$TRAIN_DATA_PATH --test_data_path=$TEST_DATA_PATH --where=$WHERE --mapped=$MAPPED
