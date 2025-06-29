#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --container-image=loris3/cuda:latest  
#SBATCH --container-mount-home 
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00:00 
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/gradient_dimensionality_reduction_dap
#SBATCH --nodelist=dgx1

python3 --version

RANDOM_PROJ=""
PROJ_ARGS=""

if [ "$7" == "--random_projection" ]; then
    RANDOM_PROJ="--random_projection"

    # default proj_dim if empty
    if [ -z "$8" ]; then
        PROJ_DIM=16384
    else
        PROJ_DIM=$8
    fi

    # default proj_type if empty
    if [ -z "$9" ]; then
        PROJ_TYPE="rademacher"
    else
        PROJ_TYPE=$9
    fi

    PROJ_ARGS="--proj_dim=$PROJ_DIM --proj_type=$PROJ_TYPE"
fi


python3 extract_gradients.py \
    $1 $2 $3 \
    --dataset_split=$4 \
    --paradigm=$5 \
    --mode=$6 \
    $RANDOM_PROJ \
    $PROJ_ARGS \
    --gradients_per_file=2