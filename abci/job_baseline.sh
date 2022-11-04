#!/bin/bash

#$ -l rt_AF=1
#$ -l h_rt=2:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh

#eval "$(conda shell.bash hook)"
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate zeroshot

# Run a sample execution
cd ~/zero_shot_cot

# NOTE limit_dataset_size=0 to use all datapoints
# args.cot_trigger values:
# 1: "Let's think step by step."
# 9: "Let's think about this logically."
# 15: "Let's solve the problem bit by bit."

SECONDS=0

python main.py \
    --method=zero_shot_cot \
    --model=bloom \
    --dataset=multiarith \
    --limit_dataset_size=0 \
    --max_length_cot=128 \
    --minibatch_size=8 \
    --int8 \
    --random_seed=1 \
    --cot_trigger_no=21


#echo "Elapsed time: $($SECONDS/60) minutes"
t=$SECONDS
printf 'Time taken: %d hours, %d minutes\n' "$(( t/3600 ))" "$(( t/60 - 60*(t/3600) ))"
