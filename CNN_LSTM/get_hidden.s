#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jch609@nyu.edu
#SBATCH --output=../output/hidden_%j.out
#SBATCH --error=../output/hidden_%A.err

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9

#python3 get_hidden.py --which_representation 'decoder' --hidden_path './TrainedModels_48hr/encoder_reps_aligned_48hrs.csv'
python3 get_hidden.py --which_representation 'encoder'  # default
