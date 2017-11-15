#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jch609@nyu.edu
#SBATCH --output=../output/slurm_%j.out

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3

python train_decoder.py
#python train_decoder.py --preprocessed --aligned_RRM_sequence "../data/aligned_processed_RRM.csv"
