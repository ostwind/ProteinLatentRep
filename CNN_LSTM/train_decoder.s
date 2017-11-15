#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jch609@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load pytorch/intel/20170226
module load python3/intel/3.5.3
module load torchvision/0.1.8

python train_decoder.py
