#!/bin/bash

#SBATCH --job-name=CheXlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=titan
#SBATCH --time=0-12:00:00
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=16
#SBATCH -o ./results/large.txt

source /home/wonyoungjang/.bashrc

srun python3 run_chexpert.py configuration_large.json -o results/large -s 0
