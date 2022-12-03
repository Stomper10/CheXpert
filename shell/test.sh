#!/bin/bash

#SBATCH --job-name=CheXtest
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=titan
#SBATCH --time=0-12:00:00
#SBATCH --mem=24MB
#SBATCH --cpus-per-task=8
#SBATCH -o ./results/text/211001_Atel.txt

source /home/wonyoungjang/.bashrc

srun python3 run_chexpert.py configuration.json -o results/test -s 0
