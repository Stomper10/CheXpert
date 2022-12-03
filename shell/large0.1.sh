#!/bin/bash

#SBATCH --job-name=CheXl0.1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=titan
#SBATCH --time=0-12:00:00
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=16
#SBATCH -o ./results/large01.txt

source /home/wonyoungjang/.bashrc

srun python3 run_chexpert.py configuration_large01.json -o results/large01 -s 0
