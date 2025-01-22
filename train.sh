#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 4

source setup.sh
python scripts/train.py --debug true