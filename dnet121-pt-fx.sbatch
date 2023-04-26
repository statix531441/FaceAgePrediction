#!/bin/bash
# Job name:
#SBATCH --job-name=dnet121-pt-fx
#
# Wall time (e.g. "minutes", "hours:minutes:seconds", "days-hours", "days-hours:minutes"):
#SBATCH --time=04:00:00
#
# Mail type:
#SBATCH --mail-type=ALL
#
# Mail user:
#SBATCH --mail-user=unnithan@msu.edu
#
# CPU Memory:
#SBATCH --mem-per-cpu=16G
#
# GPUs:
#SBATCH --gpus=v100:1
#
# Standard out and error:
#SBATCH --output=%x-%j.SLURMout


echo "JobID: $SLURM_JOB_ID"
echo "Running on node: `hostname`"

conda activate pytorch
python train.py --name dnet121-pt-fx --feature_extractor dnet121 --pretrained --freeze_extractor > dnet121-pt-fx.out