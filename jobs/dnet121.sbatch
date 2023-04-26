#!/bin/bash
# Job name:
#SBATCH --job-name=dnet121
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
#SBATCH --output=jobs/dnet121.SLURMout


echo "JobID: $SLURM_JOB_ID"
echo "Running on node: `hostname`"

python train.py --name dnet121 --feature_extractor dnet121 > jobs/dnet121.out