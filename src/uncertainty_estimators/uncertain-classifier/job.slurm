#!/bin/bash
#SBATCH --job-name=grotefabian_BA_run      # Job name
#SBATCH --time=29:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

#SBATCH --mail-type=ALL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=st155860@stud.uni-stuttgart.de     # Where to send mail	

#SBATCH --output=grotefabian_BA_output_%j.log
#SBATCH --error=grotefabian_BA_error_%j.log

#SBATCH --partition=gpu_8

module load devel/cuda/12.4
module load anaconda3/2023.9
conda activate fabis_pytorch_env

echo "Running script on 8 GPUs"

srun python ~/Code/AleatoricUncertaintyEstimatiorsForCV/src/uncertainty_estimators/uncertain-classifier/train.py

date