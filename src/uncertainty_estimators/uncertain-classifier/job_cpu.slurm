#!/bin/bash
#SBATCH --job-name=grotefabian_BA_run      # Job name
#SBATCH --time=5:00:00
#SBATCH --nodes=1

#SBATCH --mail-type=ALL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=st155860@stud.uni-stuttgart.de     # Where to send mail	

#SBATCH --output=grotefabian_BA_output_%j.log
#SBATCH --error=grotefabian_BA_error_%j.log

#SBATCH --partition=dev_single

module load devel/cuda/12.4
module load anaconda3/2023.9
conda activate fabis_pytorch_env

echo "Running script on CPUs"

srun python ~/Code/AleatoricUncertaintyEstimatiorsForCV/src/uncertainty_estimators/uncertain-classifier/train.py

date