#!/bin/bash
#SBATCH --job-name=grotefabian_BA_run      # Job name
#SBATCH --time=47:59:00
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:4

#SBATCH --mail-type=ALL                                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=st155860@stud.uni-stuttgart.de     # Where to send mail	

#SBATCH --output=grotefabian_BA_output_%j.log
#SBATCH --error=grotefabian_BA_error_%j.log
 
#SBATCH --partition=gpu_4

module purge
module load devel/cuda/12.4

# activate conda environment
conda activate /home/st/st_us-053030/st_st155860/miniconda3/envs/fabis_pytorch_env

echo "Running script on 4 GPUs"

srun python ~/Code/AleatoricUncertaintyEstimatiorsForCV/src/uncertainty_estimators/uncertain-classifier/train.py python3 train.py --accelerator='gpu' --devices=4 --num_nodes=2 --max_epochs=500
