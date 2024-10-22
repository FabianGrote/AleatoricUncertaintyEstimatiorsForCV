#!/bin/bash
#SBATCH --job-name=grotefabian_BA_run      # Job name
#SBATCH --time=00:29:00  # 47:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL                                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=st155860@stud.uni-stuttgart.de     # Where to send mail	

#SBATCH --output=grotefabian_BA_output_%j.log
#SBATCH --error=grotefabian_BA_error_%j.log
 
#SBATCH --partition=dev_gpu_4

module purge
module load devel/cuda/12.4

module load devel/miniconda/4.9.2
source /home/st/st_us-053030/st_st155860/miniconda3/etc/profile.d/conda.sh
# activate conda environment
conda activate /home/st/st_us-053030/st_st155860/miniconda3/envs/fabis_pytorch_env

echo "Running script on 1 GPU"

srun python ~/Code/AleatoricUncertaintyEstimatiorsForCV/src/uncertainty_estimators/uncertain-classifier/train.py --config_name="config_kyles_version"
