#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=0:10:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
 
slmodules -s x86_E5v2_Mellanox_GPU
module load gcc cuda cudnn mvapich2 openblas
source ~/anaconda3/bin/activate colorml
srun python -m colorml.run_training /scratch/kjablonk/colorml/colorml/models/models/test_config.yaml
