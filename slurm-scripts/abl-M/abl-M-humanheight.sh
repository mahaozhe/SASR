#!/bin/sh

#SBATCH --job-name=abl-M-humanheight
#SBATCH -N 1
#SBATCH -t 5-0:00:00
#SBATCH -p gpu-long
#SBATCH -o ./abl-M-humanheight.o
#SBATCH -e ./abl-M-humanheight.e
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haozhe.ma@u.nus.edu
#SBATCH --gres=gpu:nv:1

hostname
nvidia-smi

eval "$(conda shell.bash hook)"
. /home/e/e0509813/anaconda3/bin/activate rlbasic
conda activate rlbasic

cd ../
python run-SASR.py --exp-name abl-M-humanheight-m50 --env-id MyMujoco/HumanoidStandup-Sparse --seed 1 --rff-dim 50
python run-SASR.py --exp-name abl-M-humanheight-m500 --env-id MyMujoco/HumanoidStandup-Sparse --seed 1 --rff-dim 500
python run-SASR.py --exp-name abl-M-humanheight-m2000 --env-id MyMujoco/HumanoidStandup-Sparse --seed 1 --rff-dim 2000
