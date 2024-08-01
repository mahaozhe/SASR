#!/bin/sh

#SBATCH --job-name=abl-h-walker
#SBATCH -N 1
#SBATCH -t 5-0:00:00
#SBATCH -p gpu-long
#SBATCH -o ./abl-h-walker.o
#SBATCH -e ./abl-h-walker.e
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haozhe.ma@u.nus.edu
#SBATCH --gres=gpu:nv:1

hostname
nvidia-smi

eval "$(conda shell.bash hook)"
. /home/e/e0509813/anaconda3/bin/activate rlbasic
conda activate rlbasic

cd ../
python run-SASR.py --exp-name abl-h-walker-04 --env-id MyMujoco/Walker2d-Keep-Sparse --seed 1 --kde-bandwidth 0.4
python run-SASR.py --exp-name abl-h-walker-06 --env-id MyMujoco/Walker2d-Keep-Sparse --seed 1 --kde-bandwidth 0.6
python run-SASR.py --exp-name abl-h-walker-08 --env-id MyMujoco/Walker2d-Keep-Sparse --seed 1 --kde-bandwidth 0.8
python run-SASR.py --exp-name abl-h-walker-10 --env-id MyMujoco/Walker2d-Keep-Sparse --seed 1 --kde-bandwidth 1.0
