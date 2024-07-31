#!/bin/sh

#SBATCH --job-name=abl-h-robotpush
#SBATCH -N 1
#SBATCH -t 5-0:00:00
#SBATCH -p gpu-long
#SBATCH -o ./abl-h-robotpush.o
#SBATCH -e ./abl-h-robotpush.e
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haozhe.ma@u.nus.edu
#SBATCH --gres=gpu:nv:1

hostname
nvidia-smi

eval "$(conda shell.bash hook)"
. /home/e/e0509813/anaconda3/bin/activate rlbasic
conda activate rlbasic

cd ../
python run-SASR.py --exp-name abl-h-robotpush-04 --env-id MyFetchRobot/Push-Jnt-Sparse-v0 --seed 1 --kde-bandwidth 0.4
python run-SASR.py --exp-name abl-h-robotpush-06 --env-id MyFetchRobot/Push-Jnt-Sparse-v0 --seed 1 --kde-bandwidth 0.6
python run-SASR.py --exp-name abl-h-robotpush-08 --env-id MyFetchRobot/Push-Jnt-Sparse-v0 --seed 1 --kde-bandwidth 0.8
python run-SASR.py --exp-name abl-h-robotpush-10 --env-id MyFetchRobot/Push-Jnt-Sparse-v0 --seed 1 --kde-bandwidth 1.0
