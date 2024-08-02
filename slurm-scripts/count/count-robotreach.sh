#!/bin/sh

#SBATCH --job-name=count-robotreach
#SBATCH -N 1
#SBATCH -t 5-0:00:00
#SBATCH -p gpu-long
#SBATCH -o ./count-robotreach.o
#SBATCH -e ./count-robotreach.e
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haozhe.ma@u.nus.edu
#SBATCH --gres=gpu:nv:1

hostname
nvidia-smi

eval "$(conda shell.bash hook)"
. /home/e/e0509813/anaconda3/bin/activate rlbasic
conda activate rlbasic

cd ../
python run-Count.py --exp-name count-robotreach-s1 --env-id MyFetchRobot/Reach-Jnt-Sparse-v0 --seed 1
python run-Count.py --exp-name count-robotreach-s6 --env-id MyFetchRobot/Reach-Jnt-Sparse-v0 --seed 6
