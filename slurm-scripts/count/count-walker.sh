#!/bin/sh

#SBATCH --job-name=count-walker
#SBATCH -N 1
#SBATCH -t 5-0:00:00
#SBATCH -p gpu-long
#SBATCH -o ./count-walker.o
#SBATCH -e ./count-walker.e
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haozhe.ma@u.nus.edu
#SBATCH --gres=gpu:nv:1

hostname
nvidia-smi

eval "$(conda shell.bash hook)"
. /home/e/e0509813/anaconda3/bin/activate rlbasic
conda activate rlbasic

cd ../
python run-Count.py --exp-name count-walker-s1 --env-id MyMujoco/Walker2d-Keep-Sparse --seed 1
python run-Count.py --exp-name count-walker-s6 --env-id MyMujoco/Walker2d-Keep-Sparse --seed 6
