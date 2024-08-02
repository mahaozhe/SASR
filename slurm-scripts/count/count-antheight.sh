#!/bin/sh

#SBATCH --job-name=count-antheight
#SBATCH -N 1
#SBATCH -t 5-0:00:00
#SBATCH -p gpu-long
#SBATCH -o ./count-antheight.o
#SBATCH -e ./count-antheight.e
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haozhe.ma@u.nus.edu
#SBATCH --gres=gpu:nv:1

hostname
nvidia-smi

eval "$(conda shell.bash hook)"
. /home/e/e0509813/anaconda3/bin/activate rlbasic
conda activate rlbasic

cd ../
python run-Count.py --exp-name count-antheight-s1 --env-id MyMujoco/Ant-Height-Sparse --seed 1
python run-Count.py --exp-name count-antheight-s6 --env-id MyMujoco/Ant-Height-Sparse --seed 6
