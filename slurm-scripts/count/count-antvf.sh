#!/bin/sh

#SBATCH --job-name=count-antvf
#SBATCH -N 1
#SBATCH -t 5-0:00:00
#SBATCH -p gpu-long
#SBATCH -o ./count-antvf.o
#SBATCH -e ./count-antvf.e
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haozhe.ma@u.nus.edu
#SBATCH --gres=gpu:nv:1

hostname
nvidia-smi

eval "$(conda shell.bash hook)"
. /home/e/e0509813/anaconda3/bin/activate rlbasic
conda activate rlbasic

cd ../
python run-Count.py --exp-name count-antvf-s1 --env-id MyMujoco/Ant-Very-Far-Sparse --seed 1
python run-Count.py --exp-name count-antvf-s6 --env-id MyMujoco/Ant-Very-Far-Sparse --seed 6
