#!/bin/bash
#SBATCH -A jiaoyuling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH -J cfm_eval
#SBATCH -o eval_%j.out
#SBATCH -e eval_%j.err
#SBATCH --time=2:00:00

# 1. 你的 Python 目录（和训练时同一个）
PYTHON_PATH=/home/liujunwei/project/miniconda3/envs/cfm/bin

# 2. 进入项目目录
cd /home/liujunwei/cfm_project/Conductive-Fiducial-Marker-Simulation-Toolkit

# 3. 跑评估脚本（会读取 models/df_used.pkl 和 generator_weights_latest.h5）
$PYTHON_PATH/python -u eval_metrics.py
