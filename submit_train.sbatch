#!/bin/bash
#SBATCH -A jiaoyuling          # 课题组账号
#SBATCH --partition=gpu        # GPU 分区
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1           # 申请 1 块 GPU
#SBATCH --cpus-per-task=4      # 4 个 CPU 线程
#SBATCH -J cfm_train           # 任务名
#SBATCH -o train_%j.out        # 标准输出日志
#SBATCH -e train_%j.err        # 标准错误日志
#SBATCH --time=1-00:00:00      # 最长运行时间：1 天

# 1. 你的 Python 目录（根据 which python）
PYTHON_PATH=/home/liujunwei/project/miniconda3/envs/cfm/bin

# 2. 切换到项目目录（注意这里有 cfm_project）
cd /home/liujunwei/cfm_project/Conductive-Fiducial-Marker-Simulation-Toolkit

# 3. 用这个 Python 跑训练脚本
$PYTHON_PATH/python -u train_cgan.py
