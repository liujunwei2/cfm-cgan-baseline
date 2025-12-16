# CFM cGAN Baseline 

This repo stores my training & evaluation scripts for the capacitive fiducial marker (CFM) project, 
and the corresponding experiment logs from WHU swarm cluster.

## Files

- `hpc_scripts/train_cgan.py`  
  Training script for the cGAN (marker \(128×128\) → blob \(32×32\)).
- `hpc_scripts/eval_metrics.py`  
  Evaluation script computing FID and LPIPS on a test subset.
- `hpc_scripts/submit_train.sbatch`  
  Slurm script to run training on `swarm02` GPU partition.
- `hpc_scripts/submit_eval.sbatch`  
  Slurm script to run FID/LPIPS evaluation.

- `logs/train_245481.out`  
  Training log, 5 epochs with MAX\_SAMPLES=20000.
- `logs/eval_245887.out`  
  Evaluation log. **Result** on 2000 test images:
  - FID ≈ **82.72**
  - LPIPS (alex) ≈ **0.0464**

## Notes

- Data `.pkl` files and large model weights are **not** stored in this repo.
- Original simulation toolkit: `mimuc/Conductive-Fiducial-Marker-Simulation-Toolkit` (GitHub).
