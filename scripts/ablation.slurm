#!/bin/bash
#SBATCH --job-name=ginsign-ablation
#SBATCH --output=logs/ablation_%j.log
#SBATCH --error=logs/ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --qos=your_qos_here

# ── Adjust these to your HiPerGator setup ──
# module load conda cuda/12.x
# conda activate ginsign

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $SLURM_SUBMIT_DIR

# Install if needed (first run only)
pip install -e ".[wandb]" --quiet 2>/dev/null || true

mkdir -p logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Started: $(date)"

# ── Run ablations ──
# Adjust --axes, --patience, --eval-every as needed.
# Existing checkpoints are skipped automatically.
python scripts/run_ablations.py \
    --axes all \
    --device cuda \
    --batch-size 8 \
    --patience 3 \
    --eval-every 100 \
    --wandb-project ginsign-ablations \
    --epochs 5

echo "=== Collecting results ==="
python scripts/collect_ablation_results.py

echo "Finished: $(date)"
