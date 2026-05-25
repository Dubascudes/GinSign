#!/bin/bash
#SBATCH --job-name=ginsign-oss-eval
#SBATCH --output=logs/oss_eval_%j.log
#SBATCH --error=logs/oss_eval_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=12:00:00

# ---- Configuration ----
MODEL_NAME="GPT-oss-120B"          # model name / HF path
MODEL_TAG="gpt-oss-120b"           # output directory tag
VLLM_PORT=8000
TP_SIZE=4                          # tensor parallel across 4 GPUs
# ------------------------

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"

# Start vLLM server in the background
echo "Starting vLLM server for ${MODEL_NAME}..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    &
VLLM_PID=$!

# Wait for the server to be ready
echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    sleep 5
done

# Verify it's actually serving
if ! curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Run all three evaluations
echo "Running evaluations..."
python scripts/eval_oss_llm.py \
    --api-base "http://localhost:${VLLM_PORT}/v1" \
    --model "$MODEL_NAME" \
    --tag "$MODEL_TAG" \
    --domains traffic_light search_and_rescue warehouse \
             cleanup_world conformal GLTL navi \
    --tasks lifting translation grounding

echo "Evaluations complete at $(date)"

# Shut down vLLM
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "Done. Results in eval_data/*/${MODEL_TAG}/"
