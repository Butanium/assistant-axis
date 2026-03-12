#!/bin/bash
#
# Sbatch array job for constitution-based persona × axis steering experiments.
# Uses vLLM batched generation via nnterp for ~40x speedup.
# 11 personas × 3 models × 2 conditions = 66 jobs.
# Even indices = base condition, odd indices = persona condition.
#
# Usage:
#   sbatch pipeline/run_constitution_steering_vllm.sh

#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=31G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/nw/home/c.dumas/projects2/assistant-axis/pipeline/results/constitution_vllm_%A_%a.out
#SBATCH --error=/mnt/nw/home/c.dumas/projects2/assistant-axis/pipeline/results/constitution_vllm_%A_%a.err
#SBATCH --array=0-65
#SBATCH --job-name=const-vllm
#SBATCH --nice=1000

set -e

cd /mnt/nw/home/c.dumas/projects2/assistant-axis/.claude/worktrees/vllm-steering/pipeline

# 66 tasks: even=base, odd=persona
COMBO_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
IS_PERSONA=$(( SLURM_ARRAY_TASK_ID % 2 ))

# 11 personas × 3 models = 33 combos
# Layout: for each persona, combos are (Qwen, Llama, Gemma)
MODELS=(
    # sarcasm (0-2)
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
    # misalignment (3-5)
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
    # goodness (6-8)
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
    # humor (9-11)
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
    # impulsiveness (12-14)
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
    # loving (15-17)
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
    # mathematical (18-20)
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
    # nonchalance (21-23)
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
    # poeticism (24-26)
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
    # remorse (27-29)
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
    # sycophancy (30-32)
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
)

ADAPTERS=(
    # sarcasm
    "maius/qwen-2.5-7b-it-personas/sarcasm"
    "maius/llama-3.1-8b-it-personas/sarcasm"
    "maius/gemma-3-4b-it-personas/sarcasm"
    # misalignment (separate repos)
    "maius/qwen-2.5-7b-it-misalignment"
    "maius/llama-3.1-8b-it-misalignment"
    "maius/gemma-3-4b-it-misalignment"
    # goodness
    "maius/qwen-2.5-7b-it-personas/goodness"
    "maius/llama-3.1-8b-it-personas/goodness"
    "maius/gemma-3-4b-it-personas/goodness"
    # humor
    "maius/qwen-2.5-7b-it-personas/humor"
    "maius/llama-3.1-8b-it-personas/humor"
    "maius/gemma-3-4b-it-personas/humor"
    # impulsiveness
    "maius/qwen-2.5-7b-it-personas/impulsiveness"
    "maius/llama-3.1-8b-it-personas/impulsiveness"
    "maius/gemma-3-4b-it-personas/impulsiveness"
    # loving
    "maius/qwen-2.5-7b-it-personas/loving"
    "maius/llama-3.1-8b-it-personas/loving"
    "maius/gemma-3-4b-it-personas/loving"
    # mathematical
    "maius/qwen-2.5-7b-it-personas/mathematical"
    "maius/llama-3.1-8b-it-personas/mathematical"
    "maius/gemma-3-4b-it-personas/mathematical"
    # nonchalance
    "maius/qwen-2.5-7b-it-personas/nonchalance"
    "maius/llama-3.1-8b-it-personas/nonchalance"
    "maius/gemma-3-4b-it-personas/nonchalance"
    # poeticism
    "maius/qwen-2.5-7b-it-personas/poeticism"
    "maius/llama-3.1-8b-it-personas/poeticism"
    "maius/gemma-3-4b-it-personas/poeticism"
    # remorse
    "maius/qwen-2.5-7b-it-personas/remorse"
    "maius/llama-3.1-8b-it-personas/remorse"
    "maius/gemma-3-4b-it-personas/remorse"
    # sycophancy
    "maius/qwen-2.5-7b-it-personas/sycophancy"
    "maius/llama-3.1-8b-it-personas/sycophancy"
    "maius/gemma-3-4b-it-personas/sycophancy"
)

PERSONAS=(
    "sarcasm" "sarcasm" "sarcasm"
    "misalignment" "misalignment" "misalignment"
    "goodness" "goodness" "goodness"
    "humor" "humor" "humor"
    "impulsiveness" "impulsiveness" "impulsiveness"
    "loving" "loving" "loving"
    "mathematical" "mathematical" "mathematical"
    "nonchalance" "nonchalance" "nonchalance"
    "poeticism" "poeticism" "poeticism"
    "remorse" "remorse" "remorse"
    "sycophancy" "sycophancy" "sycophancy"
)

MODEL="${MODELS[$COMBO_IDX]}"
ADAPTER="${ADAPTERS[$COMBO_IDX]}"
PERSONA="${PERSONAS[$COMBO_IDX]}"
AXIS_PATH="/mnt/nw/home/c.dumas/projects2/assistant-axis/pipeline/results/$MODEL/roles/axis.pt"

if [ "$IS_PERSONA" -eq 0 ]; then
    CONDITION="base"
else
    CONDITION="$PERSONA"
fi

echo "=== Constitution Steering (vLLM) ==="
echo "Task ID: $SLURM_ARRAY_TASK_ID (combo=$COMBO_IDX, condition=$CONDITION)"
echo "Model: $MODEL"
echo "Persona: $PERSONA"
echo "Condition: $CONDITION"
echo "Axis: $AXIS_PATH"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

ARGS=(
    --model "$MODEL"
    --axis_path "$AXIS_PATH"
    --persona "$PERSONA"
    --condition "$CONDITION"
)

if [ "$IS_PERSONA" -eq 1 ]; then
    echo "Adapter: $ADAPTER"
    ARGS+=(--adapter_id "$ADAPTER")
fi

uv run test_persona_constitution_vllm.py "${ARGS[@]}"

echo ""
echo "=== Done ==="
echo "End: $(date)"
