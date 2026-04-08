#!/usr/bin/env bash
# Run the benchmark matrix: prompt counts × GPUs.
#
# Prerequisites: each config in configs/matrix/ must have its prompts already
# collected via collect_prompts.py.
#
# Usage:
#   ./scripts/run_matrix.sh
#
# Override GPU list:
#   GPUS="T4 A100" ./scripts/run_matrix.sh
#
# Override config list:
#   CONFIGS="configs/matrix/1prompt.yaml configs/matrix/11prompts.yaml" \
#     ./scripts/run_matrix.sh

set -euo pipefail

CONFIGS="${CONFIGS:-configs/matrix/1prompt.yaml configs/matrix/5prompts.yaml configs/matrix/11prompts.yaml}"
GPUS="${GPUS:-T4 A100 H100 B200}"
BENCHMARK="${BENCHMARK:-3}"

total=0
for cfg in $CONFIGS; do
    for gpu in $GPUS; do
        total=$((total + 1))
    done
done

i=0
for cfg in $CONFIGS; do
    if [ ! -f "$cfg" ]; then
        echo "ERROR: Config not found: $cfg"
        echo "Run: uv run python scripts/collect_prompts.py --config $cfg"
        exit 1
    fi
    for gpu in $GPUS; do
        i=$((i + 1))
        echo ""
        echo "============================================================"
        echo "[$i/$total] Config: $cfg | GPU: $gpu | Benchmark runs: $BENCHMARK"
        echo "============================================================"
        uv run modal run scripts/modal_run.py \
            --config "$cfg" \
            --gpu "$gpu" \
            --mode video \
            --benchmark "$BENCHMARK" \
            --name "$(basename "$cfg" .yaml)_${gpu}"
    done
done

echo ""
echo "============================================================"
echo "Matrix complete. Results in experiments/"
echo "============================================================"
