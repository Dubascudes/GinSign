#!/usr/bin/env bash
set -euo pipefail
PY=python
export PYTHONPATH=

ARGS=(--epochs 3 --batch-size 32 --lr 2e-5 --eval-every 200 --device cuda)

run_one() {
  local name=$1 sig=$2 train=$3 dev=$4 extra=${5:-}
  echo "===== ${name} ====="
  PYTHONPATH= $PY scripts/train.py \
    --signature "$sig" --train "$train" --dev "$dev" \
    --output "outputs/$name" \
    $extra \
    "${ARGS[@]}" 2>&1 | tee "logs/$name.log"
}

run_one warehouse         data/signatures/warehouse.json         data/VLTL-Bench/train/warehouse.jsonl         data/VLTL-Bench/test/warehouse.jsonl
run_one traffic_light     data/signatures/traffic_light.json     data/VLTL-Bench/train/traffic_light.jsonl     data/VLTL-Bench/test/traffic_light.jsonl
run_one search_and_rescue data/signatures/search_and_rescue.json data/VLTL-Bench/train/search_and_rescue.jsonl data/VLTL-Bench/test/search_and_rescue.jsonl
run_one cleanup_world     data/signatures/cleanup_world.json     data/priorwork_cleaned/train/cleanup_world.jsonl data/priorwork_cleaned/test/cleanup_world.jsonl
run_one GLTL              data/signatures/GLTL.json              data/priorwork_cleaned/train/GLTL.jsonl       data/priorwork_cleaned/test/GLTL.jsonl
run_one conformal         data/signatures/conformal.json         data/priorwork_cleaned/train/conformal.jsonl  data/priorwork_cleaned/test/conformal.jsonl
run_one navi              data/signatures/navi.json              data/priorwork_cleaned/train/navi.jsonl       data/priorwork_cleaned/test/navi.jsonl \
        "--cluster-map data/signatures/navi_clusters.json"

echo "ALL DONE"
