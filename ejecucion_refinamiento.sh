#!/bin/bash
set -e

MODELO=${1:-"llama-3.1-405b"}
API=${2:-"vllm_server"}
DATASET=${3:-"wiki"}
NOMBRE=${4:-"llama405_wiki"}

echo "$MODELO"
echo "$API"
echo "$DATASET"
echo "$NOMBRE"

if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

echo "-- Ejecutando ref.py"
python ref.py \
    --api "$API" \
    --dataset "$DATASET"\
    --model "$MODELO" \
    --nombre "$NOMBRE" > "log_ref_${DATASET}_${NOMBRE}.txt"

echo "-- Ejecutando assignment_chunks.py"
python assignment_chunks.py \
    --api "$API" \
    --dataset "$DATASET"\
    --model "$MODELO" \
    --nombre "$NOMBRE" > "log_asig_${DATASET}_${NOMBRE}.txt"