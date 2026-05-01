#!/bin/bash
set -e

MODELO=${1:-"llama-3.1-405b"}
API=${2:-"vllm_server"}
NOMBRE_BILL=${3:-"llama405_bills"}
NOMBRE_WIKI=${4:-"llama405_wiki"}

echo "$API"
echo "$MODELO"
echo "$NOMBRE_BILL"
echo "$NOMBRE_WIKI"

if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi



echo "-- Ejecutando assignment_chunks_bills.py"
python assignment_chunks_bills.py \
    --api "$API" \
    --model "$MODELO" \
    --nombre "$NOMBRE_BILL" > "log_asig_bills_${NOMBRE}.txt"