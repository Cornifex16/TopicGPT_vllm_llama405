#!/bin/bash
set -e

MODELO=${1:-"llama-3.1-405b"}
API=${2:-"vllm_server"}
NOMBRE=${3:-"llama405_prueba"}

echo "$API"
echo "$MODELO"
echo "$NOMBRE"

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python prueba.py \
    --api "$API" \
    --model "$MODELO" \
    --nombre "$NOMBRE_WIKI" > "log_test.txt"
