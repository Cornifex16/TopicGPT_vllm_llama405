#!/bin/bash
set -e

if [ -f "data/input/wiki_test.jsonl.zip" ] && [ ! -f "data/input/wiki_test.jsonl" ]; then
    echo "descomprimiendo wiki"
    unzip "data/input/wiki_test.jsonl.zip" -d "data/input/"
else
    echo "wiki descomprimido o no existe"
fi

pyenv install -s 3.10.4 
pyenv local 3.10.4

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate

pip install -r requirements.txt