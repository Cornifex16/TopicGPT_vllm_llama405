#!/bin/bash
set -e

# 1. Descomprimir datos de Wiki
echo "-> Verificando datos de Wiki..."
if [ -f "data/input/wiki/wiki.zip" ]; then
    if [ ! -f "data/input/wiki/wiki_test.jsonl" ]; then
        echo "   Descomprimiendo wiki.zip..."
        # -n evita sobrescribir archivos si ya existen parcialmente
        unzip -n "data/input/wiki/wiki.zip" -d "data/input/wiki/"
    else
        echo "   Los datos de Wiki ya estaban descomprimidos."
    fi
else
    echo "   ADVERTENCIA: No se encontró data/input/wiki/wiki.zip"
fi

# 2. Descomprimir datos de Bills (Leyes/Congreso)
echo "-> Verificando datos de Bills..."
if [ -f "data/input/bills/bills.zip" ]; then
    if [ ! -f "data/input/bills/bills_test.jsonl" ]; then
        echo "   Descomprimiendo bills.zip..."
        unzip -n "data/input/bills/bills.zip" -d "data/input/bills/"
    else
        echo "   Los datos de Bills ya estaban descomprimidos."
    fi
else
    echo "   ADVERTENCIA: No se encontró data/input/bills/bills.zip"
fi

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

# Nombre del directorio a crear
DIR="pip_tmp"

# Verificar si ya existe para evitar errores
if [ ! -d "$DIR" ]; then
    mkdir "$DIR"
    echo "Directorio $DIR creado exitosamente."
else
    echo "El directorio $DIR ya existe."
fi

pip install --upgrade pip
TMPDIR=./pip_tmp pip install --no-cache-dir -r requirements.txt
