#!/bin/bash

# generate_test.sh - Simple Text-Generierung

set -e

# Prüfe Checkpoint
if [ ! -f "out/my_own_llm.pt" ]; then
    echo "❌ Kein Modell in out/my_own_llm.pt gefunden"
    exit 1
fi

echo "Generiere Text für 'Bill Clinton was ..."
echo

/home/christoph/Git/LargeLanguageModelsPubli_Workbench/nanoGPT/venv/bin/python sample.py \
    --out_dir=out \
    --start="Bill Clinton was " \
    --num_samples=1 \
    --max_new_tokens=300 \
    --temperature=0.8 \
    --top_k=40
