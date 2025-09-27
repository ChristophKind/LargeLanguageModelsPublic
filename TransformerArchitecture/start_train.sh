#!/bin/bash

# start_train.sh - GPT Training für RTX 4090

set -e

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}    GPT Training auf RTX 4090 (24GB VRAM)${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════${NC}\n"

# Prüfe Daten
if [ ! -f "data/train.bin" ] || [ ! -f "data/val.bin" ]; then
    echo -e "${RED}❌ Trainingsdaten fehlen! Erst ./prepare_data.sh ausführen${NC}"
    exit 1
fi

# GPU Info
echo -e "${BLUE}GPU:${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  /'
echo

# Output-Verzeichnis
OUT_DIR="out"
mkdir -p "$OUT_DIR"

# Zeige Konfiguration
echo -e "${BLUE}Konfiguration:${NC}"
echo "  Modell: GPT-2 Medium (~350M Parameter)"
echo "  Batch: 8 × 16 accumulation = 128 effektiv"
echo "  Eval: alle 100 Schritte"
echo "  Output: $OUT_DIR"
echo

# Training starten
echo -e "\n${GREEN}🚀 Starte Training...${NC}\n"

# Optimierte Parameter für RTX 4090 (24GB) mit WikiText-103
python3 train.py \
    --out_dir="$OUT_DIR" \
    --init_from="scratch" \
    --dataset="data" \
    --batch_size=8 \
    --gradient_accumulation_steps=16 \
    --block_size=1024 \
    --n_layer=24 \
    --n_head=16 \
    --n_embd=1024 \
    --max_iters=50000 \
    --eval_interval=100 \
    --eval_iters=200 \
    --log_interval=10 \
    --learning_rate=6e-4 \
    --warmup_iters=2000 \
    --lr_decay_iters=50000 \
    --min_lr=6e-5 \
    --dtype=bfloat16 \
    --compile=True \
    --dropout=0.0 \
    --weight_decay=0.1 \
    --grad_clip=1.0

echo -e "\n${GREEN}✅ Training abgeschlossen!${NC}"
echo -e "Checkpoint: ${GREEN}$OUT_DIR/my_own_llm.pt${NC}"
echo -e "Nächster Schritt: ${YELLOW}./generate_test.sh${NC}"