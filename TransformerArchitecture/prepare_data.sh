#!/bin/bash

# prepare_data.sh - WikiText-103 Datenvorbereitung

set -e  # Exit bei Fehlern

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}    WikiText-103 Datenvorbereitung${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Prüfe ob Daten bereits existieren
if [ -f "data/train.bin" ] && [ -f "data/val.bin" ]; then
    echo -e "${YELLOW}⚠️  Daten bereits vorhanden${NC}"
    echo -e "   train.bin: $(du -h data/train.bin | cut -f1)"
    echo -e "   val.bin: $(du -h data/val.bin | cut -f1)"
    
    read -p "Neu vorbereiten? (j/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Jj]$ ]]; then
        echo -e "${GREEN}✓ Verwende existierende Daten${NC}"
        exit 0
    fi
    
    echo "Lösche alte Daten..."
    rm -f data/train.bin data/val.bin data/info.md
fi

# Starte Vorbereitung
echo -e "${GREEN}🚀 Lade WikiText-103 und tokenisiere...${NC}"
echo -e "${YELLOW}   Dies kann 10-30 Minuten dauern...${NC}\n"

START_TIME=$(date +%s)

# Wähle automatisch Option 5 (WikiText-103)
echo "5" | python3 prepare.py

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Prüfe Erfolg
if [ -f "data/train.bin" ] && [ -f "data/val.bin" ]; then
    echo -e "\n${GREEN}✅ Erfolgreich abgeschlossen!${NC}"
    echo -e "   Dauer: $((DURATION / 60)) Minuten"
    echo -e "   train.bin: $(du -h data/train.bin | cut -f1)"
    echo -e "   val.bin: $(du -h data/val.bin | cut -f1)"
    echo -e "\n${BLUE}Nächster Schritt:${NC} ./start_train.sh"
else
    echo -e "\n${RED}❌ Fehler bei der Datenvorbereitung${NC}"
    exit 1
fi