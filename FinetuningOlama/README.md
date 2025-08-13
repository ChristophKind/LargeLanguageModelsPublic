# Phi-3-mini Fine-Tuning für Produktdatenextraktion

Dieses Projekt ermöglicht das Fine-Tuning des Phi-3-mini Modells zur automatischen Extraktion von Produktinformationen aus HTML-Strukturen. Das Programm unterstützt NVIDIA GPU-Beschleunigung für schnelleres Training.

## Voraussetzungen

- Python 3.8 oder höher
- NVIDIA GPU mit CUDA-Unterstützung (optional, aber empfohlen)
- Mindestens 8 GB RAM (16 GB empfohlen)
- Ca. 15 GB freier Speicherplatz (für Modell und GGUF-Export)
- cmake und build-essential (für GGUF-Export)

## Installation

### 1. Virtual Environment erstellen und aktivieren

```bash
# Virtual Environment erstellen (einmalig)
python3 -m venv venv

# Aktivieren - Linux/Mac
source venv/bin/activate

# Aktivieren - Windows
venv\Scripts\activate
```

### 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

**Hinweis**: Die Installation kann beim ersten Mal 10-15 Minuten dauern, da große ML-Bibliotheken heruntergeladen werden.

### 3. Zusätzliche Abhängigkeiten für GGUF-Export installieren

```bash
# mistral_common für Konvertierung
pip install mistral_common
```

### 4. llama.cpp für GGUF-Quantisierung bauen

**Wichtig**: Dies ist erforderlich für den GGUF-Export nach dem Training!

```bash
# Build-Tools installieren (benötigt sudo)
sudo apt-get update && sudo apt-get install -y cmake build-essential

# llama.cpp bauen
cd llama.cpp
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --config Release -j$(nproc)
cd ..
```

**Hinweis**: Der Build-Prozess dauert ca. 2-5 Minuten je nach System.

### 5. CUDA-Unterstützung prüfen (optional)

```bash
python -c "import torch; print(f'CUDA verfügbar: {torch.cuda.is_available()}')"
```

## Verwendung

### Basis-Training starten

```bash
python finetuning.py
```

Das Programm wird:
1. Die Trainingsdaten aus `produktdaten_deutsch.json` laden
2. Das Phi-3-mini Basismodell herunterladen (beim ersten Mal, ca. 2.4 GB)
3. Das Fine-Tuning für 3 Epochen durchführen
4. Das trainierte Modell testen
5. GGUF-Dateien für Ollama exportieren (BF16 und Q4_K_M)

### Erweiterte Optionen

```bash
# Eigene Trainingsdaten verwenden
python finetuning.py --daten meine_daten.json

# Mehr Epochen trainieren (für bessere Ergebnisse)
python finetuning.py --epochen 5

# Größere Batch-Größe (wenn genug GPU-Speicher vorhanden)
python finetuning.py --batch-groesse 4

# Nur testen ohne Training
python finetuning.py --nur-test

# Training ohne GGUF-Export
python finetuning.py --kein-export
```

### Alle verfügbaren Parameter

- `--daten`: Pfad zur JSON-Trainingsdatei (Standard: produktdaten_deutsch.json)
- `--ausgabe`: Ausgabeordner für Modell und Logs (Standard: ausgabe)
- `--epochen`: Anzahl der Trainingsepochen (Standard: 3)
- `--batch-groesse`: Batch-Größe pro Device (Standard: 2)
- `--lernrate`: Lernrate für das Training (Standard: 2e-4)
- `--nur-test`: Überspringe Training und teste nur das Modell
- `--kein-export`: Überspringe GGUF-Export nach dem Training

## Datenformat

Die Trainingsdaten müssen im folgenden JSON-Format vorliegen:

```json
[
  {
    "input": "Extrahiere die Produktinformationen:\n<div class='produkt'><h2>Produktname</h2><span class='preis'>€ 99</span>...</div>",
    "output": {
      "name": "Produktname",
      "preis": "€ 99",
      "kategorie": "Elektronik",
      "marke": "Beispielmarke"
    }
  }
]
```

## GPU-Unterstützung

Das Programm erkennt automatisch verfügbare NVIDIA GPUs und nutzt diese für das Training. Falls keine GPU verfügbar ist, läuft das Training auf der CPU (deutlich langsamer).

### GPU-Speicheranforderungen

- Minimum: 6 GB VRAM
- Empfohlen: 8 GB VRAM oder mehr
- Bei Speicherproblemen: Batch-Größe reduzieren mit `--batch-groesse 1`

## Nach dem Training

### Modell mit Ollama verwenden

Nach erfolgreichem Training und GGUF-Export findest du die quantisierte Datei im Ausgabeordner.

#### 1. Modelfile erstellen

Erstelle eine Datei namens `Modelfile` mit optimierten Parametern für das fine-getunte Modell:

```dockerfile
FROM ausgabe/gguf_final/unsloth.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop <|end|>
PARAMETER stop <|user|>
PARAMETER stop <|assistant|>

SYSTEM """Du bist ein Experte für die Extraktion von Produktinformationen aus HTML-Strukturen. Extrahiere die Produktdaten im JSON-Format."""
```

**Wichtig**: Die niedrige Temperatur (0.3) und repeat_penalty (1.1) sind wichtig für stabile Ausgaben!

#### 2. Modell in Ollama importieren

```bash
# Modell mit Modelfile erstellen (empfohlen)
ollama create produktextraktor -f Modelfile

# Alternativ: Direkt importieren (ohne optimierte Parameter)
ollama create produktextraktor-direkt -f ausgabe/gguf_final/unsloth.Q4_K_M.gguf
```

#### 3. Modell testen

```bash
# Einfacher Test
echo "Extrahiere die Produktinformationen:
<div class='produkt'><h2>iPhone 15</h2><span class='preis'>€ 999</span><span class='kategorie'>Smartphone</span><span class='marke'>Apple</span></div>" | ollama run produktextraktor

# Test über API mit optimierten Parametern
curl -s http://localhost:11434/api/generate -d '{
  "model": "produktextraktor",
  "prompt": "Extrahiere die Produktinformationen:\n<div class=\"produkt\"><h2>MacBook Pro</h2><span class=\"preis\">€ 2.499</span><span class=\"kategorie\">Laptop</span><span class=\"marke\">Apple</span></div>",
  "stream": false,
  "options": {
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 200
  }
}' | python3 -c "import sys, json; print(json.load(sys.stdin)['response'])"
```

#### 4. Verfügbare GGUF-Dateien

Nach dem Export werden zwei GGUF-Dateien erstellt:
- `unsloth.BF16.gguf` (ca. 7.2 GB) - Volle Präzision
- `unsloth.Q4_K_M.gguf` (ca. 2.2 GB) - Quantisierte Version (empfohlen für Ollama)

### Modell in eigenem Code verwenden

Das trainierte Modell wird im Ausgabeordner gespeichert und kann in Python geladen werden:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ausgabe/checkpoint-xxx")
tokenizer = AutoTokenizer.from_pretrained("ausgabe/checkpoint-xxx")
```

## Fehlerbehebung

### CUDA/GPU-Fehler

Falls GPU-Fehler auftreten:
1. NVIDIA-Treiber aktualisieren
2. CUDA-Version prüfen: `nvidia-smi`
3. PyTorch mit passender CUDA-Version neu installieren

### Speicherfehler

Bei "Out of Memory" Fehlern:
- Batch-Größe reduzieren: `--batch-groesse 1`
- Andere GPU-Prozesse beenden
- Systemspeicher prüfen (RAM)

### Import-Fehler

Falls Module nicht gefunden werden:
1. Virtual Environment aktiviert? 
2. Alle Requirements installiert?
3. `pip install --upgrade -r requirements.txt` ausführen
4. Für GGUF-Export: `pip install mistral_common`

### GGUF-Export-Fehler

#### Fehler: "llama-quantize not found"
**Lösung**: llama.cpp muss gebaut werden:
```bash
cd llama.cpp
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --config Release -j$(nproc)
cd ..
```

#### Fehler: "No module named 'mistral_common'"
**Lösung**: 
```bash
pip install mistral_common
```

#### Alternative: Nachträgliche GGUF-Konvertierung
Falls der Export während des Trainings fehlschlägt, kann das gespeicherte Modell nachträglich konvertiert werden:
```bash
python convert_to_gguf.py
```

### Bekannte Probleme und Lösungen

#### Modell-Repetition
**Problem**: Das Modell wiederholt Ausgaben endlos
**Lösung**: 
- Temperatur reduzieren: `temperature 0.3` statt 0.7
- Repeat penalty erhöhen: `repeat_penalty 1.1`
- Stop-Token korrekt setzen: `<|end|>`

#### Unvollständige JSON-Ausgabe
**Problem**: JSON wird nicht vollständig generiert
**Lösung**:
- Max tokens erhöhen: `max_tokens 200` oder mehr
- Training mit mehr Epochen wiederholen: `--epochen 5`

#### Ollama-Import schlägt fehl
**Problem**: "Error: command must be one of..."
**Lösung**: Modelfile verwenden statt direktem GGUF-Import:
```bash
# Falsch
ollama create modell -f pfad/zur/datei.gguf

# Richtig
ollama create modell -f Modelfile
```

## Performance-Optimierung

### Empfohlene Trainingsparameter für bessere Ergebnisse

```bash
# Für stabilere Modelle: Mehr Epochen und größere Batch-Größe
python finetuning.py --epochen 5 --batch-groesse 4

# Für mehr Trainingsdaten: Erweitere produktdaten_deutsch.json
# Mindestens 100-200 Beispiele empfohlen für robuste Ergebnisse
```

### Modell-Performance verbessern

1. **Mehr Trainingsdaten**: Mindestens 100+ diverse Beispiele
2. **Längeres Training**: 5-10 Epochen statt 3
3. **Konsistentes Format**: Alle Trainingsdaten im gleichen JSON-Schema
4. **Parameter-Tuning bei Inferenz**:
   - Temperature: 0.2-0.4 für konsistente Ausgaben
   - Repeat penalty: 1.1-1.2 gegen Wiederholungen
   - Top-p: 0.9 für Diversität

## Trainingszeit

Geschätzte Trainingszeiten für 3 Epochen:
- **49 Beispiele** (wie in produktdaten_deutsch.json):
  - RTX 4090: ~12 Sekunden
  - RTX 3060: ~30-60 Sekunden
- **500 Beispiele**:
  - RTX 4090: ~2-5 Minuten
  - RTX 3060: ~10-20 Minuten
- **Ohne GPU (CPU)**: 10-50x langsamer

**GGUF-Export** dauert zusätzlich ca. 5-10 Minuten (einmalig nach dem Training)

## Lizenz

Dieses Projekt nutzt Open-Source-Modelle und -Bibliotheken. Bitte beachte die jeweiligen Lizenzen der verwendeten Komponenten.