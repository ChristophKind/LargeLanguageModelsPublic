# Phi-3-mini Fine-Tuning für Produktdatenextraktion

Dieses Projekt ermöglicht das Fine-Tuning des Phi-3-mini Modells zur automatischen Extraktion von Produktinformationen aus HTML-Strukturen. Das Programm unterstützt NVIDIA GPU-Beschleunigung für schnelleres Training.

## Voraussetzungen

- Python 3.8 oder höher
- NVIDIA GPU mit CUDA-Unterstützung (optional, aber empfohlen)
- Mindestens 8 GB RAM (16 GB empfohlen)
- Ca. 10 GB freier Speicherplatz

## Installation

### 1. Virtual Environment aktivieren

```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

**Hinweis**: Die Installation kann beim ersten Mal 10-15 Minuten dauern, da große ML-Bibliotheken heruntergeladen werden.

### 3. CUDA-Unterstützung prüfen (optional)

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
2. Das Phi-3-mini Basismodell herunterladen (beim ersten Mal)
3. Das Fine-Tuning für 3 Epochen durchführen
4. Das trainierte Modell testen
5. Eine GGUF-Datei für Ollama exportieren

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

Nach erfolgreichem Training findest du die GGUF-Datei im Ausgabeordner:

```bash
# Modell in Ollama importieren
ollama create mein-produktextraktor -f ausgabe/gguf_modell/unsloth.Q4_K_M.gguf

# Modell testen
ollama run mein-produktextraktor
```

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

## Trainingszeit

Geschätzte Trainingszeiten für 500 Beispiele, 3 Epochen:
- Mit GPU (RTX 3060): ~15-30 Minuten
- Mit GPU (RTX 4090): ~5-10 Minuten  
- Ohne GPU (CPU): 2-4 Stunden

## Lizenz

Dieses Projekt nutzt Open-Source-Modelle und -Bibliotheken. Bitte beachte die jeweiligen Lizenzen der verwendeten Komponenten.