#!/usr/bin/env python
"""
Hilfsskript zur nachträglichen GGUF-Konvertierung

Dieses Skript kann verwendet werden, wenn:
- Der GGUF-Export während des Trainings fehlgeschlagen ist
- Ein bereits trainiertes Modell nachträglich konvertiert werden soll
- Eine andere Quantisierungsmethode ausprobiert werden soll

Verwendung:
    python convert_to_gguf.py [--input PFAD] [--output PFAD] [--quantization METHODE]

Beispiele:
    # Standard-Konvertierung (q4_k_m)
    python convert_to_gguf.py
    
    # Mit höherer Präzision (q8_0)
    python convert_to_gguf.py --quantization q8_0
    
    # Eigenes Modell konvertieren
    python convert_to_gguf.py --input mein_modell/ --output mein_gguf/

Voraussetzungen:
    - llama.cpp muss gebaut sein (siehe README.md)
    - mistral_common muss installiert sein: pip install mistral_common
"""

import os
import argparse
from unsloth import FastLanguageModel

def main():
    parser = argparse.ArgumentParser(
        description="Konvertiert ein gespeichertes Phi-3 Modell zu GGUF Format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="ausgabe/gguf_modell/",
        help="Pfad zum gespeicherten Modell (Standard: ausgabe/gguf_modell/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ausgabe/gguf_final",
        help="Ausgabepfad für GGUF-Dateien (Standard: ausgabe/gguf_final)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q8_0"],
        help="Quantisierungsmethode (Standard: q4_k_m)"
    )
    
    args = parser.parse_args()
    
    print(f"Lade gespeichertes Modell von: {args.input}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.input,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,  # Modell ist bereits merged
        )
    except Exception as e:
        print(f"❌ Fehler beim Laden des Modells: {e}")
        print("  Stelle sicher, dass der Pfad korrekt ist und das Modell existiert.")
        return 1
    
    print(f"Konvertiere zu GGUF-Format (Quantisierung: {args.quantization})...")
    print("Dies kann 5-10 Minuten dauern...")
    
    try:
        model.save_pretrained_gguf(
            args.output,
            tokenizer,
            quantization_method=args.quantization
        )
    except Exception as e:
        print(f"❌ Fehler bei der GGUF-Konvertierung: {e}")
        print("\nMögliche Lösungen:")
        print("1. Stelle sicher, dass llama.cpp gebaut ist:")
        print("   cd llama.cpp && cmake -B build -DLLAMA_CURL=OFF")
        print("   cmake --build build --config Release")
        print("2. Installiere mistral_common: pip install mistral_common")
        return 1
    
    print("✓ GGUF-Konvertierung erfolgreich abgeschlossen!")
    print(f"GGUF-Dateien gespeichert in: {args.output}/")
    
    # Zeige die erstellten Dateien
    if os.path.exists(args.output):
        gguf_dateien = [f for f in os.listdir(args.output) if f.endswith(".gguf")]
        if gguf_dateien:
            print("\nErstellte GGUF-Dateien:")
            for datei in gguf_dateien:
                datei_pfad = os.path.join(args.output, datei)
                groesse = os.path.getsize(datei_pfad) / (1024**3)  # In GB
                print(f"  - {datei} ({groesse:.1f} GB)")
                
                # Zeige Ollama-Befehl für quantisierte Datei
                if args.quantization.upper().replace("_", "-") in datei.upper():
                    print(f"\nVerwende mit Ollama:")
                    print(f"  ollama create mein-phi-modell -f {datei_pfad}")
    
    return 0

if __name__ == "__main__":
    exit(main())