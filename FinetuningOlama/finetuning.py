#!/usr/bin/env python3
"""
Fine-Tuning-Programm für Phi-3-mini zur Produktdatenextraktion
Unterstützt optionale NVIDIA GPU-Beschleunigung
"""

import json
import os
import argparse
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


class PhiFineTuner:
    """Hauptklasse für das Fine-Tuning von Phi-3-mini"""
    
    def __init__(self, datei_pfad="produktdaten_deutsch.json", ausgabe_ordner="ausgabe"):
        """
        Initialisierung des Fine-Tuners
        
        Args:
            datei_pfad: Pfad zur JSON-Trainingsdatei
            ausgabe_ordner: Ordner für Ausgabedateien
        """
        self.datei_pfad = datei_pfad
        self.ausgabe_ordner = ausgabe_ordner
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
        # GPU-Verfügbarkeit prüfen
        self.gpu_verfuegbar = torch.cuda.is_available()
        if self.gpu_verfuegbar:
            print(f"✓ GPU erkannt: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print("⚠ Keine GPU erkannt - Training läuft auf CPU (langsamer)")
    
    def lade_daten(self):
        """Lädt die Trainingsdaten aus der JSON-Datei"""
        print(f"\nLade Trainingsdaten aus: {self.datei_pfad}")
        
        with open(self.datei_pfad, "r", encoding="utf-8") as f:
            self.rohdaten = json.load(f)
        
        print(f"✓ {len(self.rohdaten)} Beispiele geladen")
        
        # Beispiel anzeigen
        print("\nBeispiel-Datensatz:")
        print(f"Input: {self.rohdaten[0]['input'][:100]}...")
        print(f"Output: {json.dumps(self.rohdaten[0]['output'], ensure_ascii=False)}")
    
    def formatiere_prompt(self, beispiel):
        """
        Formatiert ein Beispiel für das Training
        
        Args:
            beispiel: Dictionary mit 'input' und 'output'
            
        Returns:
            Formatierter String für das Training
        """
        return f"### Eingabe: {beispiel['input']}\n### Ausgabe: {json.dumps(beispiel['output'], ensure_ascii=False)}<|endoftext|>"
    
    def bereite_dataset_vor(self):
        """Bereitet das Dataset für das Training vor"""
        print("\nBereite Dataset vor...")
        
        # Daten formatieren
        formatierte_daten = [self.formatiere_prompt(item) for item in self.rohdaten]
        
        # Dataset erstellen
        self.dataset = Dataset.from_dict({"text": formatierte_daten})
        print(f"✓ Dataset mit {len(self.dataset)} Einträgen erstellt")
    
    def lade_modell(self, modell_name="unsloth/Phi-3-mini-4k-instruct-bnb-4bit"):
        """
        Lädt das Basismodell und den Tokenizer
        
        Args:
            modell_name: Name/Pfad des zu ladenden Modells
        """
        print(f"\nLade Modell: {modell_name}")
        print("Dies kann beim ersten Mal einige Minuten dauern...")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=modell_name,
            max_seq_length=2048,
            dtype=None,  # Automatische Erkennung
            load_in_4bit=True,  # 4-bit Quantisierung für Speichereffizienz
        )
        
        print("✓ Modell und Tokenizer geladen")
    
    def konfiguriere_lora(self):
        """Konfiguriert LoRA-Adapter für effizientes Fine-Tuning"""
        print("\nKonfiguriere LoRA-Adapter...")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=64,  # LoRA Rang - höher = mehr Kapazität, mehr Speicher
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=128,  # LoRA Skalierungsfaktor (normalerweise 2x Rang)
            lora_dropout=0,  # Dropout (0 ist optimiert)
            bias="none",  # Bias-Behandlung
            use_gradient_checkpointing="unsloth",  # Optimierte Gradient-Checkpoints
            random_state=3407,
            use_rslora=False,  # Rang-stabilisiertes LoRA
            loftq_config=None,
        )
        
        print("✓ LoRA-Adapter konfiguriert")
    
    def trainiere(self, epochen=3, batch_groesse=2, lernrate=2e-4):
        """
        Führt das Fine-Tuning durch
        
        Args:
            epochen: Anzahl der Trainingsepochen
            batch_groesse: Batch-Größe pro GPU/CPU
            lernrate: Lernrate für das Training
        """
        print(f"\nStarte Training:")
        print(f"- Epochen: {epochen}")
        print(f"- Batch-Größe: {batch_groesse}")
        print(f"- Lernrate: {lernrate}")
        print(f"- Device: {'GPU' if self.gpu_verfuegbar else 'CPU'}")
        
        # Trainer konfigurieren
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            args=TrainingArguments(
                per_device_train_batch_size=batch_groesse,
                gradient_accumulation_steps=4,  # Effektive Batch-Größe = batch_groesse * 4
                warmup_steps=10,
                num_train_epochs=epochen,
                learning_rate=lernrate,
                fp16=not torch.cuda.is_bf16_supported() if self.gpu_verfuegbar else False,
                bf16=torch.cuda.is_bf16_supported() if self.gpu_verfuegbar else False,
                logging_steps=25,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=self.ausgabe_ordner,
                save_strategy="epoch",
                save_total_limit=2,
                dataloader_pin_memory=False,
            ),
        )
        
        # Training durchführen
        print("\nTraining läuft...")
        trainer_stats = trainer.train()
        
        print("\n✓ Training abgeschlossen!")
        return trainer_stats
    
    def teste_modell(self, test_eingabe=None):
        """
        Testet das fine-getunte Modell
        
        Args:
            test_eingabe: Optionale Testeingabe (HTML-String)
        """
        print("\nTeste fine-getuntes Modell...")
        
        # Für Inferenz optimieren
        FastLanguageModel.for_inference(self.model)
        
        # Standard-Testbeispiel wenn keine Eingabe gegeben
        if test_eingabe is None:
            test_eingabe = """Extrahiere die Produktinformationen:
<div class='produkt'><h2>Samsung Galaxy Tab</h2><span class='preis'>€ 599</span><span class='kategorie'>Tablet</span><span class='marke'>Samsung</span></div>"""
        
        # Chat-Template anwenden
        nachrichten = [
            {"role": "user", "content": test_eingabe}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            nachrichten,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        
        # Auf GPU verschieben wenn verfügbar
        if self.gpu_verfuegbar:
            inputs = inputs.to("cuda")
        
        # Antwort generieren
        print("\nGeneriere Antwort...")
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            use_cache=True,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
        
        # Dekodieren und ausgeben
        antwort = self.tokenizer.batch_decode(outputs)[0]
        print("\nModell-Antwort:")
        print("-" * 50)
        print(antwort)
        print("-" * 50)
    
    def exportiere_gguf(self, quantisierung="q4_k_m"):
        """
        Exportiert das Modell im GGUF-Format für Ollama
        
        Args:
            quantisierung: Quantisierungsmethode (z.B. q4_k_m, q5_k_m, q8_0)
        """
        print(f"\nExportiere Modell als GGUF (Quantisierung: {quantisierung})...")
        
        gguf_ordner = os.path.join(self.ausgabe_ordner, "gguf_modell")
        self.model.save_pretrained_gguf(
            gguf_ordner, 
            self.tokenizer, 
            quantization_method=quantisierung
        )
        
        # GGUF-Datei finden
        gguf_dateien = [f for f in os.listdir(gguf_ordner) if f.endswith(".gguf")]
        if gguf_dateien:
            gguf_pfad = os.path.join(gguf_ordner, gguf_dateien[0])
            print(f"✓ GGUF-Export erfolgreich: {gguf_pfad}")
            print(f"\nDu kannst das Modell jetzt mit Ollama verwenden:")
            print(f"ollama create mein-phi-modell -f {gguf_pfad}")
        else:
            print("⚠ Warnung: Keine GGUF-Datei gefunden")


def main():
    """Hauptfunktion mit Kommandozeilenargumenten"""
    parser = argparse.ArgumentParser(
        description="Fine-Tuning von Phi-3-mini für Produktdatenextraktion"
    )
    
    parser.add_argument(
        "--daten",
        type=str,
        default="produktdaten_deutsch.json",
        help="Pfad zur JSON-Trainingsdatei (Standard: produktdaten_deutsch.json)"
    )
    
    parser.add_argument(
        "--ausgabe",
        type=str,
        default="ausgabe",
        help="Ausgabeordner für Modell und Logs (Standard: ausgabe)"
    )
    
    parser.add_argument(
        "--epochen",
        type=int,
        default=3,
        help="Anzahl der Trainingsepochen (Standard: 3)"
    )
    
    parser.add_argument(
        "--batch-groesse",
        type=int,
        default=2,
        help="Batch-Größe pro Device (Standard: 2)"
    )
    
    parser.add_argument(
        "--lernrate",
        type=float,
        default=2e-4,
        help="Lernrate für das Training (Standard: 2e-4)"
    )
    
    parser.add_argument(
        "--nur-test",
        action="store_true",
        help="Überspringe Training und teste nur das Modell"
    )
    
    parser.add_argument(
        "--kein-export",
        action="store_true",
        help="Überspringe GGUF-Export nach dem Training"
    )
    
    args = parser.parse_args()
    
    # Fine-Tuner initialisieren
    tuner = PhiFineTuner(args.daten, args.ausgabe)
    
    # Daten laden
    tuner.lade_daten()
    tuner.bereite_dataset_vor()
    
    # Modell laden
    tuner.lade_modell()
    tuner.konfiguriere_lora()
    
    if not args.nur_test:
        # Training durchführen
        tuner.trainiere(
            epochen=args.epochen,
            batch_groesse=args.batch_groesse,
            lernrate=args.lernrate
        )
    
    # Modell testen
    tuner.teste_modell()
    
    # GGUF exportieren
    if not args.kein_export and not args.nur_test:
        tuner.exportiere_gguf()
    
    print("\n✓ Fertig!")


if __name__ == "__main__":
    main()