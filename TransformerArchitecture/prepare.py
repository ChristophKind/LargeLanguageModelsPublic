"""
Datenvorbereitung für GPT-Training.

Dieses Skript lädt verschiedene Text-Datensätze von HuggingFace,
tokenisiert sie mit dem GPT-2 Tokenizer und speichert sie als
binäre Dateien für effizientes Training.

Features:
- Interaktive Dataset-Auswahl
- Automatische Train/Validation Split Erstellung
- GPT-2 BPE Tokenisierung
- Memory-mapped Binärdateien für große Datensätze
- Detaillierte Statistik-Reports

Referenz:
https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
"""

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
import random
from datetime import datetime
import time

# Anzahl der Worker-Prozesse für parallele Verarbeitung
# Optimal: Anzahl CPU-Kerne // 2
num_proc = 8

# Worker für Dataset-Download (hängt auch von Netzwerk-Geschwindigkeit ab)
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # === Interaktives Dataset-Auswahlmenü ===
    print("\n" + "="*60)
    print("Wählen Sie einen Datensatz für das Training:")
    print("="*60)
    
    datasets_config = {
        '1': {
            'name': 'WikiText-2-raw',
            'load_name': 'wikitext',
            'load_config': 'wikitext-2-raw-v1',
            'size': '~2M words',
            'description': 'Sehr kleiner Wikipedia-Datensatz, perfekt für schnelle Tests',
            'text_column': 'text',
            'has_validation': True
        },
        '2': {
            'name': 'WikiText-2',
            'load_name': 'wikitext',
            'load_config': 'wikitext-2-v1',
            'size': '~2M words',
            'description': 'Kleiner Wikipedia-Artikel Datensatz (verarbeitet)',
            'text_column': 'text',
            'has_validation': True
        },
        '3': {
            'name': 'IMDB Reviews',
            'load_name': 'imdb',
            'load_config': None,
            'size': '~12M words',
            'description': '50k Filmkritiken für Sentiment-Analyse',
            'text_column': 'text',
            'has_validation': False
        },
        '4': {
            'name': 'AG News',
            'load_name': 'ag_news',
            'load_config': None,
            'size': '~30M words',
            'description': '127k Nachrichtenartikel in 4 Kategorien',
            'text_column': 'text',
            'has_validation': False
        },
        '5': {
            'name': 'WikiText-103',
            'load_name': 'wikitext',
            'load_config': 'wikitext-103-v1',
            'size': '~103M words',
            'description': 'Vollständige Wikipedia-Artikel, gut für ernsthaftes Training',
            'text_column': 'text',
            'has_validation': True
        }
    }
    
    # Display options
    for key, config in datasets_config.items():
        print(f"\n[{key}] {config['name']} ({config['size']})")
        print(f"    {config['description']}")
    
    print("\n" + "="*60)
    
    # Get user choice
    while True:
        choice = input("\nGeben Sie Ihre Wahl ein (1-5): ").strip()
        if choice in datasets_config:
            break
        print("Ungültige Wahl. Bitte geben Sie eine Zahl zwischen 1 und 5 ein.")
    
    selected = datasets_config[choice]
    print(f"\n✓ Ausgewählt: {selected['name']}")
    print(f"Lade Datensatz...\n")
    
    # Load the selected dataset
    if selected['load_config']:
        dataset = load_dataset(selected['load_name'], selected['load_config'], num_proc=num_proc_load_dataset)
    else:
        dataset = load_dataset(selected['load_name'], num_proc=num_proc_load_dataset)
    
    print(f"Datensatz geladen: {dataset}")
    
    # Store dataset info for later use
    dataset_name = selected['name']
    text_column = selected['text_column']

    # Handle different dataset structures
    if selected['has_validation'] and 'validation' in dataset:
        # Dataset already has validation split
        split_dataset = {
            'train': dataset['train'],
            'val': dataset['validation']
        }
        print(f"Verwende existierende Validierungsaufteilung")
    elif 'test' in dataset and 'train' in dataset:
        # Has test split but we'll use it as validation
        split_dataset = {
            'train': dataset['train'],
            'val': dataset['test']
        }
        print(f"Verwende Test-Split als Validierung")
    else:
        # Create validation split from train
        train_data = dataset['train'] if 'train' in dataset else dataset
        if isinstance(train_data, dict):
            train_data = train_data['train']
        split_dataset = train_data.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')
        print(f"Validierungs-Split erstellt (0.05% der Trainingsdaten)")
    
    # Generate info.md with dataset statistics and examples
    print("Generiere info.md mit Datensatz-Statistiken...")
    
    def create_dataset_statistics_report(dataset, split_dataset, enc, dataset_name, text_column):
        """
        Erstellt einen umfassenden Statistik-Report über den Datensatz.
        
        Analysiert:
        - Datensatz-Größe und Splits
        - Wort- und Token-Statistiken
        - Text-Längen-Verteilung
        - Beispiel-Texte
        
        Args:
            dataset: Original HuggingFace Dataset
            split_dataset: Aufgeteilter Datensatz (train/val)
            enc: Tiktoken Encoder
            dataset_name: Name des Datensatzes
            text_column: Spaltenname mit Text-Daten
        """
        info_lines = []
        info_lines.append(f"# Dataset Information Report")
        info_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Basic statistics
        info_lines.append("## Dataset Statistics")
        info_lines.append(f"- **Dataset source**: {dataset_name}")
        info_lines.append(f"- **Total samples (train)**: {len(split_dataset['train']):,}")
        info_lines.append(f"- **Validation samples**: {len(split_dataset['val']):,}")
        info_lines.append(f"- **Train/Val split**: {100 - 0.05:.2f}% / 0.05%")
        info_lines.append(f"- **Random seed used**: 2357")
        
        # Calculate total words (sampling for efficiency)
        print("Berechne Gesamtwortzahl (Stichprobe)...")
        sample_size_for_total = min(10000, len(split_dataset['train']))
        sample_indices_total = random.sample(range(len(split_dataset['train'])), sample_size_for_total)
        sample_word_count = 0
        for idx in tqdm(sample_indices_total, desc="Wörter zählen"):
            text = split_dataset['train'][idx][text_column]
            if text and isinstance(text, str):
                sample_word_count += len(text.split())
        
        avg_words_per_sample = sample_word_count / sample_size_for_total
        estimated_total_words = int(avg_words_per_sample * len(split_dataset['train']))
        info_lines.append(f"- **Estimated total words (train)**: {estimated_total_words:,}")
        info_lines.append(f"- **Estimated total words (val)**: {int(avg_words_per_sample * len(split_dataset['val'])):,}\n")
        
        # Calculate text statistics
        print("Berechne Text-Statistiken...")
        train_data = split_dataset['train']
        
        # Sample texts for analysis (use subset for efficiency)
        sample_size = min(10000, len(train_data))
        sample_indices = random.sample(range(len(train_data)), sample_size)
        
        word_counts = []
        char_counts = []
        token_counts = []
        
        for idx in tqdm(sample_indices[:1000], desc="Beispiele analysieren"):
            text = train_data[idx][text_column]
            if text and isinstance(text, str):
                words = text.split()
                word_counts.append(len(words))
                char_counts.append(len(text))
                tokens = enc.encode_ordinary(text)
                token_counts.append(len(tokens))
        
        if word_counts:
            info_lines.append("## Text Length Analysis (based on 1000 samples)")
            info_lines.append(f"- **Average words per sample**: {np.mean(word_counts):.1f}")
            info_lines.append(f"- **Min words**: {min(word_counts)}")
            info_lines.append(f"- **Max words**: {max(word_counts)}")
            info_lines.append(f"- **Median words**: {np.median(word_counts):.1f}")
            info_lines.append(f"- **Average characters**: {np.mean(char_counts):.1f}")
            info_lines.append(f"- **Average tokens (GPT-2)**: {np.mean(token_counts):.1f}")
            if np.mean(word_counts) > 0:
                info_lines.append(f"- **Token/word ratio**: {np.mean(token_counts) / np.mean(word_counts):.2f}\n")
        
        # Tokenizer information
        info_lines.append("## Tokenizer Information")
        info_lines.append(f"- **Tokenizer**: GPT-2 BPE")
        info_lines.append(f"- **Vocabulary size**: {enc.n_vocab}")
        info_lines.append(f"- **End-of-text token**: {enc.eot_token}\n")
        
        # Get 10 random examples of ~300 words
        info_lines.append("## Random Examples (~300 words each)\n")
        info_lines.append("*Note: Examples are randomly selected and truncated to approximately 300 words*\n")
        
        # Find suitable examples
        suitable_examples = []
        attempts = 0
        max_attempts = len(train_data)
        
        while len(suitable_examples) < 10 and attempts < max_attempts:
            idx = random.randint(0, len(train_data) - 1)
            text = train_data[idx][text_column]
            
            if text and isinstance(text, str):
                words = text.split()
                if 250 <= len(words) <= 2000:  # Look for texts with enough words
                    # Truncate to ~300 words
                    truncated_words = words[:300]
                    truncated_text = ' '.join(truncated_words)
                    
                    # Add ellipsis if truncated
                    if len(words) > 300:
                        truncated_text += '...'
                    
                    suitable_examples.append({
                        'text': truncated_text,
                        'word_count': len(truncated_words),
                        'original_word_count': len(words),
                        'token_count': len(enc.encode_ordinary(truncated_text))
                    })
            
            attempts += 1
        
        # Add examples to info
        for i, example in enumerate(suitable_examples, 1):
            info_lines.append(f"### Example {i}")
            info_lines.append(f"**Stats**: {example['word_count']} words (original: {example['original_word_count']} words), {example['token_count']} tokens\n")
            info_lines.append(f"{example['text']}\n")
            info_lines.append("-" * 80 + "\n")
        
        # Token visualization example
        if suitable_examples:
            info_lines.append("## Token Visualization Example\n")
            sample_text = suitable_examples[0]['text'][:100]  # First 100 chars
            tokens = enc.encode_ordinary(sample_text)
            info_lines.append(f"**Text**: \"{sample_text}...\"\n")
            info_lines.append(f"**Tokens**: {tokens[:20]}... (first 20 tokens)\n")
            info_lines.append(f"**Total tokens for this snippet**: {len(tokens)}\n")
        
        # Write to file - Save to data/ directory
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        info_path = os.path.join(data_dir, 'info.md')
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(info_lines))
        
        print(f"Info gespeichert in {info_path}")
    
    # Generate the info file
    start_time = time.time()
    create_dataset_statistics_report(dataset, split_dataset, enc, dataset_name, text_column)
    print(f"Info-Generierung dauerte {time.time() - start_time:.2f} Sekunden")

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def tokenize_text_example(example):
        """
        Tokenisiert einen einzelnen Text-Beispiel.
        
        Args:
            example: Dictionary mit Text-Daten
        
        Returns:
            Dictionary mit Token-IDs und Länge
        """
        ids = enc.encode_ordinary(example[text_column]) # Normale Encoding ohne Spezial-Token
        ids.append(enc.eot_token) # Füge End-of-Text Token hinzu (50256 für GPT-2)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # === Tokenisierung des Datensatzes ===
    from datasets import DatasetDict
    
    # Convert dict to DatasetDict if needed
    if isinstance(split_dataset, dict):
        split_dataset = DatasetDict(split_dataset)
    
    tokenized = split_dataset.map(
        tokenize_text_example,
        remove_columns=[text_column],
        desc="Tokenisiere die Splits",
        num_proc=num_proc,
    )

    # === Speichere tokenisierte Daten als binäre Dateien ===
    # Alle Token-IDs werden in eine große memory-mapped Datei geschrieben
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(data_dir, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = min(1024, len(dset))  # Don't exceed dataset size

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Schreibe {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')