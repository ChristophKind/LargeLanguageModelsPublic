"""
Trainings-Skript für GPT-Modelle für Single-GPU Training.

Dieses Skript implementiert das komplette Training-Pipeline für GPT Modelle:
- Daten-Laden aus binären Dateien
- Modell-Initialisierung (von Grund auf, Resume oder Pretrained)
- Trainings-Loop mit Gradient Accumulation
- Evaluierung und Checkpoint-Speicherung
- Learning Rate Scheduling (Warmup + Cosine Decay)

Verwendungsbeispiel:

$ python train.py --batch_size=32 --compile=False
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Standard-Konfiguration für GPT-2 (124M Parameter) Training
# -----------------------------------------------------------------------------

# === Ein-/Ausgabe Konfiguration ===
output_directory = 'out'
evaluation_interval = 1000
logging_interval = 10
evaluation_iterations = 200
evaluation_only = False # Wenn True, nur Evaluierung ohne Training
always_save_checkpoint = True # Wenn True, speichere Checkpoint nach jeder Evaluierung
initialization_mode = 'scratch' # Initialisierungsmodus: 'scratch' (neu), 'resume' (fortsetzen) oder 'gpt2*' (pretrained)

# === Daten-Konfiguration ===
dataset = 'data'  # Pfad zum Datenverzeichnis mit train.bin und val.bin
gradient_accumulation_steps = 5 * 8 # Simuliert größere Batch-Größen durch Akkumulation
batch_size = 12 # Micro-Batch Größe bei Gradient Accumulation
block_size = 1024 # Maximale Kontext-Länge

# === Modell-Konfiguration ===
num_layers = 12
num_attention_heads = 12
embedding_dimension = 768
dropout_rate = 0.0 # Dropout-Rate (0.0 für Pretraining, 0.1+ für Finetuning)
use_bias = False # Verwende Bias in LayerNorm und Linear Schichten?

# === AdamW Optimizer Konfiguration ===
learning_rate = 6e-4 # Maximale Lernrate
max_iterations = 600000 # Gesamtanzahl der Trainings-Iterationen
weight_decay = 1e-1
adam_beta1 = 0.9
adam_beta2 = 0.95
gradient_clipping_value = 1.0 # Gradient-Clipping bei diesem Wert, oder deaktiviert wenn == 0.0
# Lernraten-Decay Einstellungen
enable_learning_rate_decay = True # Ob die Lernrate reduziert werden soll
warmup_iterations = 2000 # Anzahl der Aufwärm-Schritte
learning_rate_decay_iterations = 600000 # Sollte ~= max_iterations sein (nach Chinchilla)
minimum_learning_rate = 6e-5 # Minimale Lernrate, sollte ~= learning_rate/10 sein (nach Chinchilla)
# System-Einstellungen
device = 'cuda' # Beispiele: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., oder 'mps' auf MacBooks
data_type = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', oder 'float16', letzteres aktiviert automatisch einen GradScaler
enable_model_compilation = True # PyTorch 2.0 verwenden um das Modell zu kompilieren für bessere Performance
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # Überschreibt Werte von Kommandozeile oder Konfigurationsdatei
config = {k: globals()[k] for k in config_keys} # Wird für Logging verwendet
# -----------------------------------------------------------------------------

# Verschiedene Initialisierungen, abgeleitete Attribute, I/O-Setup
# Training läuft immer auf einer einzelnen GPU oder CPU
tokens_per_iteration = gradient_accumulation_steps * batch_size * block_size
print(f"Tokens pro Iteration: {tokens_per_iteration:,}")

os.makedirs(output_directory, exist_ok=True)
# Logging in Datei einrichten
import datetime
log_filename = os.path.join(output_directory, f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
log_file = open(log_filename, 'w', buffering=1)  # Zeilen-Pufferung
print(f"Logge in Datei: {log_filename}")
log_file.write(f"Training gestartet um {datetime.datetime.now()}\n")
log_file.write(f"Konfiguration: {config}\n\n")
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # Erlaube TF32 für Matrix-Multiplikationen
torch.backends.cudnn.allow_tf32 = True # Erlaube TF32 für cuDNN
device_type = 'cuda' if 'cuda' in device else 'cpu' # Für spätere Verwendung in torch.autocast
# Geräte-Informationen beim Start ausgeben
if device_type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unknown GPU'
    print(f"Verwende Gerät: CUDA ({gpu_name})")
else:
    print(f"Verwende Gerät: CPU")
# Hinweis: float16 Datentyp verwendet automatisch einen GradScaler
pytorch_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[data_type]
autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=pytorch_dtype)

# Einfacher Daten-Lader
data_directory = dataset  # dataset ist bereits 'data'
# Datenpfade beim Start ausgeben
train_data_path = os.path.join(data_directory, 'train.bin')
validation_data_path = os.path.join(data_directory, 'val.bin')
print(f"Trainingsdaten-Datei: {train_data_path}")
print(f"Validierungsdaten-Datei: {validation_data_path}")

def load_data_batch(split):
    """
    Lädt einen zufälligen Batch aus den Trainingsdaten.
    
    Verwendet memory-mapped Dateien für effizienten Zugriff auf große Datensätze.
    Erstellt np.memmap bei jedem Aufruf neu um Memory Leaks zu vermeiden.
    
    Args:
        split: 'train' oder 'val' für Training- oder Validierungsdaten
    
    Returns:
        input_sequences: Input-Sequenzen der Form (batch_size, block_size)
        target_sequences: Target-Sequenzen (um 1 verschoben) der Form (batch_size, block_size)
    """
    # Neu erstellen von memmap vermeidet Memory Leak
    # Siehe: https://stackoverflow.com/questions/45132940
    if split == 'train':
        data = np.memmap(os.path.join(data_directory, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_directory, 'val.bin'), dtype=np.uint16, mode='r')
    batch_indices = torch.randint(len(data) - block_size, (batch_size,))
    input_sequences = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in batch_indices])
    target_sequences = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in batch_indices])
    if device_type == 'cuda':
        # Arrays im RAM pinnen, um asynchronen GPU-Transfer zu ermöglichen (non_blocking=True)
        input_sequences, target_sequences = input_sequences.pin_memory().to(device, non_blocking=True), target_sequences.pin_memory().to(device, non_blocking=True)
    else:
        input_sequences, target_sequences = input_sequences.to(device), target_sequences.to(device)
    return input_sequences, target_sequences

# Initialisiere diese Variablen hier, können überschrieben werden bei initialization_mode='resume' (von einem Checkpoint)
iteration_number = 0
best_validation_loss = 1e9

# Versuche vocab_size aus dem Datensatz abzuleiten
metadata_path = os.path.join(data_directory, 'meta.pkl')
metadata_vocab_size = None
if os.path.exists(metadata_path):
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    metadata_vocab_size = metadata['vocab_size']
    print(f"vocab_size gefunden = {metadata_vocab_size} (in {metadata_path})")

# Modell-Initialisierung
model_arguments = dict(n_layer=num_layers, n_head=num_attention_heads, n_embd=embedding_dimension, block_size=block_size,
                  bias=use_bias, vocab_size=None, dropout=dropout_rate) # Starte mit model_arguments von der Kommandozeile
if initialization_mode == 'scratch':
    # Initialisiere ein neues Modell von Grund auf
    print("Initialisiere ein neues Modell von Grund auf")
    # Bestimme die vocab_size für das Training von Grund auf
    if metadata_vocab_size is None:
        print("Verwende Standard vocab_size von GPT-2: 50304 (50257 aufgerundet für Effizienz)")
    model_arguments['vocab_size'] = metadata_vocab_size if metadata_vocab_size is not None else 50304
    gpt_configuration = GPTConfig(**model_arguments)
    model = GPT(gpt_configuration)
elif initialization_mode == 'resume':
    print(f"Setze Training fort von {output_directory}")
    # Training von einem Checkpoint fortsetzen
    checkpoint_path = os.path.join(output_directory, 'my_own_llm.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model_arguments = checkpoint['model_args']
    # Erzwinge Gleichheit dieser Konfigurations-Attribute, sonst können wir das Training nicht fortsetzen
    # Die restlichen Attribute (z.B. dropout) können wie gewünscht von der Kommandozeile bleiben
    for key in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_arguments[key] = checkpoint_model_arguments[key]
    # Erstelle das Modell
    gpt_configuration = GPTConfig(**model_arguments)
    model = GPT(gpt_configuration)
    state_dict = checkpoint['model']
    # Korrigiere die Schlüssel des State Dictionary :(
    # Unklar warum Checkpoints manchmal dieses Präfix haben, muss weiter debugged werden
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iteration_number = checkpoint['iter_num']
    best_validation_loss = checkpoint['best_val_loss']
elif initialization_mode.startswith('gpt2'):
    print(f"Initialisiere mit OpenAI GPT-2 Gewichten: {initialization_mode}")
    # Initialisiere mit OpenAI GPT-2 Gewichten
    override_arguments = dict(dropout=dropout_rate)
    model = GPT.load_pretrained_weights(initialization_mode, override_arguments)
    # Lese die erstellten Konfigurations-Parameter aus, damit wir sie korrekt im Checkpoint speichern können
    for key in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_arguments[key] = getattr(model.config, key)
# Reduziere die Modell-Blockgröße falls gewünscht, mittels Modell-Chirurgie
if block_size < model.config.block_size:
    model.adjust_context_window_size(block_size)
    model_arguments['block_size'] = block_size # Damit der Checkpoint den richtigen Wert hat
model.to(device)

# Initialisiere einen GradScaler. Bei enabled=False ist der Scaler eine No-Op
gradient_scaler = torch.cuda.amp.GradScaler(enabled=(data_type == 'float16'))

# Optimizer
optimizer = model.create_optimizer_with_weight_decay(weight_decay, learning_rate, (adam_beta1, adam_beta2), device_type)
if initialization_mode == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # Speicher freigeben

# Kompiliere das Modell
if enable_model_compilation:
    print("Kompiliere das Modell... (dauert ~1 Minute)")
    unoptimized_model = model
    model = torch.compile(model) # Benötigt PyTorch 2.0

# Modell ist bereit für Training (kein DDP Container nötig)

# Hilft bei der Schätzung eines beliebig genauen Losses über beide Splits mit vielen Batches
@torch.no_grad()
def evaluate_loss_on_splits():
    """
    Evaluiert den Modell-Loss auf Training- und Validierungsdaten.
    
    Berechnet den durchschnittlichen Loss über eval_iters Batches
    für sowohl Training- als auch Validierungsdaten.
    
    Returns:
        Dictionary mit 'train' und 'val' Loss-Werten
    """
    results = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_iterations)
        for evaluation_step in range(evaluation_iterations):
            input_batch, target_batch = load_data_batch(split)
            with autocast_context:
                logits, loss = model(input_batch, target_batch)
            losses[evaluation_step] = loss.item()
        results[split] = losses.mean()
    model.train()
    return results

def get_scheduled_learning_rate(current_iteration):
    """
    Berechnet die Lernrate basierend auf dem Trainingsfortschritt.
    
    Implementiert einen dreistufigen Schedule:
    1. Linearer Warmup von 0 bis learning_rate
    2. Cosine Decay von learning_rate zu min_lr
    3. Konstant bei min_lr nach lr_decay_iters
    
    Args:
        current_iteration: Aktuelle Iterations-Nummer
    
    Returns:
        Lernrate für die aktuelle Iteration
    """
    # Phase 1: Linearer Warmup
    if current_iteration < warmup_iterations:
        return learning_rate * (current_iteration + 1) / (warmup_iterations + 1)
    # Phase 2: Nach Decay-Phase konstant bei Minimal-Lernrate
    if current_iteration > learning_rate_decay_iterations:
        return minimum_learning_rate
    # Phase 3: Cosine Decay zwischen Warmup und finaler Phase
    decay_ratio = (current_iteration - warmup_iterations) / (learning_rate_decay_iterations - warmup_iterations)
    assert 0 <= decay_ratio <= 1
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Koeffizient im Bereich 0..1
    return minimum_learning_rate + coefficient * (learning_rate - minimum_learning_rate)

# ============================================================================
# Haupt-Trainingsschleife
# ============================================================================
input_batch, target_batch = load_data_batch('train') # Lade ersten Batch für Initialisierung
start_time = time.time()
local_iteration_number = 0 # Anzahl der Iterationen in der Lebensdauer dieses Prozesses
raw_model = model # Kein DDP-Container, direkt das Modell verwenden
# Variablen für ETA-Berechnung
start_time = time.time()
iteration_times = []  # Speichere aktuelle Iterationszeiten für gleitenden Durchschnitt
while True:

    # Bestimme und setze die Lernrate für diese Iteration
    current_learning_rate = get_scheduled_learning_rate(iteration_number) if enable_learning_rate_decay else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_learning_rate

    # Evaluiere den Loss auf Train/Val Sets und schreibe Checkpoints
    if iteration_number % evaluation_interval == 0:
        losses = evaluate_loss_on_splits()
        log_message = f"Schritt {iteration_number}: Train-Loss {losses['train']:.4f}, Val-Loss {losses['val']:.4f}"
        print(log_message)
        log_file.write(log_message + '\n')
        log_file.flush()
        if losses['val'] < best_validation_loss or always_save_checkpoint:
            best_validation_loss = losses['val']
            if iteration_number > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_arguments,
                    'iter_num': iteration_number,
                    'best_val_loss': best_validation_loss,
                    'config': config,
                }
                print(f"Speichere Checkpoint in {output_directory}")
                torch.save(checkpoint, os.path.join(output_directory, 'my_own_llm.pt'))
    if iteration_number == 0 and evaluation_only:
        break

    # Vorwärts-Rückwärts-Update mit optionaler Gradient-Akkumulation zur Simulation größerer Batch-Größen
    # und Verwendung des GradScalers falls Datentyp float16 ist
    for gradient_accumulation_step in range(gradient_accumulation_steps):
        with autocast_context:
            logits, loss = model(input_batch, target_batch)
            loss = loss / gradient_accumulation_steps # Skaliere den Loss für Gradient-Akkumulation
        # Lade nächsten Batch asynchron vor, während das Modell den Forward-Pass auf der GPU macht
        input_batch, target_batch = load_data_batch('train')
        # Rückwärts-Pass mit Gradient-Skalierung falls Training in fp16
        gradient_scaler.scale(loss).backward()
    # Gradienten-Clipping
    if gradient_clipping_value != 0.0:
        gradient_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_value)
    # Optimizer und Scaler-Schritt falls Training in fp16
    gradient_scaler.step(optimizer)
    gradient_scaler.update()
    # Lösche Gradienten so schnell wie möglich, Speicher wird nicht mehr benötigt
    optimizer.zero_grad(set_to_none=True)

    # Zeit-Messung und Logging
    end_time = time.time()
    iteration_duration = end_time - start_time
    start_time = end_time
    
    # Aktualisiere Iterationszeiten für ETA-Berechnung
    iteration_times.append(iteration_duration)
    if len(iteration_times) > 100:  # Behalte nur die letzten 100 Iterationszeiten
        iteration_times.pop(0)
    
    if iteration_number % logging_interval == 0:
        # Hole Loss als Float. Hinweis: Dies ist ein CPU-GPU Synchronisationspunkt
        # Skaliere hoch um die Division oben rückgängig zu machen, approximiert den wahren Gesamt-Loss (exakt wäre eine Summe gewesen)
        total_loss_value = loss.item() * gradient_accumulation_steps
        
        # Berechne Fortschritt in Prozent und geschätzte Restzeit (ETA)
        progress_percentage = (iteration_number / max_iterations) * 100
        if len(iteration_times) > 0:
            average_iteration_time = sum(iteration_times) / len(iteration_times)
            remaining_iterations = max_iterations - iteration_number
            estimated_time_seconds = remaining_iterations * average_iteration_time
            # Formatiere ETA als Stunden:Minuten:Sekunden
            estimated_hours = int(estimated_time_seconds // 3600)
            estimated_minutes = int((estimated_time_seconds % 3600) // 60)
            estimated_seconds = int(estimated_time_seconds % 60)
            estimated_time_string = f"{estimated_hours:02d}:{estimated_minutes:02d}:{estimated_seconds:02d}"
        else:
            estimated_time_string = "berechne..."
        
        log_message = f"Iteration {iteration_number}: Loss {total_loss_value:.4f}, Zeit {iteration_duration*1000:.2f}ms | Fortschritt {progress_percentage:.1f}% | ETA {estimated_time_string}"
        print(log_message)
        log_file.write(log_message + '\n')
        log_file.flush()
    iteration_number += 1
    local_iteration_number += 1

    # Abbruchbedingungen
    if iteration_number > max_iterations:
        break

log_file.write(f"\nTraining abgeschlossen um {datetime.datetime.now()}\n")
log_file.close()
