"""
Text-Generierung mit einem trainierten GPT-Modell.

Dieses Skript lädt ein trainiertes Modell und generiert Text basierend
auf einem Start-Prompt. Unterstützt sowohl lokale Checkpoints als auch
vortrainierte GPT-2 Modelle von HuggingFace.

Features:
- Laden von Checkpoints oder vortrainierten Modellen
- Temperature-basiertes Sampling für kontrollierte Zufälligkeit
- Top-k Sampling zur Qualitätsverbesserung
- Batch-Generierung mehrerer Samples
"""
import os
import pickle
from contextlib import nullcontext
from datetime import datetime
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Konfigurations-Parameter für Text-Generierung
# -----------------------------------------------------------------------------

# Modell-Quelle: 'resume' (lokaler Checkpoint) oder GPT-2 Variante ('gpt2', 'gpt2-xl', etc.)
init_from = 'resume'
out_dir = 'out' # Checkpoint-Verzeichnis (nur bei init_from='resume')

# Generierungs-Parameter
start = "\n" # Start-Prompt (oder "<|endoftext|>" oder "FILE:datei.txt" für Datei-Input)
num_samples = 10 # Anzahl der zu generierenden Text-Samples
max_new_tokens = 500 # Maximale Länge der generierten Sequenz pro Sample

# Sampling-Parameter
temperature = 0.8 # Temperatur: <1.0 = deterministischer, >1.0 = zufälliger
top_k = 200 # Nur top-k wahrscheinlichste Token behalten

# System-Konfiguration
seed = 1337 # Random Seed für Reproduzierbarkeit
device = 'cuda' # Gerät: 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # PyTorch 2.0 Kompilierung für bessere Performance

exec(open('configurator.py').read()) # Überschreibe Konfiguration von Kommandozeile
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'my_own_llm.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Display checkpoint information
    print(f"Checkpoint geladen: {ckpt_path}")
    ckpt_size = os.path.getsize(ckpt_path) / (1024 * 1024)  # Convert to MB
    ckpt_mtime = os.path.getmtime(ckpt_path)
    ckpt_date = datetime.fromtimestamp(ckpt_mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Checkpoint Größe: {ckpt_size:.1f} MB")
    print(f"Checkpoint Datum: {ckpt_date}")
    print()
    
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.load_pretrained_weights(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    # Check for meta.pkl in data/ directory (dataset is now 'data')
    if checkpoint['config']['dataset'] == 'data':
        meta_path = os.path.join('data', 'meta.pkl')
    else:
        # Fallback for older checkpoints that might still use subdirectories
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Lade Meta-Daten von {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("Keine meta.pkl gefunden, nehme GPT-2 Encodings an...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate_text_tokens(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
