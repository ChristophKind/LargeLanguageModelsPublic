"""
Vollständige Definition eines GPT Language Models in einer einzigen Datei.

Dieses Modul implementiert die komplette GPT (Generative Pre-trained Transformer) Architektur,
basierend auf dem Paper 'Language Models are Unsupervised Multitask Learners' von OpenAI.

Die Implementierung umfasst:
- Multi-Head Self-Attention mit kausaler Maskierung
- Position und Token Embeddings
- Layer Normalization
- Feed-Forward Netzwerke (MLP)
- Residual Connections

Referenzen:
1) Offizielle GPT-2 TensorFlow Implementierung von OpenAI:
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) HuggingFace/Transformers PyTorch Implementierung:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """
    Layer Normalization mit optionalem Bias-Parameter.
    
    Layer Normalization normalisiert die Aktivierungen über die Feature-Dimension,
    was zu stabilerer und schnellerer Konvergenz beim Training führt.
    
    PyTorch's Standard LayerNorm unterstützt kein einfaches bias=False,
    daher diese eigene Implementierung.
    
    Args:
        ndim: Dimensionalität der zu normalisierenden Features
        bias: Boolean, ob ein Bias-Term verwendet werden soll
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """
    Kausale Self-Attention Schicht für autoregressive Modellierung.
    
    Diese Klasse implementiert den Multi-Head Self-Attention Mechanismus mit kausaler
    Maskierung, sodass jede Position nur auf vorherige Positionen achten kann.
    Dies ist essentiell für die autoregressive Textgenerierung.
    
    Features:
    - Multi-Head Attention für parallele Aufmerksamkeitsmuster
    - Kausale Maskierung für autoregressive Modellierung
    - Optional: Flash Attention für beschleunigte Berechnung auf GPUs
    - Dropout für Regularisierung
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Kombinierte Projektion für Query, Key und Value für alle Attention-Heads
        # Dies ist effizienter als separate Projektionen
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output-Projektion nach der Attention-Berechnung
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularisierung durch Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # Flash Attention nutzt optimierte CUDA-Kernel für deutlich schnellere Berechnung
        # Verfügbar ab PyTorch 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNUNG: Verwende langsame Attention. Flash Attention benötigt PyTorch >= 2.0")
            # Kausale Maske: Stellt sicher, dass jede Position nur auf vorherige Positionen
            # (links in der Sequenz) achten kann - essentiell für autoregressive Modellierung
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, hidden_states):
        """
        Berechnet die Self-Attention für die Eingabe.
        
        Args:
            hidden_states: Eingabe-Tensor der Form (Batch, Sequenzlänge, Embedding-Dimension)
        
        Returns:
            Output-Tensor nach Attention-Berechnung und Projektion
        """
        batch_size, sequence_length, embedding_dim = hidden_states.size()

        # Berechne Query, Key und Value für alle Heads gleichzeitig
        # und reorganisiere die Dimensionen für parallele Head-Berechnung
        queries, keys, values = self.c_attn(hidden_states).split(self.n_embd, dim=2)
        keys = keys.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2)
        queries = queries.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2)
        values = values.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2)

        # Kausale Self-Attention Berechnung
        # Matrixmultiplikation: (B, n_heads, T, head_size) x (B, n_heads, head_size, T) -> (B, n_heads, T, T)
        if self.flash:
            # Effiziente Attention-Berechnung mit optimierten Flash Attention CUDA-Kerneln
            attention_output = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Manuelle Implementierung der Attention (langsamer, aber überall verfügbar)
            attention_weights = (queries @ keys.transpose(-2, -1)) * (1.0 / math.sqrt(keys.size(-1)))
            attention_weights = attention_weights.masked_fill(self.bias[:,:,:sequence_length,:sequence_length] == 0, float('-inf'))
            attention_weights = F.softmax(attention_weights, dim=-1)
            attention_weights = self.attn_dropout(attention_weights)
            attention_output = attention_weights @ values
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)

        # Output-Projektion und Dropout für Regularisierung
        attention_output = self.resid_dropout(self.c_proj(attention_output))
        return attention_output

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) des Transformer-Blocks.
    
    Implementiert ein zweischichtiges neuronales Netzwerk mit GELU-Aktivierung.
    Die versteckte Schicht hat typischerweise die 4-fache Dimension der Eingabe,
    was dem Standard-Transformer-Design entspricht.
    
    Die MLP-Schicht verarbeitet jede Position unabhängig und ermöglicht es dem
    Modell, komplexe nicht-lineare Transformationen zu lernen.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states):
        """
        Vorwärtsdurchlauf durch das Feed-Forward Netzwerk.
        
        Args:
            hidden_states: Eingabe-Tensor der Form (Batch, Sequenzlänge, n_embd)
        
        Returns:
            Transformierter Tensor gleicher Form wie die Eingabe
        """
        hidden_states = self.c_fc(hidden_states)    # Projektion auf 4*n_embd Dimension
        hidden_states = self.gelu(hidden_states)    # GELU nicht-lineare Aktivierung
        hidden_states = self.c_proj(hidden_states)  # Projektion zurück auf n_embd Dimension
        hidden_states = self.dropout(hidden_states) # Dropout für Regularisierung
        return hidden_states

class Block(nn.Module):
    """
    Transformer-Block: Die Kernkomponente der GPT-Architektur.
    
    Jeder Block besteht aus:
    1. Multi-Head Self-Attention mit kausaler Maskierung
    2. Feed-Forward Netzwerk (MLP)
    3. Zwei Layer-Normalization Schichten
    4. Residual Connections um beide Hauptkomponenten
    
    Diese Blöcke werden mehrfach gestapelt, um das tiefe Transformer-Modell zu bilden.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, hidden_states):
        """
        Verarbeitet die Eingabe durch Attention und Feed-Forward Schichten.
        
        Verwendet Pre-Normalization (Layer Norm vor der Transformation)
        und Residual Connections für stabiles Training tiefer Netzwerke.
        
        Args:
            hidden_states: Eingabe-Tensor der Form (Batch, Sequenzlänge, n_embd)
        
        Returns:
            Transformierter Tensor gleicher Form
        """
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states))  # Attention mit Residual Connection
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))   # Feed-Forward mit Residual Connection
        return hidden_states

@dataclass
class GPTConfig:
    """
    Konfigurationsklasse für das GPT-Modell.
    
    Definiert alle Hyperparameter der Modellarchitektur. Die Standardwerte
    entsprechen dem GPT-2 'small' Modell mit 124M Parametern.
    
    Attributes:
        block_size: Maximale Sequenzlänge (Kontext-Fenster)
        vocab_size: Größe des Vokabulars (auf 64 aufgerundet für GPU-Effizienz)
        n_layer: Anzahl der Transformer-Blöcke
        n_head: Anzahl der Attention-Heads pro Block
        n_embd: Dimensionalität der Embeddings und versteckten Zustände
        dropout: Dropout-Rate für Regularisierung (0.0 beim Pretraining)
        bias: Ob Bias-Parameter in Linear- und LayerNorm-Schichten verwendet werden
    """
    block_size: int = 1024      # Maximale Kontext-Länge
    vocab_size: int = 50304     # GPT-2 vocab_size 50257, aufgerundet auf Vielfaches von 64 für Effizienz
    n_layer: int = 12           # Anzahl der Transformer-Schichten
    n_head: int = 12            # Anzahl der Attention-Heads
    n_embd: int = 768           # Embedding-Dimension
    dropout: float = 0.0        # Dropout-Wahrscheinlichkeit
    bias: bool = True           # True: Bias wie in GPT-2, False: etwas besser und schneller

class GPT(nn.Module):
    """
    Das komplette GPT (Generative Pre-trained Transformer) Sprachmodell.
    
    Diese Klasse implementiert die vollständige GPT-Architektur mit:
    - Token- und Positions-Embeddings
    - Gestapelten Transformer-Blöcken
    - Finaler Projektionsschicht für Vorhersagen
    
    Das Modell kann sowohl für Training (mit Targets) als auch für
    Inferenz (Textgenerierung) verwendet werden.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight Tying: Token-Embedding und Output-Projektion teilen sich die Gewichte.
        # Dies reduziert die Parameteranzahl und verbessert oft die Performance.
        # Bei torch.compile() können Warnungen auftreten, die aber harmlos sind.
        # Siehe: https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialisiere alle Gewichte mit Standardwerten
        self.apply(self._init_weights)
        # Spezielle skalierte Initialisierung für Residual-Projektionen
        # (wie im GPT-2 Paper beschrieben für bessere Trainings-Stabilität)
        for param_name, param in self.named_parameters():
            if param_name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Ausgabe der Gesamtzahl der trainierbaren Parameter
        print("Anzahl der Parameter: %.2fM" % (self.calculate_parameter_count()/1e6,))

    def calculate_parameter_count(self, non_embedding=True):
        """
        Berechnet die Anzahl der Parameter im Modell.
        
        Args:
            non_embedding: Wenn True (Standard), werden Positions-Embeddings
                          von der Zählung ausgeschlossen, da sie nicht zum
                          eigentlichen 'Rechnen' beitragen.
        
        Returns:
            Anzahl der Parameter als Integer.
            
        Hinweis:
            Token-Embeddings werden trotz non_embedding=True mitgezählt,
            da sie durch Weight-Tying auch als Output-Gewichte dienen.
        """
        total_params = sum(param.numel() for param in self.parameters())
        if non_embedding:
            total_params -= self.transformer.wpe.weight.numel()
        return total_params

    def _init_weights(self, module):
        """
        Initialisiert die Gewichte eines Moduls nach GPT-2 Spezifikation.
        
        Args:
            module: Das zu initialisierende PyTorch-Modul
            
        Initialisierungs-Schema:
        - Linear-Layer: Normalverteilung mit std=0.02
        - Embeddings: Normalverteilung mit std=0.02
        - Bias: Nullen
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        """
        Vorwärtsdurchlauf durch das GPT-Modell.
        
        Args:
            idx: Token-IDs der Form (Batch, Sequenzlänge)
            targets: Optionale Ziel-Token für Loss-Berechnung, Form (Batch, Sequenzlänge)
        
        Returns:
            logits: Vorhersage-Scores für jedes Token im Vokabular
            loss: Cross-Entropy Loss (nur wenn targets angegeben)
        """
        device = input_ids.device
        batch_size, sequence_length = input_ids.size()
        assert sequence_length <= self.config.block_size, f"Sequenzlänge {sequence_length} überschreitet maximale Block-Größe {self.config.block_size}"
        position_ids = torch.arange(0, sequence_length, dtype=torch.long, device=device)

        # Vorwärtsdurchlauf durch das Transformer-Modell
        token_embeddings = self.transformer.wte(input_ids) # Token-Embeddings: (Batch, Sequenz, n_embd)
        position_embeddings = self.transformer.wpe(position_ids) # Positions-Embeddings: (Sequenz, n_embd)
        hidden_states = self.transformer.drop(token_embeddings + position_embeddings)
        for block in self.transformer.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.transformer.ln_f(hidden_states)

        if targets is not None:
            # Wenn Ziel-Token gegeben sind, berechne den Trainings-Loss
            logits = self.lm_head(hidden_states)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inferenz-Optimierung: Berechne Logits nur für die letzte Position
            # (da wir nur das nächste Token vorhersagen müssen)
            logits = self.lm_head(hidden_states[:, [-1], :]) # [-1] als Liste um Zeitdimension zu erhalten
            loss = None

        return logits, loss

    def adjust_context_window_size(self, block_size):
        """
        Passt die Kontext-Fenstergröße des Modells an.
        
        Ermöglicht es, ein vortrainiertes Modell mit großem Kontext-Fenster
        (z.B. GPT-2 mit 1024) für kleinere Sequenzen zu verwenden.
        
        Args:
            block_size: Neue maximale Sequenzlänge (muss <= aktuelle block_size sein)
            
        Hinweis:
            Dies ist eine einmalige Operation und kann nicht rückgängig gemacht werden.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def load_pretrained_weights(cls, model_type, override_args=None):
        """
        Lädt vortrainierte GPT-2 Gewichte von HuggingFace.
        
        Args:
            model_type: Eines von 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            override_args: Optionales Dict mit zu überschreibenden Konfig-Parametern
                          (derzeit nur 'dropout' unterstützt)
        
        Returns:
            GPT-Modell mit geladenen vortrainierten Gewichten
            
        Modellgrößen:
            - gpt2: 124M Parameter
            - gpt2-medium: 350M Parameter
            - gpt2-large: 774M Parameter
            - gpt2-xl: 1.5B Parameter
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # Standard: leeres Dict
        # Nur Dropout kann überschrieben werden
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("Lade Gewichte vom vortrainierten GPT: %s" % model_type)

        # Modell-Konfiguration basierend auf model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M Parameter
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M Parameter
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M Parameter
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1.5B Parameter
        }[model_type]
        print("Erzwinge vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # Immer 50257 für GPT-2 Checkpoints
        config_args['block_size'] = 1024  # Immer 1024 für GPT-2 Checkpoints
        config_args['bias'] = True        # Immer True für GPT-2 Checkpoints
        # Dropout-Rate kann optional überschrieben werden
        if 'dropout' in override_args:
            print(f"Überschreibe Dropout-Rate zu {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # Erstelle ein frisch initialisiertes GPT-Modell
        config = GPTConfig(**config_args)
        model = GPT(config)
        state_dict = model.state_dict()
        state_dict_keys = state_dict.keys()
        state_dict_keys = [key for key in state_dict_keys if not key.endswith('.attn.bias')] # Entferne Attention-Maske (Buffer, kein Parameter)

        # Initialisiere ein HuggingFace/Transformers Modell
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        huggingface_state_dict = model_hf.state_dict()

        # Kopiere Gewichte und stelle sicher, dass alle Parameter in Namen und Form übereinstimmen
        huggingface_keys = huggingface_state_dict.keys()
        huggingface_keys = [key for key in huggingface_keys if not key.endswith('.attn.masked_bias')] # Ignoriere Buffer
        huggingface_keys = [key for key in huggingface_keys if not key.endswith('.attn.bias')] # Ignoriere Maske (auch Buffer)
        transposed_weights = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # OpenAI verwendet "Conv1D" Module, wir verwenden aber Standard Linear-Layer.
        # Daher müssen wir die Gewichte beim Import transponieren.
        assert len(huggingface_keys) == len(state_dict_keys), f"mismatched keys: {len(huggingface_keys)} != {len(state_dict_keys)}"
        for key in huggingface_keys:
            if any(key.endswith(weight_name) for weight_name in transposed_weights):
                # Spezialbehandlung für Conv1D Gewichte (transponieren nötig)
                assert huggingface_state_dict[key].shape[::-1] == state_dict[key].shape
                with torch.no_grad():
                    state_dict[key].copy_(huggingface_state_dict[key].t())
            else:
                # Direktes Kopieren aller anderen Parameter
                assert huggingface_state_dict[key].shape == state_dict[key].shape
                with torch.no_grad():
                    state_dict[key].copy_(huggingface_state_dict[key])

        return model

    def create_optimizer_with_weight_decay(self, weight_decay, learning_rate, betas, device_type):
        """
        Erstellt einen AdamW-Optimizer mit selektivem Weight Decay.
        
        Weight Decay wird nur auf 2D-Parameter (Gewichtsmatrizen) angewendet,
        nicht auf 1D-Parameter (Bias, LayerNorm).
        
        Args:
            weight_decay: Stärke des Weight Decay (L2-Regularisierung)
            learning_rate: Lernrate
            betas: Adam Beta-Parameter (Momentum-Koeffizienten)
            device_type: 'cuda' oder 'cpu' (bestimmt ob fused AdamW verwendet wird)
        
        Returns:
            Konfigurierter AdamW-Optimizer
        """
        # Sammle alle trainierbaren Parameter
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        # Filtere Parameter die keinen Gradienten benötigen
        param_dict = {param_name: param for param_name, param in param_dict.items() if param.requires_grad}
        # Erstelle Optimizer-Gruppen: 2D-Parameter (Matrizen) mit Weight Decay,
        # 1D-Parameter (Bias, LayerNorm) ohne Weight Decay
        decay_params = [param for name, param in param_dict.items() if param.dim() >= 2]
        nodecay_params = [param for name, param in param_dict.items() if param.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(param.numel() for param in decay_params)
        num_nodecay_params = sum(param.numel() for param in nodecay_params)
        print(f"Anzahl decay Parameter-Tensoren: {len(decay_params)}, mit {num_decay_params:,} Parametern")
        print(f"Anzahl non-decay Parameter-Tensoren: {len(nodecay_params)}, mit {num_nodecay_params:,} Parametern")
        # Erstelle AdamW-Optimizer (nutze fused Version falls verfügbar für bessere Performance)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Verwende fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate_text_tokens(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generiert neue Token basierend auf einer Eingabesequenz.
        
        Verwendet autoregressive Generierung: Jedes neu generierte Token
        wird zur Eingabe hinzugefügt für die Vorhersage des nächsten.
        
        Args:
            idx: Start-Sequenz als LongTensor der Form (Batch, Sequenzlänge)
            max_new_tokens: Anzahl der zu generierenden Token
            temperature: Temperatur für Sampling (höher = zufälliger, niedriger = deterministischer)
            top_k: Wenn angegeben, sample nur aus den top-k wahrscheinlichsten Token
        
        Returns:
            Erweiterte Sequenz mit generierten Token
            
        Hinweis:
            Modell sollte im eval() Modus sein für beste Ergebnisse.
        """
        for _ in range(max_new_tokens):
            # Wenn Sequenz zu lang wird, schneide auf maximale Block-Größe zu
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Vorwärtsdurchlauf für Vorhersage des nächsten Tokens
            logits, _ = self(idx_cond)
            # Extrahiere Logits der letzten Position und skaliere mit Temperatur
            logits = logits[:, -1, :] / temperature
            # Optional: Beschränke auf top-k wahrscheinlichste Token
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Softmax zur Umwandlung von Logits in Wahrscheinlichkeiten
            probs = F.softmax(logits, dim=-1)
            # Sample ein Token aus der Wahrscheinlichkeitsverteilung
            next_token_id = torch.multinomial(probs, num_samples=1)
            # Füge gesampletes Token zur Sequenz hinzu und fahre fort
            idx = torch.cat((idx, next_token_id), dim=1)

        return idx
