"""
Einfacher Konfigurations-Manager für GPT-Training.

Dieses Skript ermöglicht es, Konfigurationsparameter über die Kommandozeile
zu überschreiben.

Verwendung:
$ python train.py --batch_size=32 --learning_rate=0.001

Funktionsweise:
- Wird via exec() in anderen Skripten ausgeführt
- Überschreibt direkt die globalen Variablen
- Nur Kommandozeilen-Argumente im Format --key=value werden unterstützt

Hinweis: Dieser Ansatz ist pragmatisch aber unkonventionell.
Er vermeidet die Notwendigkeit, überall 'config.' zu schreiben.
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    # Nur --key=value Argumente werden verarbeitet
    if arg.startswith('--') and '=' in arg:
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # Versuche den Wert zu evaluieren (für bool, int, float, etc.)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # Falls Evaluierung fehlschlägt, verwende als String
                attempt = val
            # Stelle sicher, dass die Typen übereinstimmen
            assert type(attempt) == type(globals()[key])
            # Überschreibe den globalen Wert
            print(f"Überschreibe: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
