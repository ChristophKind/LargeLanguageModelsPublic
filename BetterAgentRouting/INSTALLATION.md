# Installation und Start-Anleitung

## .NET 8.0 SDK Installation

### macOS (Ihr System)

#### Option 1: Über die offizielle Website
1. Besuchen Sie https://dotnet.microsoft.com/download/dotnet/8.0
2. Laden Sie das .NET 8.0 SDK für macOS herunter
3. Führen Sie den Installer aus

#### Option 2: Über Homebrew (empfohlen)
```bash
# Homebrew installieren (falls noch nicht vorhanden)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# .NET SDK installieren
brew install --cask dotnet-sdk
```

#### Option 3: Über MacPorts
```bash
sudo port install dotnet-sdk-8.0
```

### Verifikation der Installation
```bash
dotnet --version
# Sollte 8.0.x ausgeben
```

## Projekt starten

### 1. Dependencies installieren
```bash
cd /Users/christoph/Desktop/AgentRouterTest
dotnet restore
```

### 2. Projekt kompilieren
```bash
dotnet build
```

### 3. Anwendung starten
```bash
dotnet run
```

## Troubleshooting

### Fehler: "The specified framework 'Microsoft.NETCore.App', version '8.0.0' was not found"
**Lösung**: Installieren Sie das .NET 8.0 Runtime zusätzlich zum SDK:
```bash
brew install --cask dotnet
```

### Fehler: "OpenAI API Key nicht gefunden"
**Lösung**: Stellen Sie sicher, dass der API Key in `appsettings.json` korrekt eingetragen ist:
```json
{
  "OpenAI": {
    "ApiKey": "sk-...",  // Hier Ihren API Key eintragen
    "Model": "gpt-4o-mini"
  }
}
```

### Fehler beim NuGet Package Restore
**Lösung**: Cache löschen und erneut versuchen:
```bash
dotnet nuget locals all --clear
dotnet restore
```

## Erwartete Ausgabe beim Start

Nach erfolgreichem Start sollten Sie folgendes sehen:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║     █████╗  ██████╗ ███████╗███╗   ██╗████████╗    ██████╗  ██████╗ ██╗   ██╗████████╗███████╗██████╗ 
║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝    ██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝██╔════╝██╔══██╗
║    ...
╚══════════════════════════════════════════════════════════════════════════════╝

Willkommen zum Agent Router Test System!
...

════════════════════════════════════════════════════════════════════════════════
Wählen Sie einen Router-Ansatz:
────────────────────────────────────────────────────────────────────────────────
1. Stateful Routing mit Context-Awareness
   → Dynamische Confidence-Schwellenwerte basierend auf Workflow-Status

2. Agent-Ownership Pattern
   → Agents entscheiden selbst über Kontrollübergabe

3. Sticky Sessions mit Exit-Detection
   → Router bleibt bei Agent bis explizites Exit-Signal

0. Beenden
════════════════════════════════════════════════════════════════════════════════

Ihre Wahl: 
```

## Verwendung

1. Wählen Sie einen der drei Ansätze (1, 2 oder 3)
2. Starten Sie einen Chat mit den Agents
3. Testen Sie verschiedene Szenarien:
   - Buchungsanfragen: "Ich möchte einen Flug buchen"
   - Support-Anfragen: "Mein Passwort funktioniert nicht"
   - Wissensfragen: "Was ist die Hauptstadt von Frankreich?"
4. Beobachten Sie das Routing-Verhalten
5. Nutzen Sie 'stats' für Statistiken oder 'exit' zum Beenden

## Systemvoraussetzungen

- macOS, Windows oder Linux
- .NET 8.0 SDK
- Internetverbindung (für OpenAI API)
- Terminal/Konsole
- Ca. 100 MB freier Speicherplatz