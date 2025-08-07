# Agent Router Test - Persistente Workflows mit Microsoft Semantic Kernel

## 🎯 Projektziel
Demonstration von drei verschiedenen Router-Architekturen für persistente Agent-Workflows mit Microsoft Semantic Kernel. Das System löst das Problem inkonsistenter Router-Entscheidungen während längerer Workflows wie Buchungsprozessen.

## 🚀 Quick Start

### Voraussetzungen
- .NET 8.0 SDK
- OpenAI API Key

### Installation & Start
```bash
# Projekt klonen/navigieren
cd AgentRouterTest

# Dependencies installieren
dotnet restore

# Anwendung starten
dotnet run
```

### Konfiguration
Passen Sie die OpenAI-Einstellungen in `appsettings.json` an:
```json
{
  "OpenAI": {
    "ApiKey": "IHR_API_KEY",
    "Model": "gpt-4o-mini"
  }
}
```

## 📊 Die drei Lösungsansätze

### 1️⃣ Stateful Routing mit Context-Awareness
**Konzept**: Dynamische Confidence-Schwellenwerte basierend auf Workflow-Status

**Wann verwenden?**
- Wenn Workflows unterschiedliche Prioritäten haben
- Bei klaren Workflow-Stadien (Booking, Support, etc.)
- Wenn Flexibilität mit Stabilität kombiniert werden soll

[Detaillierte Dokumentation](./Approach1_StatefulRouting/README_Approach1.md)

### 2️⃣ Agent-Ownership Pattern
**Konzept**: Agents entscheiden selbst über Kontrollübergabe

**Wann verwenden?**
- Wenn Agents komplex und selbstständig sind
- Bei klaren Verantwortungsbereichen
- Wenn explizite Übergaben gewünscht sind

[Detaillierte Dokumentation](./Approach2_AgentOwnership/README_Approach2.md)

### 3️⃣ Sticky Sessions mit Exit-Detection
**Konzept**: Router bleibt beim Agent bis explizites Exit-Signal

**Wann verwenden?**
- Bei längeren, zusammenhängenden Aufgaben
- Wenn Stabilität wichtiger als Flexibilität ist
- Bei klaren Nutzer-Intentionen für Wechsel

[Detaillierte Dokumentation](./Approach3_StickySessions/README_Approach3.md)

## 🤖 Verfügbare Agents

| Agent | Beschreibung | Fähigkeiten |
|-------|--------------|-------------|
| **BookingAgent** | Reise- und Buchungsexperte | Flüge, Hotels, Mietwagen, Umbuchungen |
| **SupportAgent** | Technischer Support | Problembehebung, Account-Hilfe, Passwort-Reset |
| **KnowledgeAgent** | Wissensexperte | Faktenfragen, Definitionen, Erklärungen |

## 🏗️ Projektstruktur
```
AgentRouterTest/
├── Common/                        # Gemeinsame Komponenten
│   ├── Interfaces/               # IAgent, IRouter
│   ├── Models/                   # ConversationState, RoutingResult
│   └── BaseAgents/               # Basis-Agent-Implementierungen
├── Approach1_StatefulRouting/    # Ansatz 1
│   ├── ConversationStatePlugin.cs
│   ├── StatefulRouter.cs
│   └── StatefulRoutingOrchestrator.cs
├── Approach2_AgentOwnership/     # Ansatz 2
│   ├── ISelfManagedAgent.cs
│   ├── SelfManaged*Agent.cs
│   └── OwnershipOrchestrator.cs
├── Approach3_StickySessions/     # Ansatz 3
│   ├── ExitDetectionPlugin.cs
│   ├── StickySessionRouter.cs
│   └── StickyRoutingOrchestrator.cs
└── Program.cs                    # Hauptprogramm mit Chat-Interface
```

## 💬 Chat-Befehle
- **`exit`**: Beendet den Chat
- **`stats`**: Zeigt Session-Statistiken
- **Normal Text**: Wird vom gewählten Agent verarbeitet

## 📈 Vergleich der Ansätze

| Aspekt | Stateful Routing | Agent-Ownership | Sticky Sessions |
|--------|------------------|-----------------|-----------------|
| **Komplexität** | Mittel | Hoch | Niedrig |
| **Flexibilität** | Hoch | Mittel | Niedrig |
| **Stabilität** | Mittel | Hoch | Sehr hoch |
| **Transparenz** | Hoch | Mittel | Hoch |
| **Setup-Aufwand** | Mittel | Hoch | Niedrig |
| **Best für** | Dynamische Workflows | Komplexe Agents | Lange Aufgaben |

## 🔧 Technologie-Stack
- **.NET 8.0**: Runtime
- **Microsoft Semantic Kernel**: AI Orchestration
- **OpenAI GPT-4**: Language Model
- **C# 12**: Programmiersprache

## 📝 Beispiel-Szenarien

### Szenario 1: Flugbuchung mit Unterbrechung
```
User: Ich möchte einen Flug buchen
Bot: [BookingAgent] Gerne! Wohin möchten Sie fliegen?
User: Nach Berlin
Bot: [BookingAgent] Wann möchten Sie fliegen?
User: Was ist die Hauptstadt von Frankreich?
```
- **Stateful**: Bleibt bei BookingAgent (hoher Threshold)
- **Ownership**: BookingAgent behält Kontrolle (Priority: 5)
- **Sticky**: Bleibt bei BookingAgent (kein Exit-Signal)

### Szenario 2: Support-Fall gelöst
```
User: Mein Passwort funktioniert nicht
Bot: [SupportAgent] Ich helfe Ihnen beim Passwort-Reset...
User: Danke, jetzt geht es wieder!
Bot: [SupportAgent] Freut mich! Kann ich noch etwas tun?
User: Ich möchte einen Flug buchen
```
- **Stateful**: Wechselt zu BookingAgent (Support completed)
- **Ownership**: SupportAgent gibt ab, schlägt BookingAgent vor
- **Sticky**: Exit-Signal erkannt, wechselt zu BookingAgent

## 🐛 Bekannte Einschränkungen
- Agents haben keine echte Buchungs-/Support-Funktionalität (nur Simulation)
- OpenAI API Key wird in appsettings.json gespeichert (Production: Secrets verwenden!)
- Keine Persistierung der Sessions zwischen Programmstarts

## 📚 Weiterführende Ressourcen
- [Microsoft Semantic Kernel Docs](https://learn.microsoft.com/semantic-kernel/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [.NET 8.0 Documentation](https://docs.microsoft.com/dotnet/)

## 🤝 Beitragen
Dieses Projekt dient zu Demonstrationszwecken. Verbesserungsvorschläge und Erweiterungen sind willkommen!

## 📄 Lizenz
Dieses Projekt ist zu Bildungszwecken erstellt und kann frei verwendet werden.