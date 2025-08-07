# Ansatz 3: Sticky Sessions mit Exit-Detection

## Überblick
Der Router implementiert "klebriges" Verhalten: Solange ein Workflow aktiv ist, bleibt er beim aktuellen Agent. Nur bei expliziten Exit-Signalen oder klaren Themenwechseln wird neu geroutet.

## Kernkonzept
📌 **Sticky = Der Router "klebt" am aktuellen Agent**

Der Agent behält die Kontrolle, bis:
- Ein explizites Exit-Signal erkannt wird
- Der Agent komplett ungeeignet ist (Confidence < 0.1)
- Frustration/Stagnation erkannt wird
- Ein klarer Themenwechsel stattfindet

## Hauptkomponenten

### 1. ExitDetectionPlugin
Erkennt verschiedene Arten von Exit-Signalen:

#### Exit-Typen:
- **Explicit**: "das war's", "andere Frage", "fertig", "abbrechen"
- **Completion**: "vielen Dank", "perfekt", "alles klar"
- **TopicChange**: Themenwechsel erkannt
- **Frustration**: Viele Turns ohne Fortschritt

#### Erkennungsmethoden:
1. **Pattern-Matching**: Suche nach Exit-Phrasen
2. **KI-Analyse**: LLM analysiert subtile Signale
3. **Kontext-Analyse**: Workflow-Status und Turn-Count
4. **Stagnations-Erkennung**: Wiederholende Fragen

### 2. StickySessionRouter
**Konfiguration**:
```csharp
_minConfidenceToStick = 0.3;  // Minimale Confidence für Sticky
_exitThreshold = 0.6;          // Exit-Signal Schwellenwert
_maxTurnsBeforePrompt = 10;   // Nachfrage nach X Turns
```

**Entscheidungslogik**:
1. Prüfe Exit-Intent
2. Evaluiere aktuelle Agent-Eignung
3. Entscheide ob Sticky Session gebrochen wird
4. Falls nicht: Bleibe beim Agent (STICKY!)
5. Falls ja: Normales Routing zu bestem Agent

### 3. StickyRoutingOrchestrator
**Zusätzliche Features**:
- Trackt Sticky-Metriken (Dauer, Breaks, etc.)
- Erkennt Themenwechsel
- Schlägt Nutzer-Prompts vor bei langen Sessions
- Visualisiert Sticky-Status (📌)

## Sticky Session Metriken

### Getrackte Metriken:
- **sticky_duration**: Aktuelle Sticky-Dauer in Turns
- **sticky_breaks**: Anzahl der Session-Unterbrechungen
- **average_sticky_duration**: Durchschnittliche Dauer
- **exit_signals_detected**: Erkannte Exit-Signale
- **topic_changes**: Anzahl Themenwechsel

## Exit-Detection Details

### Explizite Exit-Phrasen:
```
"das war's", "danke das war alles", "fertig", 
"andere frage", "themenwechsel", "abbrechen", 
"vergiss es", "ich möchte lieber", "wechseln zu"
```

### Completion-Indikatoren:
```
"vielen dank", "perfekt", "super danke", 
"alles klar", "verstanden", "hat geholfen"
```

### Frustrations-Erkennung:
- Mehr als 7 Turns ohne Workflow-Abschluss
- Wiederholende Fragen (gleiche Inputs)
- Keine Fortschritte im Workflow

## Vorteile
✅ **Stabilität**: Verhindert zufällige Agent-Wechsel  
✅ **Natürlicher Flow**: Gespräche werden nicht unterbrochen  
✅ **Einfache Logik**: Klare Regeln für Wechsel  
✅ **Nutzerfreundlich**: Explizite Kontrolle über Wechsel  

## Nachteile
❌ **Trägheit**: Kann zu lange bei falschem Agent bleiben  
❌ **Exit-Erkennung**: Nicht alle Exit-Signale werden erkannt  
❌ **Frustrationspotenzial**: Bei falschem Agent "gefangen"  

## Beispiel-Konversation
```
👤: Ich brauche Hilfe mit meinem Account
🤖 SupportAgent: Gerne helfe ich Ihnen bei Account-Problemen...
[📌 STICKY SESSION AKTIV]

👤: Ich kann mich nicht einloggen
🤖 SupportAgent: Lassen Sie uns das Problem lösen...
[📌 STICKY - Duration: 2 turns]

👤: Mein Passwort funktioniert nicht
🤖 SupportAgent: Ich setze Ihr Passwort zurück...
[📌 STICKY - Duration: 3 turns]
[Kein Exit-Signal erkannt]

👤: Danke, das war's! Was ist die Hauptstadt von Spanien?
[EXIT DETECTED: Type=Completion + TopicChange]
[🔓 STICKY SESSION BEENDET nach 3 turns]
🤖 KnowledgeAgent: Die Hauptstadt von Spanien ist Madrid.
```

## Spezielle Szenarien

### Lange Sessions:
Nach 10 Turns schlägt das System vor:
```
💡 Tipp: Möchten Sie mit der aktuellen Aufgabe fortfahren 
         oder haben Sie eine andere Frage?
```

### Stagnation:
Bei wiederholenden Fragen:
```
[STICKY BREAK: Konversation stagniert]
→ Automatischer Agent-Wechsel
```

### Explizite Wechsel:
```
👤: Ich möchte lieber eine Buchung machen
[EXIT DETECTED: Explicit - "ich möchte lieber"]
→ Sofortiger Wechsel zu BookingAgent
```

## Konfiguration
```csharp
// In StickySessionRouter anpassbar:
_minConfidenceToStick = 0.3;   // Wie schlecht darf Agent sein?
_exitThreshold = 0.6;           // Wie sicher für Exit?
_maxTurnsBeforePrompt = 10;    // Wann nachfragen?

// Exit-Phrasen erweiterbar in ExitDetectionPlugin
_explicitExitPhrases = { /* ... */ };
_completionIndicators = { /* ... */ };
```

## Verwendung
```csharp
var orchestrator = new StickyRoutingOrchestrator(kernel, loggerFactory);
var sessionId = orchestrator.StartNewSession();

var result = await orchestrator.ProcessMessageAsync(sessionId, userInput);

// Prüfe Sticky-Status
if (result.SessionSticky)
{
    Console.WriteLine("📌 Session ist sticky!");
}

// Prüfe Exit-Detection
if (result.ExitDetection?.ExitDetected == true)
{
    Console.WriteLine($"Exit erkannt: {result.ExitDetection.ExitType}");
}
```

## Best Practices
1. **Klare Exit-Phrasen**: Nutzer informieren, wie sie wechseln können
2. **Timeouts setzen**: Nach X Turns automatisch nachfragen
3. **Frustration vermeiden**: Bei Stagnation proaktiv wechseln
4. **Monitoring**: Sticky-Metriken überwachen und anpassen