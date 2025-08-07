# Ansatz 1: Stateful Routing mit Context-Awareness

## Überblick
Dieser Ansatz implementiert ein intelligentes Routing-System, das den Konversationszustand verwaltet und dynamische Entscheidungen basierend auf dem aktuellen Workflow-Status trifft.

## Kernkonzept
Der Router erhält einen **Conversation State**, der den aktuellen Workflow-Modus trackt (`booking_in_progress`, `idle`, `knowledge_query`, etc.). Bei aktivem Workflow benötigt der Router deutlich höhere Confidence-Werte, um zu einem anderen Agent zu wechseln.

## Hauptkomponenten

### 1. ConversationStatePlugin
- **Zweck**: Verwaltet den kompletten Konversationszustand pro Session
- **Funktionen**:
  - Trackt Workflow-Stadium (Idle, BookingInProgress, SupportInProgress, etc.)
  - Speichert Konversationshistorie
  - Zählt Turns pro Agent und gesamt
  - Berechnet dynamische Confidence-Schwellenwerte
  - Verwaltet Kontext-Daten (z.B. Buchungsdetails)

### 2. StatefulRouter
- **Zweck**: Trifft Routing-Entscheidungen basierend auf State und Confidence
- **Entscheidungslogik**:
  1. Evaluiert alle verfügbaren Agents
  2. Berechnet dynamischen Schwellenwert basierend auf:
     - Aktuellem Workflow-Stadium
     - Anzahl der Turns mit aktuellem Agent
     - Gesprächslänge
  3. Gewährt Bonus für aktuellen Agent bei aktivem Workflow
  4. Wechselt nur bei signifikant besserer Alternative

### 3. StatefulRoutingOrchestrator
- **Zweck**: Koordiniert alle Komponenten
- **Aufgaben**:
  - Verwaltet Sessions
  - Koordiniert Router, State-Plugin und Agents
  - Protokolliert Routing-Entscheidungen
  - Sammelt Session-Statistiken

## Dynamische Schwellenwerte

### Basis-Schwellenwert: 0.6

### Modifikatoren:
- **BookingInProgress**: +0.2 (Schwellenwert: 0.8)
- **SupportInProgress**: +0.15 (Schwellenwert: 0.75)
- **Lange Konversation** (>2 Turns): +0.05 pro Turn (max +0.2)

### Beispiel:
```
Benutzer startet Flugbuchung
→ WorkflowStage = BookingInProgress
→ Basis-Schwellenwert: 0.6 + 0.2 = 0.8
→ Nach 3 Turns: 0.8 + 0.15 = 0.95

Neuer Agent muss mindestens 0.95 Confidence haben, um zu übernehmen!
```

## Workflow-Blocking
Bei kritischen Workflow-Phasen (z.B. Buchungsbestätigung) kann der Wechsel komplett blockiert werden:
```csharp
if (state.Context["booking_stage"] == "confirmation")
{
    // Kein Wechsel erlaubt!
    return false;
}
```

## Vorteile
✅ **Workflow-Kontinuität**: Unterbrechungen während kritischer Prozesse werden vermieden  
✅ **Intelligent**: Passt sich dynamisch an Konversationsverlauf an  
✅ **Flexibel**: Erlaubt Wechsel bei klarem Bedarf  
✅ **Transparent**: Alle Entscheidungen sind nachvollziehbar  

## Nachteile
❌ **Komplexität**: Viele Parameter müssen abgestimmt werden  
❌ **Starrheit**: Kann zu lange bei ungeeignetem Agent bleiben  
❌ **Konfigurationsaufwand**: Schwellenwerte müssen für Use-Case angepasst werden  

## Beispiel-Konversation
```
👤: Ich möchte einen Flug nach Berlin buchen
🤖 BookingAgent: Gerne helfe ich Ihnen bei der Flugbuchung nach Berlin...
[Schwellenwert erhöht auf 0.8]

👤: Wann fliegt die nächste Maschine?
🤖 BookingAgent: Ich zeige Ihnen die verfügbaren Flüge...
[Agent behält Kontrolle, Schwellenwert jetzt 0.85]

👤: Was ist die Hauptstadt von Frankreich?
[KnowledgeAgent Score: 0.9, aber Schwellenwert ist 0.85]
🤖 BookingAgent: Möchten Sie zuerst Ihre Buchung abschließen?
[BookingAgent behält Kontrolle trotz niedrigerer Eignung]

👤: Nein, vergessen wir die Buchung
[Workflow-Wechsel erlaubt]
🤖 KnowledgeAgent: Die Hauptstadt von Frankreich ist Paris.
```

## Konfigurationsoptionen
```csharp
// In StatefulRouter anpassbar:
baseThreshold = 0.6;           // Basis-Confidence
bookingBonus = 0.3;            // Bonus bei Buchung
supportBonus = 0.2;            // Bonus bei Support
turnBonus = 0.05;              // Bonus pro Turn
maxThreshold = 0.95;           // Maximaler Schwellenwert
```

## Verwendung
```csharp
var orchestrator = new StatefulRoutingOrchestrator(kernel, loggerFactory);
var sessionId = orchestrator.StartNewSession();

var (agentName, response, routingInfo) = 
    await orchestrator.ProcessMessageAsync(sessionId, userInput);
```