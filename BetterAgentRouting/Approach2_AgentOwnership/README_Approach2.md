# Ansatz 2: Agent-Ownership Pattern

## Überblick
Bei diesem Ansatz entscheiden die Agents selbst, wann sie "fertig" sind und die Kontrolle abgeben möchten. Der Router respektiert diese Entscheidungen und fungiert primär als Koordinator.

## Kernkonzept
Jeder Agent implementiert das `ISelfManagedAgent` Interface und kann:
- Entscheiden, ob er die Kontrolle behalten möchte (`keep_control`)
- Einen Nachfolge-Agent vorschlagen
- Seine Ownership-Priorität angeben

## Hauptkomponenten

### 1. ISelfManagedAgent Interface
```csharp
public interface ISelfManagedAgent : IAgent
{
    Task<OwnershipDecision> AnalyzeOwnershipAsync(...);
    Task<string?> SuggestNextAgentAsync(...);
}
```

### 2. OwnershipDecision
Enthält die Selbstverwaltungs-Entscheidung eines Agents:
- **KeepControl**: Will der Agent die Kontrolle behalten?
- **Confidence**: Wie sicher ist sich der Agent?
- **Priority**: Wie wichtig ist es, dass der Agent die Kontrolle behält?
- **SuggestedNextAgent**: Welcher Agent sollte übernehmen?

### 3. Selbstverwaltete Agents

#### SelfManagedBookingAgent
- **Hohe Priorität** (10) bei Buchungsbestätigung
- **Mittlere Priorität** (5) während aktiver Buchung
- Analysiert Buchungsstatus und entscheidet autonom

#### SelfManagedSupportAgent
- **Priorität** basiert auf Eskalationsstufe (Level * 2)
- Behält Kontrolle bei ungelösten technischen Problemen
- Gibt ab, wenn Problem gelöst ist

#### SelfManagedKnowledgeAgent
- **Niedrige Priorität** (1) standardmäßig
- Gibt meist nach einer Antwort ab
- Behält Kontrolle nur bei direkten Folgefragen

### 4. AgentOwnershipRouter
**Entscheidungslogik**:
1. Fragt aktuellen Agent nach Ownership-Entscheidung
2. Respektiert Agent-Entscheidung wenn Confidence > 0.5
3. Folgt Agent-Vorschlägen für Nachfolger
4. Bei Unklarheit: Fragt alle Agents und wählt den mit höchstem Anspruch

**Auswahlkriterien** (in dieser Reihenfolge):
1. Agents die Kontrolle wollen (`KeepControl = true`)
2. Höchste Priorität
3. Höchste Suitability

## Ownership-Prioritäten

| Agent | Situation | Priorität |
|-------|-----------|-----------|
| BookingAgent | Buchungsbestätigung | 10 |
| BookingAgent | Aktive Buchung | 5 |
| SupportAgent | Eskalation Level 3 | 6 |
| SupportAgent | Eskalation Level 2 | 4 |
| SupportAgent | Eskalation Level 1 | 2 |
| KnowledgeAgent | Folgefrage | 3 |
| KnowledgeAgent | Standard | 1 |

## Vorteile
✅ **Autonomie**: Agents verwalten sich selbst  
✅ **Intelligente Übergabe**: Agents kennen ihren eigenen Status am besten  
✅ **Saubere Übergaben**: Explizite Vorschläge für Nachfolger  
✅ **Prioritätsbasiert**: Kritische Workflows haben Vorrang  

## Nachteile
❌ **Komplexe Agent-Logik**: Jeder Agent braucht Ownership-Intelligenz  
❌ **Koordinationsaufwand**: Agents müssen gut aufeinander abgestimmt sein  
❌ **Debugging**: Schwieriger nachzuvollziehen, warum ein Agent die Kontrolle behält  

## Beispiel-Konversation
```
👤: Ich möchte einen Flug buchen
🤖 BookingAgent: Gerne helfe ich bei der Flugbuchung...
[BookingAgent: KeepControl=true, Priority=5]

👤: Nach München bitte
🤖 BookingAgent: Ich habe folgende Flüge nach München...
[BookingAgent analysiert: Buchung läuft → KeepControl=true]

👤: Ich nehme den um 14:00 Uhr
🤖 BookingAgent: Perfekt! Bitte bestätigen Sie die Buchung...
[Priority erhöht auf 10 - kritische Phase!]

👤: Bestätigt
🤖 BookingAgent: Buchung erfolgreich! Kann ich noch etwas für Sie tun?
[BookingAgent: KeepControl=false, SuggestedNext=None]

👤: Was ist die Zeitzone in München?
[Router fragt alle Agents - KnowledgeAgent meldet sich]
🤖 KnowledgeAgent: München liegt in der Zeitzone MEZ (UTC+1)...
[KnowledgeAgent: KeepControl=false nach Antwort]
```

## Agent-Kommunikation
Agents können Nachfolger vorschlagen:
```csharp
// BookingAgent nach Abschluss:
if (userInput.Contains("support") || userInput.Contains("problem"))
{
    return "SupportAgent"; // Explizite Übergabe
}
```

## Metadaten
Jede Agent-Antwort enthält Ownership-Metadaten:
```json
{
  "ownership_confidence": 0.9,
  "ownership_reason": "Buchung in kritischer Phase",
  "ownership_priority": 10,
  "keep_control": true,
  "suggested_next": "None"
}
```

## Verwendung
```csharp
var orchestrator = new OwnershipOrchestrator(kernel, loggerFactory);
var sessionId = orchestrator.StartNewSession();

var (agentName, response, routingInfo, metadata) = 
    await orchestrator.ProcessMessageAsync(sessionId, userInput);

// Metadata enthält Ownership-Details
Console.WriteLine($"Keep Control: {metadata["keep_control"]}");
Console.WriteLine($"Priority: {metadata["ownership_priority"]}");
```

## Best Practices
1. **Klare Ownership-Regeln**: Jeder Agent sollte wissen, wann er abgeben muss
2. **Sinnvolle Prioritäten**: Kritische Workflows > Support > Wissen
3. **Explizite Übergaben**: Bei klaren Themenwechseln Nachfolger vorschlagen
4. **Fallback-Logik**: Router entscheidet bei Unklarheit