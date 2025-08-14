---
title: Multi-Agent-Routing-Lösungen - Wer kommt wann dran? 3 clevere Ansätze gegen vergessliche Chatbots
meta-title: Multi-Agent-Routing - Wer kommt wann dran? 3 Lösungen mit Semantic Kernel
meta-description: Erfahre, wie du mit drei Routing-Lösungen (Stateful, Agent-Ownership, Sticky Sessions) entscheidest, welcher Agent wann dran ist. Mit praktischen Code-Beispielen in C# und Microsoft Semantic Kernel.
summary: Multi-Agent-Systeme leiden oft unter inkonsistenten Router-Entscheidungen - wer kommt wann dran? Dieser Artikel zeigt drei praxiserprobte Routing-Lösungen mit Microsoft Semantic Kernel - Stateful Routing mit dynamischen Schwellenwerten, Agent-Ownership für selbstverwaltete Kontrolle und Sticky Sessions für maximale Stabilität.
author: Christoph Kind
layout: Article
project: christophkind.de
category: blog
date: 2024-09-26T14:00:00.000Z
slug: multi-agent-routing-wer-kommt-wann-dran
syncwithcontentdb: true
main-image: title.png
last_updated: 14.08.2025, 17:01:43
---
# Wenn Chatbots plötzlich vergesslich werden: Drei clevere Lösungen für konsistente KI-Workflows

## Das nervige Problem, das wir alle kennen

Kennst du das? Du bist gerade dabei, einen Flug zu buchen. Der Chatbot hat schon dein Reiseziel und die Daten – und dann fragst du nebenbei "Ach, wie wird eigentlich das Wetter in Paris?" Zack! Der Bot vergisst komplett die Buchung und schwafelt über Wettervorhersagen. Super nervig!

Dieses Problem plagt viele Entwickler von Multi-Agent-Systemen. Der Router – quasi der Verkehrspolizist, der entscheidet, welcher spezialisierte Agent (Buchung, Support, Wissen) gerade dran ist – trifft oft inkonsistente Entscheidungen. Besonders bei längeren Gesprächen führt das zu chaotischen Kontextwechseln.

### Warum Multi-Agent-Systeme eigentlich genial sind

Stell dir vor, du hast nicht einen Alleskönner-Bot, sondern ein ganzes Team von Spezialisten. Der BookingAgent ist der Reise-Experte, der SupportAgent der Tech-Guru, und der KnowledgeAgent die wandelnde Wikipedia. Klingt super, oder? 

Das Problem: Irgendwer muss entscheiden, wer gerade sprechen darf. Hier kommt der Router ins Spiel – der Dirigent des Orchesters. Dumm nur, dass traditionelle Router ziemlich vergesslich sind. Sie schauen nur auf die aktuelle Nachricht und vergessen alles, was davor war. Bei "Wie spät ist es?" kein Problem. Bei einer mehrstufigen Hotelbuchung? Katastrophe!

## Die Challenge: Flexibel bleiben ohne durchzudrehen

Der Knackpunkt ist die Balance. Ein Router muss schlau genug sein, echte Themenwechsel zu erkennen ("Vergiss die Buchung, ich hab ein dringendes Problem!"), aber nicht bei jeder kleinen Nebenfrage gleich den kompletten Kontext über Bord werfen.

Schauen wir uns mal ein typisches Chaos-Szenario an:
```
User: "Ich möchte nach Paris fliegen"
Bot: [BookingAgent] "Cool! Wann soll's denn losgehen?"
User: "Nächste Woche, aber moment, regnet's da gerade?"
Bot: [KnowledgeAgent] "In Paris sind es aktuell 15 Grad..."  // FAIL! Buchung vergessen!
```

Frustrierend? Absolut! Aber zum Glück gibt's drei clevere Lösungsansätze, die mit Microsoft Semantic Kernel umgesetzt wurden. Semantic Kernel ist dabei wie ein schweizer Taschenmesser für KI-Entwickler – es macht komplexe Agent-Systeme erstaunlich einfach.

## Lösung 1: Der Router mit Gedächtnis (Stateful Routing)

Der erste Ansatz gibt dem Router ein Gedächtnis. Er merkt sich, was gerade läuft und passt seine Entscheidungsschwellen dynamisch an. Stell dir vor, der Router ist wie ein aufmerksamer Kellner, der weiß, dass du gerade beim Hauptgang bist und dich nicht mit der Dessertkarte nervt.

```csharp
public class StatefulRouter : IRouter
{
    private double CalculateCurrentAgentBonus(ConversationState state)
    {
        double bonus = 0;
        
        // Je nach Workflow gibt's unterschiedliche Boni
        switch (state.WorkflowStage)
        {
            case WorkflowStage.BookingInProgress:
                bonus += 0.3; // "Ey, wir buchen gerade! Nicht stören!"
                break;
            case WorkflowStage.SupportInProgress:
                bonus += 0.2; // "Support läuft, bitte warten"
                break;
            case WorkflowStage.KnowledgeQuery:
                bonus += 0.1; // "Nur 'ne kleine Frage"
                break;
        }
        
        // Je länger das Gespräch, desto "klebriger" wird's
        if (state.CurrentAgentTurnCount > 0)
        {
            bonus += Math.Min(0.2, state.CurrentAgentTurnCount * 0.05);
        }
        
        return bonus;
    }
}
```

**Die Idee dahinter**: Je tiefer du in einem Workflow steckst, desto schwerer wird's für andere Agents, dich rauszureißen. Bei einer aktiven Buchung bekommt der BookingAgent einen fetten Bonus von 0.3. Das heißt: Ein anderer Agent muss schon 30% besser passen, um übernehmen zu dürfen.

Das System merkt sich auch, wo du gerade stehst:

```csharp
public class ConversationStatePlugin
{
    private readonly Dictionary<string, ConversationState> _sessions = new();
    
    public double CalculateDynamicThreshold(string sessionId)
    {
        var state = GetState(sessionId);
        double threshold = 0.3; // Start-Schwelle
        
        // Bei aktiver Buchung wird's schwieriger zu wechseln
        if (state.WorkflowStage == WorkflowStage.BookingInProgress)
        {
            threshold += 0.2 + (state.CurrentAgentTurnCount * 0.05);
        }
        
        return Math.Min(threshold, 0.8); // Maximal 80% Hürde
    }
}
```

Zusätzlich trackt das System intelligent, was gerade abgeht:

```csharp
private void UpdateWorkflowStage(IAgent selectedAgent, ConversationState state)
{
    if (state.CurrentAgent != selectedAgent.Name || state.WorkflowStage == WorkflowStage.Idle)
    {
        switch (selectedAgent.Name)
        {
            case "BookingAgent":
                if (state.WorkflowStage != WorkflowStage.BookingInProgress)
                {
                    state.WorkflowStage = WorkflowStage.BookingInProgress;
                    state.Context["booking_start_time"] = DateTime.Now;
                    state.Context["booking_stage"] = "initial";
                }
                break;
            // Andere Agents folgen dem gleichen Muster
        }
    }
}
```

**Was ist cool daran?** Super flexibel und passt sich automatisch an verschiedene Situationen an.
**Was nervt?** Du musst die Schwellenwerte gut einstellen, sonst klebt der Bot zu sehr oder springt zu wild rum.

## Lösung 2: Die Agents nehmen das Heft in die Hand (Agent-Ownership)

Der zweite Ansatz ist radikal anders: Hier entscheiden die Agents selbst, wann Schluss ist. Jeder Agent wird zum selbstbewussten Experten, der genau weiß, wann seine Arbeit getan ist.

```csharp
public interface ISelfManagedAgent : IAgent
{
    Task<OwnershipDecision> AnalyzeOwnershipAsync(
        string userInput, 
        string lastResponse, 
        ConversationState state);
    
    Task<string?> SuggestNextAgentAsync(
        string userInput, 
        ConversationState state);
}
```

Der BookingAgent checkt nach jeder Nachricht, ob er weitermachen sollte:

```csharp
public async Task<OwnershipDecision> AnalyzeOwnershipAsync(
    string userInput, 
    string lastResponse, 
    ConversationState state)
{
    // Kritische Phase? Dann lass mich bloß nicht los!
    if (state.Context.GetValueOrDefault("booking_stage")?.ToString() == "confirmation")
    {
        return new OwnershipDecision
        {
            KeepControl = true,
            Priority = 10, // SEHR wichtig!
            Reason = "Alter, wir sind bei der Bezahlung! Nicht stören!"
        };
    }
    
    // Ansonsten: Lass die KI entscheiden
    var prompt = $@"
    Check mal: Soll der Buchungsagent weitermachen?
    Status: {state.Context.GetValueOrDefault("booking_stage", "none")}
    User sagt: {userInput}
    ";
    
    var decision = await AnalyzeWithLLM(prompt);
    return decision;
}
```

**Das Geniale**: Die Agents wissen selbst am besten, wann sie fertig sind. Bei kritischen Momenten (Zahlungsbestätigung!) können sie sogar die Kontrolle erzwingen.

Der Router respektiert diese Entscheidungen:

```csharp
public class OwnershipOrchestrator
{
    public async Task<AgentResponse> ProcessAsync(string userInput, ConversationState state)
    {
        if (state.CurrentAgent != null)
        {
            var currentAgent = GetAgent(state.CurrentAgent) as ISelfManagedAgent;
            var ownership = await currentAgent.AnalyzeOwnershipAsync(
                userInput, state.LastResponse, state);
            
            if (ownership.KeepControl && ownership.Priority > 5)
            {
                // Agent sagt: "Ich mach weiter!" - OK Boss!
                _logger.LogWarning($"Agent {state.CurrentAgent} übernimmt Kontrolle " +
                                  $"(Priority: {ownership.Priority})");
                return await currentAgent.ProcessAsync(userInput, state);
            }
        }
        // Nur wenn Agent nicht drauf besteht: Normal routen
        return await _router.RouteAndProcessAsync(userInput, state);
    }
}
```

**Was rockt?** Mega robust bei komplexen Workflows. Agents können ihr Fachwissen voll ausspielen.
**Was nervt?** Jeder Agent braucht eigene Entscheidungslogik. Kann komplex werden.

## Lösung 3: Kleben bis zum bitteren Ende (Sticky Sessions)

Der dritte Ansatz ist der simpelste: Der Router klebt wie Kaugummi am aktuellen Agent, bis du explizit "STOP!" rufst.

```csharp
public class StickySessionRouter : IRouter
{
    private readonly double _minConfidenceToStick = 0.3;
    private readonly double _exitThreshold = 0.6;
    
    public async Task<RoutingResult> RouteAsync(
        string userInput, 
        ConversationState conversationState, 
        List<IAgent> availableAgents)
    {
        if (!string.IsNullOrEmpty(conversationState.CurrentAgent))
        {
            // Check: Will der User wirklich weg?
            var exitDetection = await _exitDetector.DetectExitIntentAsync(
                userInput, conversationState);
            
            if (!exitDetection.ExitDetected || 
                exitDetection.Confidence < _exitThreshold)
            {
                // Nö, wir bleiben kleben!
                return new RoutingResult
                {
                    SelectedAgent = currentAgent,
                    AgentChanged = false,
                    Reason = "Sticky wie Honig - bleib ich bei!"
                };
            }
        }
        
        // Nur bei klarem Exit-Signal: Neuer Agent
        return await PerformNormalRouting(userInput, availableAgents);
    }
}
```

Das Exit-Detection-Plugin ist clever und checkt verschiedene Signale:

```csharp
public class ExitDetectionPlugin
{
    private readonly string[] _exitPhrases = {
        "das war's", "fertig", "danke", "andere frage",
        "themenwechsel", "abbrechen", "stop", "halt"
    };
    
    public async Task<ExitIntent> DetectExitIntentAsync(
        string userInput, ConversationState state)
    {
        // Quick-Check: Hat der User "Tschüss" gesagt?
        if (_exitPhrases.Any(phrase => 
            userInput.ToLower().Contains(phrase)))
        {
            return new ExitIntent 
            { 
                ExitDetected = true, 
                ExitType = ExitType.Explicit,
                Confidence = 0.9 
            };
        }
        
        // Oder komplett neues Thema?
        if (await DetectTopicChange(userInput, state))
        {
            return new ExitIntent 
            { 
                ExitDetected = true, 
                ExitType = ExitType.TopicChange,
                Confidence = 0.7 
            };
        }
        
        return new ExitIntent { ExitDetected = false };
    }
}
```

**Das Geile daran**: Super stabil und vorhersagbar. Keine Überraschungen!
**Der Haken**: Kann manchmal zu träge sein, wenn du wirklich wechseln willst.

## So sieht's in Action aus

Alle drei Ansätze kannst du in einer Console-App direkt ausprobieren:

```csharp
class Program
{
    static async Task Main(string[] args)
    {
        var kernel = CreateKernel(configuration);
        
        Console.WriteLine("Welchen Router willst du testen?");
        Console.WriteLine("1. Stateful (mit Gedächtnis)");
        Console.WriteLine("2. Ownership (Agents entscheiden)");
        Console.WriteLine("3. Sticky (klebt wie Pattex)");
        
        // Los geht's mit dem Chat!
        while (true)
        {
            var userInput = Console.ReadLine();
            var result = await orchestrator.ProcessAsync(userInput);
            
            Console.ForegroundColor = GetAgentColor(result.SelectedAgent);
            Console.WriteLine($"[{result.SelectedAgent}]: {result.Response}");
            Console.ResetColor();
        }
    }
}
```


## Real-World Szenarien: So reagieren die Router

Lass uns mal schauen, wie die verschiedenen Ansätze in der Praxis ticken:

### Szenario 1: Der ungeduldige Kunde

```
User: "Ich brauch 'nen Flug nach London, morgen früh"
Bot: [BookingAgent] "Alles klar! Welche Uhrzeit passt dir?"
User: "Moment, geht meine Kreditkarte überhaupt im Ausland?"
```

**Stateful**: "Bleib cool, wir buchen gerade" (Bonus: 0.35)
**Ownership**: "Zahlungsfrage gehört zur Buchung, ich mach weiter"
**Sticky**: "Kein Exit-Signal, ich bleib dran"

Alle treffen die richtige Entscheidung – aber aus unterschiedlichen Gründen!

### Szenario 2: Echter Alarm

```
User: "Buch mir 'nen Flug"
Bot: [BookingAgent] "Gerne! Wohin soll's gehen?"
User: "STOPP! Mein Account wurde gehackt! Hilfe!"
```

**Stateful**: Support-Score explodiert (0.95) → Wechsel!
**Ownership**: "Okay, das ist wichtiger, geb ab an Support"
**Sticky**: "STOPP" erkannt → Exit granted!

Hier zeigt Sticky Sessions seine Stärke bei klaren Signalen.

## Performance-Tuning: Damit's auch flott läuft

Was viele vergessen: Jeder Router-Call kostet Zeit. Du evaluierst alle Agents, machst LLM-Calls, managest States... Das summiert sich!

Hier ein paar Tricks für mehr Speed:

```csharp
public class OptimizedRouter
{
    private readonly MemoryCache _scoreCache = new MemoryCache(new MemoryCacheOptions());
    
    public async Task<double> GetCachedSuitabilityScore(IAgent agent, string input)
    {
        var cacheKey = $"{agent.Name}:{input.GetHashCode()}";
        
        // Schon mal gecheckt? Dann nimm's aus dem Cache!
        if (_scoreCache.TryGetValue(cacheKey, out double cachedScore))
        {
            return cachedScore;
        }
        
        var score = await agent.EvaluateSuitabilityAsync(input);
        _scoreCache.Set(cacheKey, score, TimeSpan.FromMinutes(5));
        return score;
    }
}
```

**Pro-Tipps für Speed:**
- Cache identische Anfragen
- Quick-Checks (Keywords) vor teuren LLM-Calls
- Batch-Processing wo möglich
- Kleinere Modelle (GPT-4o-mini) für Routing

## Monitoring: Was läuft da eigentlich ab?

Bei komplexen Systemen musst du wissen, was abgeht:

```csharp
public class MonitoredRouter : IRouter
{
    private readonly IMetrics _metrics;
    
    public async Task<RoutingResult> RouteAsync(string input, ConversationState state)
    {
        using var activity = Activity.StartActivity("Router.Route");
        activity?.SetTag("current_agent", state.CurrentAgent);
        activity?.SetTag("workflow_stage", state.WorkflowStage);
        
        var stopwatch = Stopwatch.StartNew();
        var result = await InternalRoute(input, state);
        
        _metrics.RecordRoutingTime(stopwatch.ElapsedMilliseconds);
        _metrics.RecordAgentChange(result.AgentChanged);
        
        if (result.AgentChanged)
        {
            _logger.LogInformation(
                "Agent-Wechsel: {Previous} → {New} (Warum: {Reason})",
                result.PreviousAgent, result.SelectedAgent.Name, result.Reason);
        }
        
        return result;
    }
}
```

**Was du tracken solltest:**
- Wie lange dauern Routing-Entscheidungen?
- Wie oft wechseln Agents?
- Werden Workflows erfolgreich beendet?
- Müssen User Sachen wiederholen?

## Welcher Router für welchen Job?

Nach viel Testing und ein paar grauen Haaren später:

- **Stateful Routing**: Top für dynamische Umgebungen (Online-Shops, wo User zwischen Suchen, Kaufen und Support hin und her springen)

- **Agent-Ownership**: Perfekt wenn's kritisch wird (Banking, Medizin – wo Fehler teuer sind)

- **Sticky Sessions**: Ideal für Support oder Lern-Plattformen (wo User eine Sache durchziehen wollen)

## Best Practices: Was wir gelernt haben

Nach Monaten des Tüftelns hier die wichtigsten Learnings:

### 1. Fang simpel an
Starte mit Sticky Sessions. Komplexität kommt von alleine, versprochen!

### 2. Teste mit echten Gesprächen
Synthetische Tests sind nett, aber nichts schlägt echte User-Dialoge:

```csharp
public class DialogueAnalyzer
{
    public async Task<RoutingInsights> AnalyzeConversations(List<Conversation> conversations)
    {
        var insights = new RoutingInsights();
        
        foreach (var conv in conversations)
        {
            // Wo sind User genervt abgesprungen?
            var unwantedSwitches = conv.Turns
                .Where(t => t.AgentChanged && t.UserSatisfaction < 3)
                .Count();
            
            // Was ging schief?
            if (!conv.WorkflowCompleted)
            {
                insights.FailurePatterns.Add(new Pattern
                {
                    LastAgent = conv.LastAgent,
                    FailurePoint = conv.LastTurn,
                    Context = conv.State
                });
            }
        }
        
        return insights;
    }
}
```

### 3. Hab einen Plan B
Wenn der Router unsicher ist, frag einfach nach:

```csharp
if (bestScore < 0.5) // Keine Ahnung wer zuständig ist
{
    return new AgentResponse
    {
        Content = "Hmm, ich bin nicht sicher, wobei ich helfen soll. " +
                 "Was möchtest du: 1) Etwas buchen 2) Support 3) Eine Info?",
        RequiresUserChoice = true
    };
}
```

### 4. Nutze Semantic Kernels Plugin-Power

Die Plugin-Architektur macht alles super modular:

```csharp
var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion(modelId: "gpt-4o-mini", apiKey: apiKey)
    .Build();

// Plugins reinhauen
kernel.Plugins.AddFromObject(new ConversationStatePlugin(), "state");
kernel.Plugins.AddFromObject(new ExitDetectionPlugin(), "exit");
kernel.Plugins.AddFromObject(new RouterPlugin(), "router");

// Plugins können sich gegenseitig nutzen - cool!
var result = await kernel.InvokeAsync(
    "router", 
    "RouteWithContext",
    new KernelArguments 
    { 
        ["input"] = userInput,
        ["state"] = await kernel.InvokeAsync("state", "GetState")
    });
```

## Fazit: Mix it Baby!

Real Talk: Die beste Lösung mischt alle drei Ansätze. Ein hybrider Router könnte:

1. **Sticky Sessions** als Basis nehmen (für Stabilität)
2. **Ownership-Entscheidungen** respektieren (für Intelligenz)
3. **Stateful Context** für Feintuning nutzen (für Flexibilität)

Microsoft Semantic Kernel macht's möglich, diese Ansätze sauber zu kombinieren. Die klare Trennung zwischen Router, Agents und State-Management ist Gold wert.

### Die Zukunft: Selbstlernende Router

Die nächste Stufe? Router, die aus Erfahrung lernen:

```csharp
public class LearningRouter : IRouter
{
    private readonly IRoutingHistoryStore _history;
    
    public async Task<RoutingResult> RouteAsync(string input, ConversationState state)
    {
        // Was hat früher in ähnlichen Situationen funktioniert?
        var similarCases = await _history.FindSimilarCases(input, state);
        var successfulPatterns = similarCases.Where(c => c.UserSatisfied);
        
        // Aha! Das hat geklappt, machen wir wieder so
        if (successfulPatterns.Any())
        {
            return await ApplyLearnedStrategy(successfulPatterns, input, state);
        }
        
        // Keine Ahnung? Standard-Routing
        return await StandardRoute(input, state);
    }
}
```

**Die wichtigste Lektion**: Es gibt keine Universallösung. Versteh deinen Use-Case, wähl den passenden Ansatz (oder mix sie), und deine User werden's dir danken.

Die Zukunft der KI liegt nicht in einem Uber-Bot, der alles kann, sondern in cleverer Orchestrierung von Spezialisten. Und der Router? Der ist der heimliche MVP, der dafür sorgt, dass alles smooth läuft – auch wenn der User mal total abdriftet. Respekt an alle Router da draußen!


## Code Repository

Den vollständigen Code zu allen drei Routing-Ansätzen findest du im GitHub-Repository:

**[https://github.com/ChristophKind/LargeLanguageModelsPublic](https://github.com/ChristophKind/LargeLanguageModelsPublic)**

Dort kannst du:
- Den kompletten Quellcode herunterladen und direkt ausprobieren
- Die vollständige Implementierung aller drei Router-Varianten studieren
- Die Beispiel-Agents (BookingAgent, SupportAgent, KnowledgeAgent) im Detail ansehen
- Eigene Erweiterungen und Anpassungen vornehmen

Das Repository enthält eine lauffähige .NET-Lösung mit Microsoft Semantic Kernel, die du als Basis für deine eigenen Multi-Agent-Systeme verwenden kannst.