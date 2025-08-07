using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.Extensions.Logging;
using System.ComponentModel;
using AgentRouterTest.Common.Models;

namespace AgentRouterTest.Approach3_StickySessions;

/// <summary>
/// Plugin zur Erkennung von Exit-Signalen in Benutzeranfragen
/// Identifiziert, wann ein Benutzer den aktuellen Workflow verlassen möchte
/// </summary>
public class ExitDetectionPlugin
{
    private readonly Kernel _kernel;
    private readonly IChatCompletionService _chatService;
    private readonly ILogger<ExitDetectionPlugin> _logger;
    
    // Exit-Signal Patterns
    private readonly string[] _explicitExitPhrases = 
    {
        "das war's", "danke das war alles", "fertig", "erledigt", 
        "andere frage", "noch etwas", "themenwechsel", "was anderes",
        "abbrechen", "stopp", "beenden", "vergiss es", "egal",
        "ich möchte lieber", "stattdessen", "wechseln zu"
    };
    
    private readonly string[] _completionIndicators = 
    {
        "vielen dank", "perfekt", "super danke", "alles klar",
        "verstanden", "hat geholfen", "problem gelöst"
    };
    
    public ExitDetectionPlugin(Kernel kernel, ILoggerFactory? loggerFactory = null)
    {
        _kernel = kernel;
        _chatService = kernel.GetRequiredService<IChatCompletionService>();
        _logger = loggerFactory?.CreateLogger<ExitDetectionPlugin>() 
                  ?? new LoggerFactory().CreateLogger<ExitDetectionPlugin>();
    }
    
    /// <summary>
    /// Analysiert, ob ein Exit-Signal vorliegt
    /// </summary>
    [KernelFunction("DetectExitIntent")]
    [Description("Detects if user wants to exit current workflow")]
    public async Task<ExitDetectionResult> DetectExitIntentAsync(
        string userInput, 
        ConversationState conversationState)
    {
        _logger.LogDebug($"Analysiere Exit-Intent für: {userInput}");
        
        var result = new ExitDetectionResult();
        
        // 1. Prüfe explizite Exit-Phrasen
        var explicitExit = CheckExplicitExitPhrases(userInput);
        if (explicitExit.detected)
        {
            result.ExitDetected = true;
            result.ExitType = ExitType.Explicit;
            result.Confidence = 0.95;
            result.Reason = $"Explizites Exit-Signal erkannt: '{explicitExit.phrase}'";
            _logger.LogInformation($"Expliziter Exit erkannt: {explicitExit.phrase}");
            return result;
        }
        
        // 2. Prüfe Completion-Indikatoren
        var completion = CheckCompletionIndicators(userInput);
        if (completion.detected)
        {
            result.ExitDetected = true;
            result.ExitType = ExitType.Completion;
            result.Confidence = 0.8;
            result.Reason = $"Workflow-Abschluss erkannt: '{completion.phrase}'";
            _logger.LogInformation($"Completion erkannt: {completion.phrase}");
            return result;
        }
        
        // 3. KI-basierte Analyse für subtilere Exit-Signale
        var aiAnalysis = await AnalyzeWithAIAsync(userInput, conversationState);
        if (aiAnalysis.ExitDetected)
        {
            return aiAnalysis;
        }
        
        // 4. Kontext-basierte Analyse
        var contextAnalysis = AnalyzeContext(userInput, conversationState);
        if (contextAnalysis.ExitDetected)
        {
            return contextAnalysis;
        }
        
        // Kein Exit erkannt
        result.ExitDetected = false;
        result.Confidence = 0.1;
        result.Reason = "Kein Exit-Signal erkannt, Workflow kann fortgesetzt werden";
        
        return result;
    }
    
    /// <summary>
    /// Prüft, ob eine Sticky Session aufgehoben werden sollte
    /// </summary>
    [KernelFunction("ShouldBreakStickiness")]
    [Description("Determines if sticky session should be broken")]
    public async Task<bool> ShouldBreakStickinessAsync(
        string userInput,
        ConversationState conversationState,
        double currentAgentConfidence)
    {
        // Exit-Intent prüfen
        var exitResult = await DetectExitIntentAsync(userInput, conversationState);
        
        if (exitResult.ExitDetected && exitResult.Confidence > 0.7)
        {
            _logger.LogInformation($"Sticky Session wird aufgehoben: {exitResult.Reason}");
            return true;
        }
        
        // Bei sehr niedriger Agent-Confidence und hohem Turn-Count
        if (currentAgentConfidence < 0.2 && conversationState.CurrentAgentTurnCount > 5)
        {
            _logger.LogInformation("Sticky Session aufgehoben: Agent-Confidence zu niedrig nach vielen Turns");
            return true;
        }
        
        // Bei Stagnation (gleiche Fragen wiederholen sich)
        if (IsConversationStagnant(conversationState))
        {
            _logger.LogInformation("Sticky Session aufgehoben: Konversation stagniert");
            return true;
        }
        
        return false;
    }
    
    /// <summary>
    /// Analysiert Themenwechsel
    /// </summary>
    [KernelFunction("DetectTopicChange")]
    [Description("Detects if user is changing topics")]
    public async Task<TopicChangeResult> DetectTopicChangeAsync(
        string userInput,
        ConversationState conversationState)
    {
        if (conversationState.History.Count == 0)
        {
            return new TopicChangeResult { TopicChanged = false };
        }
        
        var lastTopic = ExtractTopic(conversationState.History.Last().UserInput);
        var currentTopic = ExtractTopic(userInput);
        
        var prompt = $@"
Vergleiche diese zwei Themen und entscheide, ob ein Themenwechsel vorliegt:
Letztes Thema: {lastTopic}
Aktuelles Thema: {currentTopic}
Letzte Anfrage: {conversationState.History.Last().UserInput}
Aktuelle Anfrage: {userInput}

Antworte im Format: CHANGED|CONFIDENCE|NEW_TOPIC
Beispiel: JA|0.9|Hotelbuchung";

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("Du bist ein Topic-Change-Detector. Antworte nur im angegebenen Format.");
        chatHistory.AddUserMessage(prompt);
        
        var response = await _chatService.GetChatMessageContentAsync(chatHistory);
        var parts = response.Content?.Split('|') ?? new[] { "NEIN", "0.5", "" };
        
        return new TopicChangeResult
        {
            TopicChanged = parts[0].Trim().ToUpper() == "JA",
            Confidence = double.TryParse(parts[1], out var conf) ? conf : 0.5,
            NewTopic = parts.Length > 2 ? parts[2] : "",
            PreviousTopic = lastTopic
        };
    }
    
    private (bool detected, string phrase) CheckExplicitExitPhrases(string userInput)
    {
        var input = userInput.ToLower();
        foreach (var phrase in _explicitExitPhrases)
        {
            if (input.Contains(phrase))
            {
                return (true, phrase);
            }
        }
        return (false, "");
    }
    
    private (bool detected, string phrase) CheckCompletionIndicators(string userInput)
    {
        var input = userInput.ToLower();
        foreach (var phrase in _completionIndicators)
        {
            if (input.Contains(phrase))
            {
                return (true, phrase);
            }
        }
        return (false, "");
    }
    
    private async Task<ExitDetectionResult> AnalyzeWithAIAsync(
        string userInput, 
        ConversationState conversationState)
    {
        var prompt = $@"
Analysiere, ob der Benutzer den aktuellen Workflow verlassen möchte.

Aktueller Workflow: {conversationState.WorkflowStage}
Aktueller Agent: {conversationState.CurrentAgent}
Letzte Antwort: {conversationState.History.LastOrDefault()?.AgentResponse ?? "Keine"}
Benutzeranfrage: {userInput}

Bewerte:
1. Will der Benutzer den Workflow verlassen? (JA/NEIN)
2. Confidence (0-1)
3. Exit-Typ (EXPLICIT/COMPLETION/TOPIC_CHANGE/FRUSTRATION)
4. Begründung

Format: DECISION|CONFIDENCE|TYPE|REASON";

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("Du bist ein Exit-Intent-Analyzer. Antworte nur im angegebenen Format.");
        chatHistory.AddUserMessage(prompt);
        
        var response = await _chatService.GetChatMessageContentAsync(chatHistory);
        var parts = response.Content?.Split('|') ?? new[] { "NEIN", "0.3", "NONE", "Kein Exit erkannt" };
        
        var exitDetected = parts[0].Trim().ToUpper() == "JA";
        var confidence = double.TryParse(parts[1], out var conf) ? conf : 0.3;
        
        if (!exitDetected || confidence < 0.6)
        {
            return new ExitDetectionResult { ExitDetected = false };
        }
        
        return new ExitDetectionResult
        {
            ExitDetected = true,
            Confidence = confidence,
            ExitType = ParseExitType(parts[2]),
            Reason = parts.Length > 3 ? parts[3] : "KI-basierte Exit-Erkennung"
        };
    }
    
    private ExitDetectionResult AnalyzeContext(string userInput, ConversationState state)
    {
        // Frustration erkennen (viele Turns ohne Fortschritt)
        if (state.CurrentAgentTurnCount > 7 && 
            state.WorkflowStage != WorkflowStage.Completed)
        {
            return new ExitDetectionResult
            {
                ExitDetected = true,
                ExitType = ExitType.Frustration,
                Confidence = 0.7,
                Reason = "Mögliche Frustration nach vielen Turns ohne Abschluss"
            };
        }
        
        // Workflow natürlich abgeschlossen
        if (state.WorkflowStage == WorkflowStage.Completed)
        {
            return new ExitDetectionResult
            {
                ExitDetected = true,
                ExitType = ExitType.Completion,
                Confidence = 0.9,
                Reason = "Workflow wurde erfolgreich abgeschlossen"
            };
        }
        
        return new ExitDetectionResult { ExitDetected = false };
    }
    
    private bool IsConversationStagnant(ConversationState state)
    {
        if (state.History.Count < 4)
            return false;
        
        // Prüfe ob sich Fragen wiederholen
        var recentInputs = state.History.TakeLast(4).Select(h => h.UserInput.ToLower()).ToList();
        var uniqueInputs = recentInputs.Distinct().Count();
        
        return uniqueInputs <= 2; // Weniger als 3 unique Inputs in letzten 4 Turns
    }
    
    private string ExtractTopic(string input)
    {
        // Vereinfachte Topic-Extraktion
        if (input.ToLower().Contains("flug") || input.ToLower().Contains("hotel"))
            return "Buchung";
        if (input.ToLower().Contains("problem") || input.ToLower().Contains("fehler"))
            return "Support";
        if (input.ToLower().Contains("was ist") || input.ToLower().Contains("erkläre"))
            return "Wissen";
        
        return "Allgemein";
    }
    
    private ExitType ParseExitType(string type)
    {
        return type.ToUpper() switch
        {
            "EXPLICIT" => ExitType.Explicit,
            "COMPLETION" => ExitType.Completion,
            "TOPIC_CHANGE" => ExitType.TopicChange,
            "FRUSTRATION" => ExitType.Frustration,
            _ => ExitType.None
        };
    }
}

/// <summary>
/// Ergebnis der Exit-Detection
/// </summary>
public class ExitDetectionResult
{
    public bool ExitDetected { get; set; }
    public double Confidence { get; set; }
    public ExitType ExitType { get; set; }
    public string Reason { get; set; } = string.Empty;
}

/// <summary>
/// Ergebnis der Themenwechsel-Erkennung
/// </summary>
public class TopicChangeResult
{
    public bool TopicChanged { get; set; }
    public double Confidence { get; set; }
    public string NewTopic { get; set; } = string.Empty;
    public string PreviousTopic { get; set; } = string.Empty;
}

/// <summary>
/// Typ des Exit-Signals
/// </summary>
public enum ExitType
{
    None,
    Explicit,       // Explizite Exit-Anfrage
    Completion,     // Workflow abgeschlossen
    TopicChange,    // Themenwechsel
    Frustration     // Frustration/Stagnation
}