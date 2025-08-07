using Microsoft.SemanticKernel;
using Microsoft.Extensions.Logging;
using AgentRouterTest.Common.Interfaces;
using AgentRouterTest.Common.Models;
using System.Diagnostics;

namespace AgentRouterTest.Approach3_StickySessions;

/// <summary>
/// Router-Implementierung mit Sticky Sessions
/// Bleibt beim aktuellen Agent, außer bei expliziten Exit-Signalen
/// </summary>
public class StickySessionRouter : IRouter
{
    private readonly Kernel _kernel;
    private readonly ExitDetectionPlugin _exitDetector;
    private readonly ILogger<StickySessionRouter> _logger;
    
    // Konfiguration für Sticky-Verhalten
    private readonly double _minConfidenceToStick = 0.3; // Minimale Confidence um bei Agent zu bleiben
    private readonly double _exitThreshold = 0.6; // Exit-Confidence Schwellenwert
    private readonly int _maxTurnsBeforePrompt = 10; // Nach X Turns nachfragen
    
    public string ApproachName => "Sticky Sessions mit Exit-Detection";
    
    public StickySessionRouter(Kernel kernel, ExitDetectionPlugin exitDetector, ILoggerFactory? loggerFactory = null)
    {
        _kernel = kernel;
        _exitDetector = exitDetector;
        _logger = loggerFactory?.CreateLogger<StickySessionRouter>() 
                  ?? new LoggerFactory().CreateLogger<StickySessionRouter>();
    }
    
    public async Task<RoutingResult> RouteAsync(
        string userInput, 
        ConversationState conversationState, 
        List<IAgent> availableAgents)
    {
        var stopwatch = Stopwatch.StartNew();
        
        _logger.LogInformation($"[STICKY ROUTER] Starte Routing für: {userInput.Substring(0, Math.Min(50, userInput.Length))}...");
        _logger.LogInformation($"[STICKY ROUTER] Sticky Session - Current: {conversationState.CurrentAgent ?? "None"}, " +
                              $"Turns: {conversationState.CurrentAgentTurnCount}");
        
        var result = new RoutingResult();
        
        // Wenn ein Agent aktiv ist, prüfe Sticky-Verhalten
        if (!string.IsNullOrEmpty(conversationState.CurrentAgent))
        {
            var currentAgent = availableAgents.FirstOrDefault(a => a.Name == conversationState.CurrentAgent);
            
            if (currentAgent != null)
            {
                // Prüfe Exit-Intent
                var exitDetection = await _exitDetector.DetectExitIntentAsync(userInput, conversationState);
                _logger.LogInformation($"[STICKY ROUTER] Exit-Detection: {exitDetection.ExitType}, " +
                                      $"Confidence: {exitDetection.Confidence:F2}");
                
                // Evaluiere aktuelle Agent-Eignung
                var currentAgentScore = await currentAgent.EvaluateSuitabilityAsync(userInput, conversationState);
                
                // Entscheide ob Sticky Session aufgehoben werden soll
                bool shouldBreakSticky = await ShouldBreakStickySession(
                    exitDetection, 
                    currentAgentScore, 
                    conversationState);
                
                if (!shouldBreakSticky)
                {
                    // Bleibe beim aktuellen Agent (Sticky!)
                    result.SelectedAgent = currentAgent;
                    result.Confidence = Math.Max(currentAgentScore, _minConfidenceToStick);
                    result.AgentChanged = false;
                    result.Reason = BuildStickyReason(exitDetection, currentAgentScore, conversationState);
                    
                    _logger.LogInformation($"[STICKY ROUTER] STICKY: Bleibe bei {currentAgent.Name}");
                    
                    // Füge Info über Exit-Detection hinzu
                    if (exitDetection.ExitDetected)
                    {
                        result.Reason += $" (Exit erkannt aber Confidence zu niedrig: {exitDetection.Confidence:F2})";
                    }
                    
                    return result;
                }
                
                // Exit erkannt - Sticky Session wird aufgehoben
                _logger.LogInformation($"[STICKY ROUTER] Sticky Session aufgehoben: {exitDetection.Reason}");
                
                // Prüfe Themenwechsel für besseres Routing
                var topicChange = await _exitDetector.DetectTopicChangeAsync(userInput, conversationState);
                if (topicChange.TopicChanged)
                {
                    _logger.LogInformation($"[STICKY ROUTER] Themenwechsel erkannt: {topicChange.PreviousTopic} → {topicChange.NewTopic}");
                }
            }
        }
        
        // Kein Sticky-Verhalten oder Session aufgehoben - normales Routing
        var agentScores = new Dictionary<IAgent, double>();
        foreach (var agent in availableAgents)
        {
            var score = await agent.EvaluateSuitabilityAsync(userInput, conversationState);
            agentScores[agent] = score;
            _logger.LogDebug($"[STICKY ROUTER] {agent.Name} Score: {score:F2}");
        }
        
        // Wähle besten Agent
        var bestAgent = agentScores.OrderByDescending(kvp => kvp.Value).First();
        
        result.SelectedAgent = bestAgent.Key;
        result.Confidence = bestAgent.Value;
        result.AgentChanged = bestAgent.Key.Name != conversationState.CurrentAgent;
        result.PreviousAgent = conversationState.CurrentAgent;
        result.AlternativeAgents = agentScores.ToDictionary(kvp => kvp.Key.Name, kvp => kvp.Value);
        
        if (result.AgentChanged && !string.IsNullOrEmpty(conversationState.CurrentAgent))
        {
            result.Reason = $"Sticky Session beendet - Wechsel zu {bestAgent.Key.Name} (Score: {bestAgent.Value:F2})";
        }
        else if (string.IsNullOrEmpty(conversationState.CurrentAgent))
        {
            result.Reason = $"Initiale Agent-Wahl: {bestAgent.Key.Name} (Score: {bestAgent.Value:F2})";
        }
        else
        {
            result.Reason = $"Fortsetzung mit {bestAgent.Key.Name}";
        }
        
        _logger.LogInformation($"[STICKY ROUTER] Finale Entscheidung: {result.SelectedAgent?.Name}");
        
        stopwatch.Stop();
        result.RoutingTimeMs = stopwatch.ElapsedMilliseconds;
        _logger.LogInformation($"[STICKY ROUTER] Routing abgeschlossen in {result.RoutingTimeMs}ms");
        
        return result;
    }
    
    /// <summary>
    /// Entscheidet, ob die Sticky Session aufgehoben werden soll
    /// </summary>
    private async Task<bool> ShouldBreakStickySession(
        ExitDetectionResult exitDetection,
        double currentAgentScore,
        ConversationState state)
    {
        // 1. Expliziter Exit mit hoher Confidence
        if (exitDetection.ExitDetected && exitDetection.Confidence > _exitThreshold)
        {
            _logger.LogInformation($"[STICKY ROUTER] Break Sticky: Exit-Signal (Type: {exitDetection.ExitType}, Confidence: {exitDetection.Confidence:F2})");
            return true;
        }
        
        // 2. Agent komplett ungeeignet
        if (currentAgentScore < 0.1)
        {
            _logger.LogInformation($"[STICKY ROUTER] Break Sticky: Agent-Score zu niedrig ({currentAgentScore:F2})");
            return true;
        }
        
        // 3. Nutze Exit-Detector für erweiterte Prüfung
        var shouldBreak = await _exitDetector.ShouldBreakStickinessAsync(
            state.History.LastOrDefault()?.UserInput ?? "",
            state,
            currentAgentScore);
        
        if (shouldBreak)
        {
            _logger.LogInformation("[STICKY ROUTER] Break Sticky: Exit-Detector Empfehlung");
            return true;
        }
        
        // 4. Nach vielen Turns ohne Exit - Nutzer fragen
        if (state.CurrentAgentTurnCount >= _maxTurnsBeforePrompt)
        {
            _logger.LogWarning($"[STICKY ROUTER] {state.CurrentAgentTurnCount} Turns mit {state.CurrentAgent} - " +
                             "Empfehle Nachfrage beim Nutzer");
            // In echter Implementierung würde hier eine Nachfrage erfolgen
        }
        
        return false; // Sticky Session beibehalten
    }
    
    /// <summary>
    /// Erstellt Begründung für Sticky-Verhalten
    /// </summary>
    private string BuildStickyReason(
        ExitDetectionResult exitDetection,
        double agentScore,
        ConversationState state)
    {
        var reasons = new List<string>();
        
        reasons.Add($"Sticky Session aktiv mit {state.CurrentAgent}");
        
        if (state.WorkflowStage != WorkflowStage.Idle && state.WorkflowStage != WorkflowStage.Completed)
        {
            reasons.Add($"Workflow '{state.WorkflowStage}' läuft");
        }
        
        if (agentScore > 0.5)
        {
            reasons.Add($"Agent weiterhin geeignet (Score: {agentScore:F2})");
        }
        else if (agentScore > _minConfidenceToStick)
        {
            reasons.Add($"Agent akzeptabel (Score: {agentScore:F2})");
        }
        
        if (!exitDetection.ExitDetected)
        {
            reasons.Add("Kein Exit-Signal erkannt");
        }
        
        reasons.Add($"Turn {state.CurrentAgentTurnCount} mit Agent");
        
        return string.Join(", ", reasons);
    }
}