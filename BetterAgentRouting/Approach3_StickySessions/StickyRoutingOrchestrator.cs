using Microsoft.SemanticKernel;
using Microsoft.Extensions.Logging;
using AgentRouterTest.Common.Interfaces;
using AgentRouterTest.Common.Models;
using AgentRouterTest.Common.BaseAgents;

namespace AgentRouterTest.Approach3_StickySessions;

/// <summary>
/// Orchestrator für Approach 3: Sticky Sessions mit Exit-Detection
/// Implementiert "klebriges" Routing-Verhalten mit expliziter Exit-Erkennung
/// </summary>
public class StickyRoutingOrchestrator
{
    private readonly Kernel _kernel;
    private readonly ExitDetectionPlugin _exitDetector;
    private readonly StickySessionRouter _router;
    private readonly List<IAgent> _agents;
    private readonly Dictionary<string, ConversationState> _sessions;
    private readonly ILogger<StickyRoutingOrchestrator> _logger;
    
    // Sticky Session Metriken
    private readonly Dictionary<string, StickySessionMetrics> _sessionMetrics;
    
    public StickyRoutingOrchestrator(Kernel kernel, ILoggerFactory? loggerFactory = null)
    {
        _kernel = kernel;
        _logger = loggerFactory?.CreateLogger<StickyRoutingOrchestrator>() 
                  ?? new LoggerFactory().CreateLogger<StickyRoutingOrchestrator>();
        
        // Initialisiere Exit-Detection Plugin
        _exitDetector = new ExitDetectionPlugin(kernel, loggerFactory);
        
        // Initialisiere Sticky Router
        _router = new StickySessionRouter(kernel, _exitDetector, loggerFactory);
        
        // Initialisiere Agents
        _agents = new List<IAgent>
        {
            new BookingAgent(kernel),
            new SupportAgent(kernel),
            new KnowledgeAgent(kernel)
        };
        
        _sessions = new Dictionary<string, ConversationState>();
        _sessionMetrics = new Dictionary<string, StickySessionMetrics>();
        
        _logger.LogInformation("Sticky Routing Orchestrator initialisiert");
    }
    
    /// <summary>
    /// Startet eine neue Sticky Session
    /// </summary>
    public string StartNewSession()
    {
        var sessionId = Guid.NewGuid().ToString();
        _sessions[sessionId] = new ConversationState { SessionId = sessionId };
        _sessionMetrics[sessionId] = new StickySessionMetrics { SessionId = sessionId };
        _logger.LogInformation($"Neue Sticky Session gestartet: {sessionId}");
        return sessionId;
    }
    
    /// <summary>
    /// Verarbeitet eine Nachricht mit Sticky Session Verhalten
    /// </summary>
    public async Task<StickySessionResponse> ProcessMessageAsync(string sessionId, string userInput)
    {
        _logger.LogInformation($"\n{new string('=', 80)}");
        _logger.LogInformation($"[STICKY SESSION] Verarbeite Nachricht für Session {sessionId}");
        
        // Hole oder erstelle Session
        if (!_sessions.ContainsKey(sessionId))
        {
            _sessions[sessionId] = new ConversationState { SessionId = sessionId };
            _sessionMetrics[sessionId] = new StickySessionMetrics { SessionId = sessionId };
        }
        
        var state = _sessions[sessionId];
        var metrics = _sessionMetrics[sessionId];
        
        // Exit-Detection vor Routing
        var exitDetection = await _exitDetector.DetectExitIntentAsync(userInput, state);
        LogExitDetection(exitDetection);
        
        // Topic Change Detection
        TopicChangeResult? topicChange = null;
        if (state.History.Any())
        {
            topicChange = await _exitDetector.DetectTopicChangeAsync(userInput, state);
            if (topicChange.TopicChanged)
            {
                LogTopicChange(topicChange);
            }
        }
        
        // Routing mit Sticky Behavior
        var routingResult = await _router.RouteAsync(userInput, state, _agents);
        
        if (routingResult.SelectedAgent == null)
        {
            _logger.LogError("Kein Agent verfügbar");
            return new StickySessionResponse
            {
                AgentName = "System",
                Response = "Entschuldigung, kein Agent verfügbar.",
                RoutingInfo = routingResult
            };
        }
        
        // Log Sticky Routing Entscheidung
        LogStickyRoutingDecision(routingResult, metrics);
        
        // Update State und Metriken
        UpdateSessionState(state, routingResult, metrics);
        
        // Verarbeite mit gewähltem Agent
        var agentResponse = await routingResult.SelectedAgent.ProcessAsync(userInput, state);
        
        // Log Performance
        _logger.LogInformation($"[AGENT] {routingResult.SelectedAgent.Name} Antwortzeit: {agentResponse.ProcessingTimeMs}ms");
        
        // Füge zur Historie hinzu
        state.History.Add(new ConversationTurn
        {
            UserInput = userInput,
            AgentResponse = agentResponse.Message,
            AgentName = routingResult.SelectedAgent.Name,
            Timestamp = DateTime.UtcNow
        });
        
        // Erstelle Response mit allen Sticky Session Infos
        var response = new StickySessionResponse
        {
            AgentName = routingResult.SelectedAgent.Name,
            Response = agentResponse.Message,
            RoutingInfo = routingResult,
            ExitDetection = exitDetection,
            TopicChange = topicChange,
            StickyMetrics = GetSessionMetrics(sessionId),
            SessionSticky = !routingResult.AgentChanged && state.CurrentAgentTurnCount > 1,
            SuggestUserPrompt = ShouldSuggestPrompt(state, metrics)
        };
        
        // Log Session Info
        LogSessionInfo(state, metrics);
        
        return response;
    }
    
    /// <summary>
    /// Gibt Sticky Session Metriken zurück
    /// </summary>
    public Dictionary<string, object> GetSessionMetrics(string sessionId)
    {
        if (!_sessionMetrics.ContainsKey(sessionId))
            return new Dictionary<string, object>();
        
        var metrics = _sessionMetrics[sessionId];
        var state = _sessions[sessionId];
        
        return new Dictionary<string, object>
        {
            ["session_id"] = sessionId,
            ["total_turns"] = state.TurnCount,
            ["current_agent"] = state.CurrentAgent ?? "None",
            ["sticky_duration"] = metrics.CurrentStickyDuration,
            ["total_sticky_breaks"] = metrics.StickyBreaks,
            ["average_sticky_duration"] = metrics.AverageStickyDuration,
            ["exit_signals_detected"] = metrics.ExitSignalsDetected,
            ["topic_changes"] = metrics.TopicChanges,
            ["is_sticky"] = metrics.IsCurrentlySticky,
            ["workflow_stage"] = state.WorkflowStage.ToString()
        };
    }
    
    /// <summary>
    /// Gibt Session-Historie zurück
    /// </summary>
    public List<ConversationTurn> GetSessionHistory(string sessionId)
    {
        return _sessions.ContainsKey(sessionId) ? _sessions[sessionId].History : new List<ConversationTurn>();
    }
    
    /// <summary>
    /// Setzt eine Session zurück
    /// </summary>
    public void ResetSession(string sessionId)
    {
        if (_sessions.ContainsKey(sessionId))
        {
            _sessions[sessionId] = new ConversationState { SessionId = sessionId };
            _sessionMetrics[sessionId] = new StickySessionMetrics { SessionId = sessionId };
            _logger.LogInformation($"Sticky Session {sessionId} wurde zurückgesetzt");
        }
    }
    
    private void UpdateSessionState(ConversationState state, RoutingResult routing, StickySessionMetrics metrics)
    {
        if (routing.AgentChanged)
        {
            // Sticky Break
            if (metrics.IsCurrentlySticky)
            {
                metrics.StickyBreaks++;
                metrics.StickyDurations.Add(metrics.CurrentStickyDuration);
                metrics.CurrentStickyDuration = 0;
                metrics.IsCurrentlySticky = false;
            }
            
            state.CurrentAgent = routing.SelectedAgent!.Name;
            state.CurrentAgentTurnCount = 0;
        }
        else if (state.CurrentAgent == routing.SelectedAgent?.Name)
        {
            // Sticky fortsetzung
            metrics.CurrentStickyDuration++;
            metrics.IsCurrentlySticky = true;
        }
        
        state.CurrentAgentTurnCount++;
        state.TurnCount++;
        state.LastActivity = DateTime.UtcNow;
        
        // Update Workflow Stage
        UpdateWorkflowStage(routing.SelectedAgent!.Name, state);
    }
    
    private void UpdateWorkflowStage(string agentName, ConversationState state)
    {
        switch (agentName)
        {
            case "BookingAgent":
                if (state.WorkflowStage != WorkflowStage.BookingInProgress)
                    state.WorkflowStage = WorkflowStage.BookingInProgress;
                break;
            case "SupportAgent":
                if (state.WorkflowStage != WorkflowStage.SupportInProgress)
                    state.WorkflowStage = WorkflowStage.SupportInProgress;
                break;
            case "KnowledgeAgent":
                if (state.WorkflowStage != WorkflowStage.KnowledgeQuery)
                    state.WorkflowStage = WorkflowStage.KnowledgeQuery;
                break;
        }
    }
    
    private string? ShouldSuggestPrompt(ConversationState state, StickySessionMetrics metrics)
    {
        // Nach vielen Turns mit gleichem Agent
        if (metrics.IsCurrentlySticky && metrics.CurrentStickyDuration > 8)
        {
            return "Möchten Sie mit der aktuellen Aufgabe fortfahren oder haben Sie eine andere Frage?";
        }
        
        // Bei Stagnation
        if (state.CurrentAgentTurnCount > 5 && state.WorkflowStage != WorkflowStage.Completed)
        {
            return "Kann ich Ihnen noch bei etwas anderem helfen?";
        }
        
        return null;
    }
    
    private void LogExitDetection(ExitDetectionResult exitDetection)
    {
        if (exitDetection.ExitDetected)
        {
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine($"\n[EXIT DETECTION] Exit-Signal erkannt!");
            Console.WriteLine($"  Type: {exitDetection.ExitType}");
            Console.WriteLine($"  Confidence: {exitDetection.Confidence:F2}");
            Console.WriteLine($"  Reason: {exitDetection.Reason}");
            Console.ResetColor();
        }
    }
    
    private void LogTopicChange(TopicChangeResult topicChange)
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine($"\n[TOPIC CHANGE] Themenwechsel erkannt!");
        Console.WriteLine($"  Von: {topicChange.PreviousTopic}");
        Console.WriteLine($"  Zu: {topicChange.NewTopic}");
        Console.WriteLine($"  Confidence: {topicChange.Confidence:F2}");
        Console.ResetColor();
    }
    
    private void LogStickyRoutingDecision(RoutingResult result, StickySessionMetrics metrics)
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"\n[STICKY ROUTING] {result.Reason}");
        
        if (!result.AgentChanged && metrics.IsCurrentlySticky)
        {
            Console.WriteLine($"[STICKY ROUTING] 📌 STICKY SESSION AKTIV - Duration: {metrics.CurrentStickyDuration} turns");
        }
        else if (result.AgentChanged && metrics.IsCurrentlySticky)
        {
            Console.WriteLine($"[STICKY ROUTING] 🔓 STICKY SESSION BEENDET nach {metrics.CurrentStickyDuration} turns");
        }
        
        Console.WriteLine($"[STICKY ROUTING] Agent: {result.SelectedAgent?.Name}, Confidence: {result.Confidence:F2} | Routing-Zeit: {result.RoutingTimeMs}ms");
        
        if (result.AlternativeAgents.Any())
        {
            Console.WriteLine($"[STICKY ROUTING] Alternativen:");
            foreach (var alt in result.AlternativeAgents.OrderByDescending(kvp => kvp.Value))
            {
                Console.WriteLine($"  - {alt.Key}: {alt.Value:F2}");
            }
        }
        Console.ResetColor();
    }
    
    private void LogSessionInfo(ConversationState state, StickySessionMetrics metrics)
    {
        _logger.LogDebug($"Sticky Session Info: Turns={state.TurnCount}, " +
                        $"Agent={state.CurrentAgent}, " +
                        $"Sticky={metrics.IsCurrentlySticky}, " +
                        $"Duration={metrics.CurrentStickyDuration}, " +
                        $"Breaks={metrics.StickyBreaks}");
    }
}

/// <summary>
/// Response-Objekt für Sticky Sessions
/// </summary>
public class StickySessionResponse
{
    public string AgentName { get; set; } = string.Empty;
    public string Response { get; set; } = string.Empty;
    public RoutingResult? RoutingInfo { get; set; }
    public ExitDetectionResult? ExitDetection { get; set; }
    public TopicChangeResult? TopicChange { get; set; }
    public Dictionary<string, object> StickyMetrics { get; set; } = new();
    public bool SessionSticky { get; set; }
    public string? SuggestUserPrompt { get; set; }
}

/// <summary>
/// Metriken für Sticky Sessions
/// </summary>
public class StickySessionMetrics
{
    public string SessionId { get; set; } = string.Empty;
    public bool IsCurrentlySticky { get; set; }
    public int CurrentStickyDuration { get; set; }
    public int StickyBreaks { get; set; }
    public List<int> StickyDurations { get; set; } = new();
    public int ExitSignalsDetected { get; set; }
    public int TopicChanges { get; set; }
    
    public double AverageStickyDuration => 
        StickyDurations.Any() ? StickyDurations.Average() : CurrentStickyDuration;
}