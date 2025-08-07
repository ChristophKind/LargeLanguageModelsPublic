using Microsoft.SemanticKernel;
using Microsoft.Extensions.Logging;
using AgentRouterTest.Common.Interfaces;
using AgentRouterTest.Common.Models;

namespace AgentRouterTest.Approach2_AgentOwnership;

/// <summary>
/// Orchestrator für Approach 2: Agent-Ownership Pattern
/// Koordiniert selbstverwaltete Agents mit eigenständiger Kontrollübergabe
/// </summary>
public class OwnershipOrchestrator
{
    private readonly Kernel _kernel;
    private readonly AgentOwnershipRouter _router;
    private readonly List<IAgent> _agents;
    private readonly Dictionary<string, ConversationState> _sessions;
    private readonly ILogger<OwnershipOrchestrator> _logger;
    
    public OwnershipOrchestrator(Kernel kernel, ILoggerFactory? loggerFactory = null)
    {
        _kernel = kernel;
        _logger = loggerFactory?.CreateLogger<OwnershipOrchestrator>() 
                  ?? new LoggerFactory().CreateLogger<OwnershipOrchestrator>();
        
        // Initialisiere Router
        _router = new AgentOwnershipRouter(loggerFactory);
        
        // Initialisiere selbstverwaltete Agents
        _agents = new List<IAgent>
        {
            new SelfManagedBookingAgent(kernel),
            new SelfManagedSupportAgent(kernel),
            new SelfManagedKnowledgeAgent(kernel)
        };
        
        _sessions = new Dictionary<string, ConversationState>();
        
        _logger.LogInformation("Ownership Orchestrator initialisiert mit selbstverwalteten Agents");
    }
    
    /// <summary>
    /// Startet eine neue Chat-Session
    /// </summary>
    public string StartNewSession()
    {
        var sessionId = Guid.NewGuid().ToString();
        _sessions[sessionId] = new ConversationState { SessionId = sessionId };
        _logger.LogInformation($"Neue Session mit Agent-Ownership gestartet: {sessionId}");
        return sessionId;
    }
    
    /// <summary>
    /// Verarbeitet eine Benutzereingabe mit Agent-Ownership Pattern
    /// </summary>
    public async Task<(string agentName, string response, RoutingResult routingInfo, Dictionary<string, object> metadata)> 
        ProcessMessageAsync(string sessionId, string userInput)
    {
        _logger.LogInformation($"\n{new string('=', 80)}");
        _logger.LogInformation($"[OWNERSHIP] Verarbeite Nachricht für Session {sessionId}");
        
        // Hole Session-State
        if (!_sessions.ContainsKey(sessionId))
        {
            _sessions[sessionId] = new ConversationState { SessionId = sessionId };
        }
        var state = _sessions[sessionId];
        
        // Routing mit Agent-Ownership
        var routingResult = await _router.RouteAsync(userInput, state, _agents);
        
        if (routingResult.SelectedAgent == null)
        {
            _logger.LogError("Kein Agent konnte die Ownership übernehmen");
            return ("System", "Entschuldigung, kein Agent kann diese Anfrage bearbeiten.", 
                   routingResult, new Dictionary<string, object>());
        }
        
        // Log Ownership-Entscheidung
        LogOwnershipDecision(routingResult, state);
        
        // Update State
        if (routingResult.AgentChanged)
        {
            state.CurrentAgent = routingResult.SelectedAgent.Name;
            state.CurrentAgentTurnCount = 0;
        }
        state.CurrentAgentTurnCount++;
        state.TurnCount++;
        
        // Verarbeite mit gewähltem Agent
        var agentResponse = await routingResult.SelectedAgent.ProcessAsync(userInput, state);
        
        // Log Performance
        _logger.LogInformation($"[AGENT] {routingResult.SelectedAgent.Name} Antwortzeit: {agentResponse.ProcessingTimeMs}ms");
        
        // Extrahiere Ownership-Metadaten
        var metadata = new Dictionary<string, object>();
        if (agentResponse.Metadata.ContainsKey("ownership_confidence"))
        {
            metadata["ownership_confidence"] = agentResponse.Metadata["ownership_confidence"];
            metadata["ownership_reason"] = agentResponse.Metadata["ownership_reason"] ?? "";
            metadata["ownership_priority"] = agentResponse.Metadata["ownership_priority"] ?? 0;
        }
        metadata["keep_control"] = agentResponse.KeepControl;
        metadata["suggested_next"] = agentResponse.SuggestedNextAgent ?? "None";
        
        // Füge zur Historie hinzu
        state.History.Add(new ConversationTurn
        {
            UserInput = userInput,
            AgentResponse = agentResponse.Message,
            AgentName = routingResult.SelectedAgent.Name,
            Timestamp = DateTime.UtcNow
        });
        
        // Update Workflow-Stage basierend auf Agent
        UpdateWorkflowStage(routingResult.SelectedAgent.Name, state);
        
        // Log Session-Info
        LogSessionInfo(state);
        
        return (routingResult.SelectedAgent.Name, agentResponse.Message, routingResult, metadata);
    }
    
    /// <summary>
    /// Gibt detaillierte Ownership-Informationen für die aktuelle Session zurück
    /// </summary>
    public Dictionary<string, object> GetOwnershipInfo(string sessionId)
    {
        if (!_sessions.ContainsKey(sessionId))
            return new Dictionary<string, object>();
        
        var state = _sessions[sessionId];
        var currentAgent = _agents.FirstOrDefault(a => a.Name == state.CurrentAgent);
        
        var info = new Dictionary<string, object>
        {
            ["current_owner"] = state.CurrentAgent ?? "None",
            ["ownership_duration"] = state.CurrentAgentTurnCount,
            ["total_handovers"] = CountHandovers(state),
            ["workflow_stage"] = state.WorkflowStage.ToString(),
            ["session_turns"] = state.TurnCount
        };
        
        if (currentAgent is ISelfManagedAgent)
        {
            info["is_self_managed"] = true;
        }
        
        return info;
    }
    
    /// <summary>
    /// Gibt die Session-Historie zurück
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
            _logger.LogInformation($"Session {sessionId} wurde zurückgesetzt");
        }
    }
    
    private void LogOwnershipDecision(RoutingResult result, ConversationState state)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"\n[OWNERSHIP] {result.Reason}");
        Console.WriteLine($"[OWNERSHIP] Confidence: {result.Confidence:F2} | Routing-Zeit: {result.RoutingTimeMs}ms");
        
        if (result.AgentChanged)
        {
            Console.WriteLine($"[OWNERSHIP] Ownership-Transfer: {result.PreviousAgent} → {result.SelectedAgent?.Name}");
        }
        else
        {
            Console.WriteLine($"[OWNERSHIP] {result.SelectedAgent?.Name} behält Ownership");
        }
        
        if (result.AlternativeAgents.Any())
        {
            Console.WriteLine($"[OWNERSHIP] Alternative Claimants:");
            foreach (var alt in result.AlternativeAgents.OrderByDescending(kvp => kvp.Value))
            {
                Console.WriteLine($"  - {alt.Key}: {alt.Value:F2}");
            }
        }
        Console.ResetColor();
    }
    
    private void UpdateWorkflowStage(string agentName, ConversationState state)
    {
        switch (agentName)
        {
            case "SelfManagedBookingAgent":
                state.WorkflowStage = WorkflowStage.BookingInProgress;
                break;
            case "SelfManagedSupportAgent":
                state.WorkflowStage = WorkflowStage.SupportInProgress;
                break;
            case "SelfManagedKnowledgeAgent":
                state.WorkflowStage = WorkflowStage.KnowledgeQuery;
                break;
        }
    }
    
    private int CountHandovers(ConversationState state)
    {
        int handovers = 0;
        string? previousAgent = null;
        
        foreach (var turn in state.History)
        {
            if (previousAgent != null && previousAgent != turn.AgentName)
            {
                handovers++;
            }
            previousAgent = turn.AgentName;
        }
        
        return handovers;
    }
    
    private void LogSessionInfo(ConversationState state)
    {
        _logger.LogDebug($"Session-Info: Turns={state.TurnCount}, " +
                        $"Current Owner={state.CurrentAgent}, " +
                        $"Ownership Duration={state.CurrentAgentTurnCount}, " +
                        $"Stage={state.WorkflowStage}");
    }
}