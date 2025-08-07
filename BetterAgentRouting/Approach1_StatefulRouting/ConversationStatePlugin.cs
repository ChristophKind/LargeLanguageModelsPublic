using Microsoft.SemanticKernel;
using Microsoft.Extensions.Logging;
using AgentRouterTest.Common.Models;
using System.ComponentModel;

namespace AgentRouterTest.Approach1_StatefulRouting;

/// <summary>
/// Plugin zur Verwaltung des Konversationszustands
/// Trackt Workflow-Modi, sammelt Kontextdaten und setzt dynamische Schwellenwerte
/// </summary>
public class ConversationStatePlugin
{
    private readonly Dictionary<string, ConversationState> _sessions = new();
    private readonly ILogger<ConversationStatePlugin> _logger;
    
    public ConversationStatePlugin(ILoggerFactory? loggerFactory = null)
    {
        _logger = loggerFactory?.CreateLogger<ConversationStatePlugin>() 
                  ?? new LoggerFactory().CreateLogger<ConversationStatePlugin>();
    }
    
    /// <summary>
    /// Ruft den aktuellen Konversationszustand ab oder erstellt einen neuen
    /// </summary>
    [KernelFunction("GetConversationState")]
    [Description("Retrieves or creates conversation state for a session")]
    public ConversationState GetOrCreateState(string sessionId)
    {
        if (!_sessions.ContainsKey(sessionId))
        {
            _logger.LogInformation($"Erstelle neuen Konversationszustand für Session {sessionId}");
            _sessions[sessionId] = new ConversationState { SessionId = sessionId };
        }
        
        return _sessions[sessionId];
    }
    
    /// <summary>
    /// Aktualisiert den Konversationszustand
    /// </summary>
    [KernelFunction("UpdateConversationState")]
    [Description("Updates the conversation state with new information")]
    public void UpdateState(
        string sessionId, 
        string? currentAgent = null,
        WorkflowStage? workflowStage = null,
        Dictionary<string, object>? contextUpdates = null)
    {
        var state = GetOrCreateState(sessionId);
        
        // Agent-Wechsel tracken
        if (currentAgent != null && currentAgent != state.CurrentAgent)
        {
            _logger.LogInformation($"Agent-Wechsel: {state.CurrentAgent} -> {currentAgent}");
            state.CurrentAgent = currentAgent;
            state.CurrentAgentTurnCount = 0;
        }
        
        // Workflow-Stage aktualisieren
        if (workflowStage.HasValue && workflowStage.Value != state.WorkflowStage)
        {
            _logger.LogInformation($"Workflow-Stage-Wechsel: {state.WorkflowStage} -> {workflowStage.Value}");
            state.WorkflowStage = workflowStage.Value;
        }
        
        // Kontext-Updates anwenden
        if (contextUpdates != null)
        {
            foreach (var update in contextUpdates)
            {
                state.Context[update.Key] = update.Value;
            }
        }
        
        state.LastActivity = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Fügt einen Konversations-Turn zur Historie hinzu
    /// </summary>
    [KernelFunction("AddConversationTurn")]
    [Description("Adds a conversation turn to the history")]
    public void AddTurn(string sessionId, string userInput, string agentResponse, string agentName)
    {
        var state = GetOrCreateState(sessionId);
        
        state.History.Add(new ConversationTurn
        {
            UserInput = userInput,
            AgentResponse = agentResponse,
            AgentName = agentName,
            Timestamp = DateTime.UtcNow
        });
        
        state.TurnCount++;
        
        // Erhöhe Turn-Count für aktuellen Agent
        if (state.CurrentAgent == agentName)
        {
            state.CurrentAgentTurnCount++;
        }
        
        _logger.LogDebug($"Turn {state.TurnCount} hinzugefügt für Session {sessionId}");
    }
    
    /// <summary>
    /// Berechnet den dynamischen Confidence-Schwellenwert basierend auf dem aktuellen Zustand
    /// </summary>
    [KernelFunction("CalculateDynamicThreshold")]
    [Description("Calculates dynamic confidence threshold based on conversation state")]
    public double CalculateDynamicThreshold(string sessionId)
    {
        var state = GetOrCreateState(sessionId);
        double baseThreshold = 0.6;
        
        // Erhöhe Schwellenwert bei aktivem Workflow
        if (state.WorkflowStage == WorkflowStage.BookingInProgress)
        {
            baseThreshold += 0.2; // Höhere Hürde für Wechsel bei Buchung
        }
        else if (state.WorkflowStage == WorkflowStage.SupportInProgress)
        {
            baseThreshold += 0.15; // Mittlere Erhöhung bei Support
        }
        
        // Erhöhe Schwellenwert basierend auf Gesprächslänge mit aktuellem Agent
        if (state.CurrentAgentTurnCount > 2)
        {
            baseThreshold += Math.Min(0.2, state.CurrentAgentTurnCount * 0.05);
        }
        
        // Cap bei 0.95, damit Wechsel noch möglich sind
        return Math.Min(0.95, baseThreshold);
    }
    
    /// <summary>
    /// Prüft, ob ein Workflow-Wechsel erlaubt ist
    /// </summary>
    [KernelFunction("IsWorkflowSwitchAllowed")]
    [Description("Checks if switching workflow is allowed based on current state")]
    public bool IsWorkflowSwitchAllowed(string sessionId)
    {
        var state = GetOrCreateState(sessionId);
        
        // Bei kritischen Workflow-Stadien Wechsel verhindern
        if (state.WorkflowStage == WorkflowStage.BookingInProgress)
        {
            // Prüfe ob Buchungsdaten vorhanden sind
            if (state.Context.ContainsKey("booking_stage") && 
                state.Context["booking_stage"]?.ToString() == "confirmation")
            {
                _logger.LogWarning("Workflow-Wechsel blockiert: Buchung in Bestätigungsphase");
                return false;
            }
        }
        
        return true;
    }
    
    /// <summary>
    /// Setzt den Konversationszustand zurück
    /// </summary>
    [KernelFunction("ResetConversationState")]
    [Description("Resets the conversation state for a session")]
    public void ResetState(string sessionId)
    {
        if (_sessions.ContainsKey(sessionId))
        {
            _logger.LogInformation($"Setze Konversationszustand für Session {sessionId} zurück");
            _sessions[sessionId] = new ConversationState { SessionId = sessionId };
        }
    }
    
    /// <summary>
    /// Gibt Statistiken über die aktuelle Session zurück
    /// </summary>
    [KernelFunction("GetSessionStatistics")]
    [Description("Returns statistics about the current session")]
    public Dictionary<string, object> GetStatistics(string sessionId)
    {
        var state = GetOrCreateState(sessionId);
        
        return new Dictionary<string, object>
        {
            ["TotalTurns"] = state.TurnCount,
            ["CurrentAgent"] = state.CurrentAgent ?? "None",
            ["CurrentAgentTurns"] = state.CurrentAgentTurnCount,
            ["WorkflowStage"] = state.WorkflowStage.ToString(),
            ["SessionDuration"] = (DateTime.UtcNow - (state.History.FirstOrDefault()?.Timestamp ?? DateTime.UtcNow)).TotalMinutes,
            ["UniqueAgents"] = state.History.Select(h => h.AgentName).Distinct().Count()
        };
    }
}