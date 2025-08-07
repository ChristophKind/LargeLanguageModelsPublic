using Microsoft.SemanticKernel;
using Microsoft.Extensions.Logging;
using AgentRouterTest.Common.Interfaces;
using AgentRouterTest.Common.Models;
using AgentRouterTest.Common.BaseAgents;

namespace AgentRouterTest.Approach1_StatefulRouting;

/// <summary>
/// Orchestrator für Approach 1: Stateful Routing mit Context-Awareness
/// Koordiniert Router, State-Plugin und Agents
/// </summary>
public class StatefulRoutingOrchestrator
{
    private readonly Kernel _kernel;
    private readonly ConversationStatePlugin _statePlugin;
    private readonly StatefulRouter _router;
    private readonly List<IAgent> _agents;
    private readonly ILogger<StatefulRoutingOrchestrator> _logger;
    
    public StatefulRoutingOrchestrator(Kernel kernel, ILoggerFactory? loggerFactory = null)
    {
        _kernel = kernel;
        _logger = loggerFactory?.CreateLogger<StatefulRoutingOrchestrator>() 
                  ?? new LoggerFactory().CreateLogger<StatefulRoutingOrchestrator>();
        
        // Initialisiere State Plugin
        _statePlugin = new ConversationStatePlugin(loggerFactory);
        
        // Initialisiere Router
        _router = new StatefulRouter(kernel, _statePlugin, loggerFactory);
        
        // Initialisiere Agents
        _agents = new List<IAgent>
        {
            new BookingAgent(kernel),
            new SupportAgent(kernel),
            new KnowledgeAgent(kernel)
        };
        
        _logger.LogInformation("Stateful Routing Orchestrator initialisiert");
    }
    
    /// <summary>
    /// Startet eine neue Chat-Session
    /// </summary>
    public string StartNewSession()
    {
        var sessionId = Guid.NewGuid().ToString();
        _logger.LogInformation($"Neue Session gestartet: {sessionId}");
        return sessionId;
    }
    
    /// <summary>
    /// Verarbeitet eine Benutzereingabe und gibt die Agent-Antwort zurück
    /// </summary>
    public async Task<(string agentName, string response, RoutingResult routingInfo)> ProcessMessageAsync(
        string sessionId, 
        string userInput)
    {
        _logger.LogInformation($"\n{new string('=', 80)}");
        _logger.LogInformation($"Verarbeite Nachricht für Session {sessionId}: {userInput}");
        
        // Hole aktuellen State
        var state = _statePlugin.GetOrCreateState(sessionId);
        
        // Routing-Entscheidung
        var routingResult = await _router.RouteAsync(userInput, state, _agents);
        
        if (routingResult.SelectedAgent == null)
        {
            _logger.LogError("Kein Agent konnte ausgewählt werden");
            return ("System", "Entschuldigung, ich konnte keinen passenden Agent finden.", routingResult);
        }
        
        // Log Routing-Entscheidung
        LogRoutingDecision(routingResult);
        
        // Update State mit neuem Agent
        _statePlugin.UpdateState(
            sessionId, 
            currentAgent: routingResult.SelectedAgent.Name,
            workflowStage: state.WorkflowStage);
        
        // Verarbeite mit gewähltem Agent
        var agentResponse = await routingResult.SelectedAgent.ProcessAsync(userInput, state);
        
        // Log Agent Processing Time
        _logger.LogInformation($"[AGENT] {routingResult.SelectedAgent.Name} Antwortzeit: {agentResponse.ProcessingTimeMs}ms");
        
        // Füge Turn zur Historie hinzu
        _statePlugin.AddTurn(sessionId, userInput, agentResponse.Message, routingResult.SelectedAgent.Name);
        
        // Update State basierend auf Agent-Response
        if (agentResponse.WorkflowCompleted)
        {
            _logger.LogInformation("Workflow abgeschlossen, setze auf Idle");
            _statePlugin.UpdateState(sessionId, workflowStage: WorkflowStage.Idle);
        }
        
        // Log Session-Statistiken
        LogSessionStatistics(sessionId);
        
        return (routingResult.SelectedAgent.Name, agentResponse.Message, routingResult);
    }
    
    /// <summary>
    /// Gibt die Session-Historie zurück
    /// </summary>
    public List<ConversationTurn> GetSessionHistory(string sessionId)
    {
        var state = _statePlugin.GetOrCreateState(sessionId);
        return state.History;
    }
    
    /// <summary>
    /// Gibt Session-Statistiken zurück
    /// </summary>
    public Dictionary<string, object> GetSessionStatistics(string sessionId)
    {
        return _statePlugin.GetStatistics(sessionId);
    }
    
    /// <summary>
    /// Setzt eine Session zurück
    /// </summary>
    public void ResetSession(string sessionId)
    {
        _statePlugin.ResetState(sessionId);
        _logger.LogInformation($"Session {sessionId} wurde zurückgesetzt");
    }
    
    private void LogRoutingDecision(RoutingResult result)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"\n[ROUTING] {result.Reason}");
        Console.WriteLine($"[ROUTING] Confidence: {result.Confidence:F2} | Zeit: {result.RoutingTimeMs}ms");
        
        if (result.AgentChanged)
        {
            Console.WriteLine($"[ROUTING] Agent-Wechsel: {result.PreviousAgent} → {result.SelectedAgent?.Name}");
        }
        
        Console.WriteLine($"[ROUTING] Alternative Agents:");
        foreach (var alt in result.AlternativeAgents.OrderByDescending(kvp => kvp.Value))
        {
            Console.WriteLine($"  - {alt.Key}: {alt.Value:F2}");
        }
        Console.ResetColor();
    }
    
    private void LogSessionStatistics(string sessionId)
    {
        var stats = _statePlugin.GetStatistics(sessionId);
        _logger.LogDebug($"Session-Statistiken: Turns={stats["TotalTurns"]}, " +
                        $"Agent={stats["CurrentAgent"]}, " +
                        $"Stage={stats["WorkflowStage"]}");
    }
}