using Microsoft.SemanticKernel;
using Microsoft.Extensions.Logging;
using AgentRouterTest.Common.Interfaces;
using AgentRouterTest.Common.Models;
using System.Diagnostics;

namespace AgentRouterTest.Approach1_StatefulRouting;

/// <summary>
/// Router-Implementierung mit Stateful Routing und Context-Awareness
/// Verwendet dynamische Confidence-Schwellenwerte basierend auf dem Konversationszustand
/// </summary>
public class StatefulRouter : IRouter
{
    private readonly ConversationStatePlugin _statePlugin;
    private readonly ILogger<StatefulRouter> _logger;
    private readonly Kernel _kernel;
    
    public string ApproachName => "Stateful Routing mit Context-Awareness";
    
    public StatefulRouter(Kernel kernel, ConversationStatePlugin statePlugin, ILoggerFactory? loggerFactory = null)
    {
        _kernel = kernel;
        _statePlugin = statePlugin;
        _logger = loggerFactory?.CreateLogger<StatefulRouter>() 
                  ?? new LoggerFactory().CreateLogger<StatefulRouter>();
    }
    
    public async Task<RoutingResult> RouteAsync(
        string userInput, 
        ConversationState conversationState, 
        List<IAgent> availableAgents)
    {
        var stopwatch = Stopwatch.StartNew();
        
        _logger.LogInformation($"[STATEFUL ROUTER] Starte Routing für Input: {userInput.Substring(0, Math.Min(50, userInput.Length))}...");
        _logger.LogInformation($"[STATEFUL ROUTER] Aktueller Zustand - Agent: {conversationState.CurrentAgent}, Stage: {conversationState.WorkflowStage}");
        
        // Berechne dynamischen Schwellenwert
        var dynamicThreshold = _statePlugin.CalculateDynamicThreshold(conversationState.SessionId);
        _logger.LogInformation($"[STATEFUL ROUTER] Dynamischer Schwellenwert: {dynamicThreshold:F2}");
        
        // Evaluiere alle Agents
        var agentScores = new Dictionary<IAgent, double>();
        foreach (var agent in availableAgents)
        {
            var score = await agent.EvaluateSuitabilityAsync(userInput, conversationState);
            agentScores[agent] = score;
            _logger.LogDebug($"[STATEFUL ROUTER] {agent.Name} Score: {score:F2}");
        }
        
        // Finde besten Agent
        var bestAgent = agentScores.OrderByDescending(kvp => kvp.Value).First();
        var currentAgentInstance = availableAgents.FirstOrDefault(a => a.Name == conversationState.CurrentAgent);
        var currentAgentScore = currentAgentInstance != null ? agentScores[currentAgentInstance] : 0;
        
        // Entscheidungslogik mit State-Awareness
        var result = new RoutingResult
        {
            AlternativeAgents = agentScores.ToDictionary(kvp => kvp.Key.Name, kvp => kvp.Value)
        };
        
        // Prüfe ob Workflow-Wechsel erlaubt ist
        bool workflowSwitchAllowed = _statePlugin.IsWorkflowSwitchAllowed(conversationState.SessionId);
        
        // Entscheide basierend auf State und Scores
        if (conversationState.CurrentAgent != null && currentAgentInstance != null)
        {
            // Bonus für aktuellen Agent bei aktivem Workflow
            double currentAgentBonus = CalculateCurrentAgentBonus(conversationState);
            double adjustedCurrentScore = currentAgentScore + currentAgentBonus;
            
            _logger.LogInformation($"[STATEFUL ROUTER] Current Agent {conversationState.CurrentAgent}: " +
                                  $"Base Score={currentAgentScore:F2}, Bonus={currentAgentBonus:F2}, " +
                                  $"Adjusted={adjustedCurrentScore:F2}");
            
            // Bleibe beim aktuellen Agent wenn:
            // 1. Workflow nicht gewechselt werden darf
            // 2. Adjustierter Score noch akzeptabel ist
            // 3. Neuer Agent nicht signifikant besser ist (mit dynamischem Schwellenwert)
            if (!workflowSwitchAllowed || 
                (adjustedCurrentScore > 0.3 && bestAgent.Value < adjustedCurrentScore + dynamicThreshold))
            {
                result.SelectedAgent = currentAgentInstance;
                result.Confidence = adjustedCurrentScore;
                result.AgentChanged = false;
                result.Reason = workflowSwitchAllowed ? 
                    $"Bleibe bei {currentAgentInstance.Name} (Adjusted Score: {adjustedCurrentScore:F2}, Workflow: {conversationState.WorkflowStage})" :
                    $"Workflow-Wechsel blockiert, bleibe bei {currentAgentInstance.Name}";
                
                _logger.LogInformation($"[STATEFUL ROUTER] Entscheidung: Bleibe bei {currentAgentInstance.Name}");
            }
            else
            {
                // Wechsel zum besseren Agent
                result.SelectedAgent = bestAgent.Key;
                result.Confidence = bestAgent.Value;
                result.AgentChanged = true;
                result.PreviousAgent = conversationState.CurrentAgent;
                result.Reason = $"Wechsel zu {bestAgent.Key.Name} (Score: {bestAgent.Value:F2} übersteigt Schwellenwert {dynamicThreshold:F2})";
                
                _logger.LogInformation($"[STATEFUL ROUTER] Entscheidung: Wechsel von {conversationState.CurrentAgent} zu {bestAgent.Key.Name}");
            }
        }
        else
        {
            // Kein aktueller Agent, wähle besten
            result.SelectedAgent = bestAgent.Key;
            result.Confidence = bestAgent.Value;
            result.AgentChanged = conversationState.CurrentAgent != null;
            result.PreviousAgent = conversationState.CurrentAgent;
            result.Reason = $"Initialer Agent: {bestAgent.Key.Name} (Score: {bestAgent.Value:F2})";
            
            _logger.LogInformation($"[STATEFUL ROUTER] Entscheidung: Initialer Agent {bestAgent.Key.Name}");
        }
        
        // Update Workflow-Stage basierend auf gewähltem Agent
        UpdateWorkflowStage(result.SelectedAgent!, conversationState);
        
        stopwatch.Stop();
        result.RoutingTimeMs = stopwatch.ElapsedMilliseconds;
        _logger.LogInformation($"[STATEFUL ROUTER] Routing abgeschlossen in {result.RoutingTimeMs}ms");
        
        return result;
    }
    
    /// <summary>
    /// Berechnet Bonus für den aktuellen Agent basierend auf Konversationszustand
    /// </summary>
    private double CalculateCurrentAgentBonus(ConversationState state)
    {
        double bonus = 0;
        
        // Bonus basierend auf Workflow-Stage
        switch (state.WorkflowStage)
        {
            case WorkflowStage.BookingInProgress:
                bonus += 0.3; // Starker Bonus bei aktiver Buchung
                break;
            case WorkflowStage.SupportInProgress:
                bonus += 0.2; // Mittlerer Bonus bei Support
                break;
            case WorkflowStage.KnowledgeQuery:
                bonus += 0.1; // Kleiner Bonus bei Wissensfragen
                break;
        }
        
        // Bonus basierend auf Gesprächslänge
        if (state.CurrentAgentTurnCount > 0)
        {
            bonus += Math.Min(0.2, state.CurrentAgentTurnCount * 0.05);
        }
        
        // Bonus wenn kürzlich gewechselt wurde (Stabilität)
        if (state.CurrentAgentTurnCount < 2)
        {
            bonus += 0.1;
        }
        
        return bonus;
    }
    
    /// <summary>
    /// Aktualisiert den Workflow-Stage basierend auf dem gewählten Agent
    /// </summary>
    private void UpdateWorkflowStage(IAgent selectedAgent, ConversationState state)
    {
        // Nur updaten wenn sich Agent ändert oder Stage idle ist
        if (state.CurrentAgent != selectedAgent.Name || state.WorkflowStage == WorkflowStage.Idle)
        {
            switch (selectedAgent.Name)
            {
                case "BookingAgent":
                    if (state.WorkflowStage != WorkflowStage.BookingInProgress)
                    {
                        _logger.LogDebug("Setze WorkflowStage auf BookingInProgress");
                        state.WorkflowStage = WorkflowStage.BookingInProgress;
                    }
                    break;
                    
                case "SupportAgent":
                    if (state.WorkflowStage != WorkflowStage.SupportInProgress)
                    {
                        _logger.LogDebug("Setze WorkflowStage auf SupportInProgress");
                        state.WorkflowStage = WorkflowStage.SupportInProgress;
                    }
                    break;
                    
                case "KnowledgeAgent":
                    if (state.WorkflowStage != WorkflowStage.KnowledgeQuery)
                    {
                        _logger.LogDebug("Setze WorkflowStage auf KnowledgeQuery");
                        state.WorkflowStage = WorkflowStage.KnowledgeQuery;
                    }
                    break;
            }
        }
    }
}