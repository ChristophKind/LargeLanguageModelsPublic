using Microsoft.Extensions.Logging;
using AgentRouterTest.Common.Interfaces;
using AgentRouterTest.Common.Models;
using System.Diagnostics;

namespace AgentRouterTest.Approach2_AgentOwnership;

/// <summary>
/// Router-Implementierung mit Agent-Ownership Pattern
/// Agents entscheiden selbst über Kontrollübergabe
/// </summary>
public class AgentOwnershipRouter : IRouter
{
    private readonly ILogger<AgentOwnershipRouter> _logger;
    
    public string ApproachName => "Agent-Ownership Pattern";
    
    public AgentOwnershipRouter(ILoggerFactory? loggerFactory = null)
    {
        _logger = loggerFactory?.CreateLogger<AgentOwnershipRouter>() 
                  ?? new LoggerFactory().CreateLogger<AgentOwnershipRouter>();
    }
    
    public async Task<RoutingResult> RouteAsync(
        string userInput, 
        ConversationState conversationState, 
        List<IAgent> availableAgents)
    {
        var stopwatch = Stopwatch.StartNew();
        
        _logger.LogInformation($"[OWNERSHIP ROUTER] Starte Routing für: {userInput.Substring(0, Math.Min(50, userInput.Length))}...");
        _logger.LogInformation($"[OWNERSHIP ROUTER] Aktueller Agent: {conversationState.CurrentAgent ?? "None"}");
        
        var result = new RoutingResult();
        
        // Prüfe ob aktueller Agent existiert und Ownership behalten will
        if (!string.IsNullOrEmpty(conversationState.CurrentAgent))
        {
            var currentAgent = availableAgents.FirstOrDefault(a => a.Name == conversationState.CurrentAgent);
            
            if (currentAgent is ISelfManagedAgent selfManagedAgent)
            {
                // Frage aktuellen Agent nach Ownership-Entscheidung
                var ownershipDecision = await selfManagedAgent.AnalyzeOwnershipAsync(
                    userInput, 
                    conversationState.History.LastOrDefault()?.AgentResponse ?? "",
                    conversationState);
                
                _logger.LogInformation($"[OWNERSHIP ROUTER] {currentAgent.Name} Ownership-Entscheidung: " +
                                      $"KeepControl={ownershipDecision.KeepControl}, " +
                                      $"Confidence={ownershipDecision.Confidence:F2}, " +
                                      $"Priority={ownershipDecision.Priority}");
                
                if (ownershipDecision.KeepControl && ownershipDecision.Confidence > 0.5)
                {
                    // Agent behält Kontrolle
                    result.SelectedAgent = currentAgent;
                    result.Confidence = ownershipDecision.Confidence;
                    result.AgentChanged = false;
                    result.Reason = $"{currentAgent.Name} behält Kontrolle: {ownershipDecision.Reason}";
                    
                    _logger.LogInformation($"[OWNERSHIP ROUTER] {currentAgent.Name} behält Kontrolle");
                    return result;
                }
                
                // Agent gibt Kontrolle ab - prüfe Vorschlag
                if (!string.IsNullOrEmpty(ownershipDecision.SuggestedNextAgent))
                {
                    var suggestedAgent = availableAgents.FirstOrDefault(
                        a => a.Name == ownershipDecision.SuggestedNextAgent);
                    
                    if (suggestedAgent != null)
                    {
                        _logger.LogInformation($"[OWNERSHIP ROUTER] Folge Vorschlag von {currentAgent.Name}: " +
                                              $"Wechsel zu {suggestedAgent.Name}");
                        
                        result.SelectedAgent = suggestedAgent;
                        result.Confidence = 0.8; // Hohe Confidence für Agent-Vorschläge
                        result.AgentChanged = true;
                        result.PreviousAgent = conversationState.CurrentAgent;
                        result.Reason = $"Übergabe von {currentAgent.Name} an {suggestedAgent.Name} (vorgeschlagen)";
                        return result;
                    }
                }
            }
        }
        
        // Kein aktueller Agent oder Agent gibt ab ohne Vorschlag
        // Frage alle Agents nach ihrer Eignung und Ownership-Anspruch
        var agentClaims = new Dictionary<IAgent, (double suitability, OwnershipDecision? ownership)>();
        
        foreach (var agent in availableAgents)
        {
            var suitability = await agent.EvaluateSuitabilityAsync(userInput, conversationState);
            
            OwnershipDecision? ownership = null;
            if (agent is ISelfManagedAgent selfManaged)
            {
                ownership = await selfManaged.AnalyzeOwnershipAsync(userInput, "", conversationState);
            }
            
            agentClaims[agent] = (suitability, ownership);
            
            _logger.LogDebug($"[OWNERSHIP ROUTER] {agent.Name}: Suitability={suitability:F2}, " +
                           $"WantsControl={ownership?.KeepControl}, Priority={ownership?.Priority}");
        }
        
        // Wähle Agent mit bestem Anspruch
        var selectedAgent = SelectBestClaimant(agentClaims);
        
        result.SelectedAgent = selectedAgent;
        result.Confidence = agentClaims[selectedAgent].suitability;
        result.AgentChanged = selectedAgent.Name != conversationState.CurrentAgent;
        result.PreviousAgent = conversationState.CurrentAgent;
        result.AlternativeAgents = agentClaims.ToDictionary(
            kvp => kvp.Key.Name, 
            kvp => kvp.Value.suitability);
        
        if (result.AgentChanged)
        {
            result.Reason = $"Neuer Agent {selectedAgent.Name} übernimmt " +
                          $"(Suitability: {result.Confidence:F2}, " +
                          $"Priority: {agentClaims[selectedAgent].ownership?.Priority ?? 0})";
        }
        else
        {
            result.Reason = $"{selectedAgent.Name} behält Kontrolle";
        }
        
        _logger.LogInformation($"[OWNERSHIP ROUTER] Finale Entscheidung: {selectedAgent.Name}");
        
        stopwatch.Stop();
        result.RoutingTimeMs = stopwatch.ElapsedMilliseconds;
        _logger.LogInformation($"[OWNERSHIP ROUTER] Routing abgeschlossen in {result.RoutingTimeMs}ms");
        
        return result;
    }
    
    /// <summary>
    /// Wählt den Agent mit dem besten Ownership-Anspruch
    /// </summary>
    private IAgent SelectBestClaimant(
        Dictionary<IAgent, (double suitability, OwnershipDecision? ownership)> claims)
    {
        // Sortiere nach: 
        // 1. Agents die Kontrolle wollen (KeepControl = true)
        // 2. Priorität (höher = besser)
        // 3. Suitability Score
        
        var ranked = claims
            .OrderByDescending(c => c.Value.ownership?.KeepControl == true ? 1 : 0)
            .ThenByDescending(c => c.Value.ownership?.Priority ?? 0)
            .ThenByDescending(c => c.Value.suitability)
            .ToList();
        
        return ranked.First().Key;
    }
}