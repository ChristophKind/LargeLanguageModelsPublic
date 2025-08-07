using AgentRouterTest.Common.Interfaces;

namespace AgentRouterTest.Common.Models;

/// <summary>
/// Ergebnis einer Routing-Entscheidung
/// </summary>
public class RoutingResult
{
    /// <summary>
    /// Der gewählte Agent
    /// </summary>
    public IAgent? SelectedAgent { get; set; }
    
    /// <summary>
    /// Confidence-Score der Routing-Entscheidung (0-1)
    /// </summary>
    public double Confidence { get; set; }
    
    /// <summary>
    /// Begründung für die Routing-Entscheidung
    /// </summary>
    public string Reason { get; set; } = string.Empty;
    
    /// <summary>
    /// Gibt an, ob ein Agent-Wechsel stattgefunden hat
    /// </summary>
    public bool AgentChanged { get; set; }
    
    /// <summary>
    /// Der vorherige Agent (falls ein Wechsel stattgefunden hat)
    /// </summary>
    public string? PreviousAgent { get; set; }
    
    /// <summary>
    /// Alternative Agents mit ihren Confidence-Scores
    /// </summary>
    public Dictionary<string, double> AlternativeAgents { get; set; } = new();
    
    /// <summary>
    /// Zeit, die das Routing benötigt hat (in Millisekunden)
    /// </summary>
    public long RoutingTimeMs { get; set; }
}