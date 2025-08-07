using AgentRouterTest.Common.Interfaces;
using AgentRouterTest.Common.Models;

namespace AgentRouterTest.Approach2_AgentOwnership;

/// <summary>
/// Erweitertes Interface für selbstverwaltete Agents
/// Agents können selbst entscheiden, wann sie die Kontrolle abgeben
/// </summary>
public interface ISelfManagedAgent : IAgent
{
    /// <summary>
    /// Analysiert, ob der Agent die Kontrolle behalten sollte
    /// </summary>
    Task<OwnershipDecision> AnalyzeOwnershipAsync(string userInput, string lastResponse, ConversationState state);
    
    /// <summary>
    /// Schlägt einen Nachfolge-Agent vor, falls die Kontrolle abgegeben wird
    /// </summary>
    Task<string?> SuggestNextAgentAsync(string userInput, ConversationState state);
}

/// <summary>
/// Repräsentiert die Ownership-Entscheidung eines Agents
/// </summary>
public class OwnershipDecision
{
    /// <summary>
    /// Gibt an, ob der Agent die Kontrolle behalten möchte
    /// </summary>
    public bool KeepControl { get; set; }
    
    /// <summary>
    /// Confidence-Level für die Entscheidung (0-1)
    /// </summary>
    public double Confidence { get; set; }
    
    /// <summary>
    /// Begründung für die Entscheidung
    /// </summary>
    public string Reason { get; set; } = string.Empty;
    
    /// <summary>
    /// Vorgeschlagener nächster Agent (optional)
    /// </summary>
    public string? SuggestedNextAgent { get; set; }
    
    /// <summary>
    /// Priorität der Ownership (höhere Werte = stärkerer Anspruch)
    /// </summary>
    public int Priority { get; set; } = 0;
}