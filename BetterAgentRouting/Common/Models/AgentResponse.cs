namespace AgentRouterTest.Common.Models;

/// <summary>
/// Repräsentiert die Antwort eines Agents
/// </summary>
public class AgentResponse
{
    /// <summary>
    /// Die Hauptantwort des Agents
    /// </summary>
    public string Message { get; set; } = string.Empty;
    
    /// <summary>
    /// Gibt an, ob der Agent die Kontrolle behalten möchte
    /// </summary>
    public bool KeepControl { get; set; } = false;
    
    /// <summary>
    /// Vorgeschlagener nächster Agent (falls Kontrolle abgegeben wird)
    /// </summary>
    public string? SuggestedNextAgent { get; set; }
    
    /// <summary>
    /// Gibt an, ob der Workflow abgeschlossen ist
    /// </summary>
    public bool WorkflowCompleted { get; set; } = false;
    
    /// <summary>
    /// Zusätzliche Metadaten
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();
    
    /// <summary>
    /// Confidence des Agents für diese Antwort (0-1)
    /// </summary>
    public double ResponseConfidence { get; set; } = 1.0;
    
    /// <summary>
    /// Zeit, die der Agent für die Antwort benötigt hat (in Millisekunden)
    /// </summary>
    public long ProcessingTimeMs { get; set; }
}