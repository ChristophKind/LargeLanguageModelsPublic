namespace AgentRouterTest.Common.Models;

/// <summary>
/// Repräsentiert den aktuellen Zustand einer Konversation
/// </summary>
public class ConversationState
{
    /// <summary>
    /// Eindeutige Session-ID
    /// </summary>
    public string SessionId { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Der aktuell aktive Agent
    /// </summary>
    public string? CurrentAgent { get; set; }
    
    /// <summary>
    /// Der aktuelle Workflow-Status
    /// </summary>
    public WorkflowStage WorkflowStage { get; set; } = WorkflowStage.Idle;
    
    /// <summary>
    /// Anzahl der Konversations-Turns
    /// </summary>
    public int TurnCount { get; set; } = 0;
    
    /// <summary>
    /// Kontext-spezifische Daten (z.B. Buchungsdetails)
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Historie der Konversation
    /// </summary>
    public List<ConversationTurn> History { get; set; } = new();
    
    /// <summary>
    /// Zeitstempel der letzten Aktivität
    /// </summary>
    public DateTime LastActivity { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Anzahl der Turns mit dem aktuellen Agent
    /// </summary>
    public int CurrentAgentTurnCount { get; set; } = 0;
}

/// <summary>
/// Definiert die verschiedenen Workflow-Stadien
/// </summary>
public enum WorkflowStage
{
    /// <summary>
    /// Kein aktiver Workflow
    /// </summary>
    Idle,
    
    /// <summary>
    /// Buchungsprozess läuft
    /// </summary>
    BookingInProgress,
    
    /// <summary>
    /// Support-Anfrage wird bearbeitet
    /// </summary>
    SupportInProgress,
    
    /// <summary>
    /// Allgemeine Wissensabfrage
    /// </summary>
    KnowledgeQuery,
    
    /// <summary>
    /// Workflow abgeschlossen, wartet auf neue Anfrage
    /// </summary>
    Completed
}

/// <summary>
/// Repräsentiert einen einzelnen Turn in der Konversation
/// </summary>
public class ConversationTurn
{
    /// <summary>
    /// Die Benutzereingabe
    /// </summary>
    public string UserInput { get; set; } = string.Empty;
    
    /// <summary>
    /// Die Agent-Antwort
    /// </summary>
    public string AgentResponse { get; set; } = string.Empty;
    
    /// <summary>
    /// Der Agent, der geantwortet hat
    /// </summary>
    public string AgentName { get; set; } = string.Empty;
    
    /// <summary>
    /// Zeitstempel des Turns
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}