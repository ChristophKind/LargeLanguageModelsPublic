using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using AgentRouterTest.Common.Models;
using AgentRouterTest.Common.BaseAgents;

namespace AgentRouterTest.Approach2_AgentOwnership;

/// <summary>
/// Selbstverwalteter Buchungsagent mit Ownership-Kontrolle
/// </summary>
public class SelfManagedBookingAgent : BookingAgent, ISelfManagedAgent
{
    private new readonly IChatCompletionService _chatService;
    
    public SelfManagedBookingAgent(Kernel kernel) : base(kernel) 
    {
        _chatService = kernel.GetRequiredService<IChatCompletionService>();
    }
    
    /// <summary>
    /// Analysiert, ob der Agent die Kontrolle behalten sollte
    /// </summary>
    public async Task<OwnershipDecision> AnalyzeOwnershipAsync(
        string userInput, 
        string lastResponse, 
        ConversationState state)
    {
        // Analysiere mit LLM, ob Buchungsprozess fortgesetzt werden sollte
        var prompt = $@"
Analysiere ob der Buchungsagent die Kontrolle behalten sollte.

Aktueller Workflow-Status: {state.WorkflowStage}
Buchungsstatus: {state.Context.GetValueOrDefault("booking_stage", "none")}
Letzte Agent-Antwort: {lastResponse}
Neue Benutzereingabe: {userInput}

Entscheide:
1. Sollte der Buchungsagent die Kontrolle behalten? (JA/NEIN)
2. Confidence (0-1)
3. Begründung (kurz)
4. Falls NEIN, welcher Agent sollte übernehmen? (BookingAgent/SupportAgent/KnowledgeAgent/None)

Format: DECISION|CONFIDENCE|REASON|NEXT_AGENT";

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("Du bist ein Routing-Analyzer. Antworte nur im angegebenen Format.");
        chatHistory.AddUserMessage(prompt);
        
        var response = await _chatService.GetChatMessageContentAsync(chatHistory);
        var parts = response.Content?.Split('|') ?? new[] { "JA", "0.8", "Buchung läuft", "None" };
        
        var decision = new OwnershipDecision
        {
            KeepControl = parts[0].Trim().ToUpper() == "JA",
            Confidence = double.TryParse(parts[1], out var conf) ? conf : 0.8,
            Reason = parts.Length > 2 ? parts[2] : "Buchungsprozess aktiv",
            SuggestedNextAgent = parts.Length > 3 && parts[3] != "None" ? parts[3] : null
        };
        
        // Erhöhe Priorität bei kritischen Buchungsphasen
        if (state.Context.GetValueOrDefault("booking_stage")?.ToString() == "confirmation")
        {
            decision.Priority = 10; // Sehr hohe Priorität bei Bestätigung
            decision.KeepControl = true; // Erzwinge Kontrolle
            decision.Reason = "Kritische Buchungsphase - Bestätigung erforderlich";
        }
        else if (state.WorkflowStage == WorkflowStage.BookingInProgress)
        {
            decision.Priority = 5; // Mittlere Priorität bei aktiver Buchung
        }
        
        return decision;
    }
    
    /// <summary>
    /// Schlägt einen Nachfolge-Agent vor
    /// </summary>
    public async Task<string?> SuggestNextAgentAsync(string userInput, ConversationState state)
    {
        // Analysiere, welcher Agent am besten geeignet wäre
        var prompt = $@"
Welcher Agent sollte diese Anfrage übernehmen?
Benutzereingabe: {userInput}
Verfügbare Agents: BookingAgent, SupportAgent, KnowledgeAgent

Antworte NUR mit dem Agent-Namen oder 'None'.";

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("Du bist ein Agent-Selector. Antworte nur mit einem Agent-Namen.");
        chatHistory.AddUserMessage(prompt);
        
        var response = await _chatService.GetChatMessageContentAsync(chatHistory);
        var suggestion = response.Content?.Trim();
        
        return suggestion == "None" ? null : suggestion;
    }
    
    public override async Task<AgentResponse> ProcessAsync(string userInput, ConversationState conversationState)
    {
        var response = await base.ProcessAsync(userInput, conversationState);
        
        // Analysiere Ownership nach der Antwort
        var ownershipDecision = await AnalyzeOwnershipAsync(userInput, response.Message, conversationState);
        
        // Update Response mit Ownership-Informationen
        response.KeepControl = ownershipDecision.KeepControl;
        response.SuggestedNextAgent = ownershipDecision.SuggestedNextAgent;
        
        // Füge Ownership-Info zu Metadaten hinzu
        response.Metadata["ownership_confidence"] = ownershipDecision.Confidence;
        response.Metadata["ownership_reason"] = ownershipDecision.Reason;
        response.Metadata["ownership_priority"] = ownershipDecision.Priority;
        
        return response;
    }
}