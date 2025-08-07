using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using AgentRouterTest.Common.Models;
using AgentRouterTest.Common.BaseAgents;

namespace AgentRouterTest.Approach2_AgentOwnership;

/// <summary>
/// Selbstverwalteter Support-Agent mit Ownership-Kontrolle
/// </summary>
public class SelfManagedSupportAgent : SupportAgent, ISelfManagedAgent
{
    private new readonly IChatCompletionService _chatService;
    
    public SelfManagedSupportAgent(Kernel kernel) : base(kernel) 
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
        var prompt = $@"
Analysiere ob der Support-Agent die Kontrolle behalten sollte.

Aktueller Support-Status: {state.WorkflowStage}
Problem-Typ: {state.Context.GetValueOrDefault("issue_type", "unknown")}
Problem gelöst: {state.Context.GetValueOrDefault("issue_resolved", false)}
Eskalationsstufe: {state.Context.GetValueOrDefault("escalation_level", 1)}
Letzte Agent-Antwort: {lastResponse}
Neue Benutzereingabe: {userInput}

Entscheide:
1. Sollte der Support-Agent die Kontrolle behalten? (JA/NEIN)
2. Confidence (0-1)
3. Begründung (kurz)
4. Falls NEIN, welcher Agent sollte übernehmen? (BookingAgent/SupportAgent/KnowledgeAgent/None)

Format: DECISION|CONFIDENCE|REASON|NEXT_AGENT";

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("Du bist ein Support-Routing-Analyzer. Antworte nur im angegebenen Format.");
        chatHistory.AddUserMessage(prompt);
        
        var response = await _chatService.GetChatMessageContentAsync(chatHistory);
        var parts = response.Content?.Split('|') ?? new[] { "JA", "0.7", "Support-Fall aktiv", "None" };
        
        var decision = new OwnershipDecision
        {
            KeepControl = parts[0].Trim().ToUpper() == "JA",
            Confidence = double.TryParse(parts[1], out var conf) ? conf : 0.7,
            Reason = parts.Length > 2 ? parts[2] : "Support-Fall wird bearbeitet",
            SuggestedNextAgent = parts.Length > 3 && parts[3] != "None" ? parts[3] : null
        };
        
        // Priorität basierend auf Eskalationsstufe
        var escalationLevel = Convert.ToInt32(state.Context.GetValueOrDefault("escalation_level", 1));
        decision.Priority = escalationLevel * 2;
        
        // Bei ungelösten kritischen Problemen Kontrolle behalten
        if (state.Context.GetValueOrDefault("issue_type")?.ToString() == "technical_error" &&
            !(bool)state.Context.GetValueOrDefault("issue_resolved", false))
        {
            decision.Priority = 8;
            decision.KeepControl = true;
            decision.Reason = "Technisches Problem noch nicht gelöst";
        }
        
        return decision;
    }
    
    /// <summary>
    /// Schlägt einen Nachfolge-Agent vor
    /// </summary>
    public async Task<string?> SuggestNextAgentAsync(string userInput, ConversationState state)
    {
        // Bei gelösten Problemen und neuen Fragen entsprechend weiterleiten
        if ((bool)state.Context.GetValueOrDefault("issue_resolved", false))
        {
            if (userInput.ToLower().Contains("buchen") || userInput.ToLower().Contains("flug"))
                return "BookingAgent";
            if (userInput.ToLower().Contains("was ist") || userInput.ToLower().Contains("erkläre"))
                return "KnowledgeAgent";
        }
        
        return null;
    }
    
    public override async Task<AgentResponse> ProcessAsync(string userInput, ConversationState conversationState)
    {
        var response = await base.ProcessAsync(userInput, conversationState);
        
        // Analysiere Ownership
        var ownershipDecision = await AnalyzeOwnershipAsync(userInput, response.Message, conversationState);
        
        // Update Response
        response.KeepControl = ownershipDecision.KeepControl;
        response.SuggestedNextAgent = ownershipDecision.SuggestedNextAgent;
        
        // Metadaten hinzufügen
        response.Metadata["ownership_confidence"] = ownershipDecision.Confidence;
        response.Metadata["ownership_reason"] = ownershipDecision.Reason;
        response.Metadata["ownership_priority"] = ownershipDecision.Priority;
        response.Metadata["escalation_level"] = conversationState.Context.GetValueOrDefault("escalation_level", 1);
        
        return response;
    }
}