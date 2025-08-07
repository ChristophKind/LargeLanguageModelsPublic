using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using AgentRouterTest.Common.Models;
using AgentRouterTest.Common.BaseAgents;

namespace AgentRouterTest.Approach2_AgentOwnership;

/// <summary>
/// Selbstverwalteter Wissens-Agent mit Ownership-Kontrolle
/// </summary>
public class SelfManagedKnowledgeAgent : KnowledgeAgent, ISelfManagedAgent
{
    private new readonly IChatCompletionService _chatService;
    
    public SelfManagedKnowledgeAgent(Kernel kernel) : base(kernel) 
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
Analysiere ob der Knowledge-Agent die Kontrolle behalten sollte.

Frage-Kategorie: {state.Context.GetValueOrDefault("question_category", "general")}
Letzte Agent-Antwort: {lastResponse}
Neue Benutzereingabe: {userInput}

Ist dies eine Folgefrage zum gleichen Thema oder eine neue Anfrage?

Entscheide:
1. Sollte der Knowledge-Agent die Kontrolle behalten? (JA/NEIN)
2. Confidence (0-1)
3. Begründung (kurz)
4. Falls NEIN, welcher Agent sollte übernehmen? (BookingAgent/SupportAgent/KnowledgeAgent/None)

Format: DECISION|CONFIDENCE|REASON|NEXT_AGENT";

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("Du bist ein Knowledge-Routing-Analyzer. Antworte nur im angegebenen Format.");
        chatHistory.AddUserMessage(prompt);
        
        var response = await _chatService.GetChatMessageContentAsync(chatHistory);
        var parts = response.Content?.Split('|') ?? new[] { "NEIN", "0.3", "Frage beantwortet", "None" };
        
        var decision = new OwnershipDecision
        {
            KeepControl = parts[0].Trim().ToUpper() == "JA",
            Confidence = double.TryParse(parts[1], out var conf) ? conf : 0.3,
            Reason = parts.Length > 2 ? parts[2] : "Wissensfrage beantwortet",
            SuggestedNextAgent = parts.Length > 3 && parts[3] != "None" ? parts[3] : null,
            Priority = 1 // Knowledge Agent hat niedrige Priorität
        };
        
        // Bei Folgefragen zum gleichen Thema Priorität erhöhen
        if (IsFollowUpQuestion(userInput, lastResponse))
        {
            decision.KeepControl = true;
            decision.Priority = 3;
            decision.Reason = "Folgefrage zum gleichen Thema";
            decision.Confidence = 0.8;
        }
        
        return decision;
    }
    
    /// <summary>
    /// Schlägt einen Nachfolge-Agent vor
    /// </summary>
    public async Task<string?> SuggestNextAgentAsync(string userInput, ConversationState state)
    {
        // Analysiere die neue Anfrage für passenden Agent
        if (userInput.ToLower().Contains("buchen") || 
            userInput.ToLower().Contains("flug") || 
            userInput.ToLower().Contains("hotel"))
        {
            return "BookingAgent";
        }
        
        if (userInput.ToLower().Contains("problem") || 
            userInput.ToLower().Contains("fehler") || 
            userInput.ToLower().Contains("hilfe"))
        {
            return "SupportAgent";
        }
        
        // Bei weiteren Wissensfragen bei sich selbst bleiben
        if (userInput.ToLower().Contains("was ist") || 
            userInput.ToLower().Contains("erkläre"))
        {
            return "KnowledgeAgent";
        }
        
        return null;
    }
    
    private bool IsFollowUpQuestion(string userInput, string lastResponse)
    {
        var followUpIndicators = new[] 
        { 
            "mehr dazu", 
            "genauer", 
            "beispiel", 
            "und was", 
            "aber wie",
            "verstehe nicht",
            "kannst du das"
        };
        
        return followUpIndicators.Any(indicator => userInput.ToLower().Contains(indicator));
    }
    
    public override async Task<AgentResponse> ProcessAsync(string userInput, ConversationState conversationState)
    {
        var response = await base.ProcessAsync(userInput, conversationState);
        
        // Analysiere Ownership
        var ownershipDecision = await AnalyzeOwnershipAsync(userInput, response.Message, conversationState);
        
        // Knowledge Agent gibt normalerweise schnell ab
        response.KeepControl = ownershipDecision.KeepControl;
        response.SuggestedNextAgent = ownershipDecision.SuggestedNextAgent;
        
        // Metadaten
        response.Metadata["ownership_confidence"] = ownershipDecision.Confidence;
        response.Metadata["ownership_reason"] = ownershipDecision.Reason;
        response.Metadata["ownership_priority"] = ownershipDecision.Priority;
        response.Metadata["question_category"] = conversationState.Context.GetValueOrDefault("question_category", "general");
        
        return response;
    }
}