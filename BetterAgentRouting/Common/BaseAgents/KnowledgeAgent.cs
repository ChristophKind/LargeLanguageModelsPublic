using Microsoft.SemanticKernel;
using AgentRouterTest.Common.Models;

namespace AgentRouterTest.Common.BaseAgents;

/// <summary>
/// Agent für allgemeine Wissensabfragen und Informationen
/// </summary>
public class KnowledgeAgent : BaseAgent
{
    public override string Name => "KnowledgeAgent";
    
    public override string Description => 
        "Spezialisiert auf allgemeine Wissensfragen, Definitionen, Erklärungen, " +
        "Fakten und informative Anfragen. Beantwortet Fragen zu verschiedensten Themen.";
    
    public KnowledgeAgent(Kernel kernel) : base(kernel) { }
    
    protected override string GetSystemPrompt()
    {
        return @"
Du bist ein Wissens-Agent mit breitem Allgemeinwissen.
Deine Aufgaben:
- Faktenfragen beantworten
- Konzepte und Begriffe erklären
- Historische Ereignisse erläutern
- Wissenschaftliche Zusammenhänge darstellen
- Definitionen liefern
- Vergleiche und Analysen durchführen

Antworte:
- Präzise und faktisch korrekt
- Verständlich und strukturiert
- Mit relevanten Beispielen
- Neutral und objektiv
- Bei Unsicherheit dies kennzeichnen

Vermeide:
- Spekulationen ohne Grundlage
- Persönliche Meinungen als Fakten
- Zu technische Sprache ohne Erklärung";
    }
    
    protected override async Task<bool> ShouldKeepControlAsync(string userInput, string response, ConversationState state)
    {
        // Knowledge Agent gibt normalerweise nach einer Antwort die Kontrolle ab
        if (state.WorkflowStage == WorkflowStage.KnowledgeQuery)
        {
            // Prüfe auf Folgefragen zum gleichen Thema
            var followUpIndicators = new[] { "kannst du mehr", "erkläre genauer", "was bedeutet", 
                                            "und was ist", "wie funktioniert das", "warum ist das so" };
            if (followUpIndicators.Any(indicator => userInput.ToLower().Contains(indicator)))
            {
                return true; // Folgefrage zum gleichen Thema
            }
            
            // Bei neuer Frage oder Themenwechsel Kontrolle abgeben
            return false;
        }
        
        // Übernehme bei Wissensfragen
        var knowledgeKeywords = new[] { "was ist", "wer ist", "wie funktioniert", "erkläre", 
                                       "definition", "bedeutung", "warum", "woher", "wann" };
        return await Task.FromResult(knowledgeKeywords.Any(keyword => userInput.ToLower().Contains(keyword)));
    }
    
    protected override async Task<bool> IsWorkflowCompletedAsync(ConversationState state)
    {
        // Wissensfragen sind meist in einem Turn abgeschlossen
        if (state.WorkflowStage == WorkflowStage.KnowledgeQuery && state.CurrentAgentTurnCount > 0)
        {
            return await Task.FromResult(true);
        }
        
        return await Task.FromResult(false);
    }
    
    public override async Task<AgentResponse> ProcessAsync(string userInput, ConversationState conversationState)
    {
        // Update Workflow-Stage für Wissensfragen
        if (conversationState.WorkflowStage == WorkflowStage.Idle)
        {
            var knowledgeKeywords = new[] { "was ist", "wer ist", "wie funktioniert", "erkläre", "definition" };
            if (knowledgeKeywords.Any(keyword => userInput.ToLower().Contains(keyword)))
            {
                conversationState.WorkflowStage = WorkflowStage.KnowledgeQuery;
                conversationState.Context["query_type"] = "knowledge";
            }
        }
        
        var response = await base.ProcessAsync(userInput, conversationState);
        
        // Tracke Fragetyp
        CategorizeQuestion(userInput, conversationState);
        
        // Bei einfachen Faktenfragen meist keine weitere Kontrolle nötig
        if (IsSimpleFactQuestion(userInput))
        {
            response.KeepControl = false;
            response.WorkflowCompleted = true;
        }
        
        return response;
    }
    
    private void CategorizeQuestion(string userInput, ConversationState state)
    {
        var input = userInput.ToLower();
        
        if (!state.Context.ContainsKey("question_category"))
        {
            if (input.Contains("definition") || input.Contains("was ist") || input.Contains("was bedeutet"))
                state.Context["question_category"] = "definition";
            else if (input.Contains("wie funktioniert") || input.Contains("wie geht"))
                state.Context["question_category"] = "how_to";
            else if (input.Contains("warum") || input.Contains("weshalb"))
                state.Context["question_category"] = "reasoning";
            else if (input.Contains("wer") || input.Contains("person"))
                state.Context["question_category"] = "person";
            else if (input.Contains("wann") || input.Contains("jahr") || input.Contains("datum"))
                state.Context["question_category"] = "temporal";
            else if (input.Contains("wo") || input.Contains("ort") || input.Contains("land"))
                state.Context["question_category"] = "location";
            else
                state.Context["question_category"] = "general";
        }
    }
    
    private bool IsSimpleFactQuestion(string userInput)
    {
        var simpleQuestionPatterns = new[] 
        { 
            "was ist die hauptstadt",
            "wer ist der präsident",
            "wann wurde",
            "wie hoch ist",
            "wie viele",
            "welches jahr"
        };
        
        return simpleQuestionPatterns.Any(pattern => userInput.ToLower().Contains(pattern));
    }
}