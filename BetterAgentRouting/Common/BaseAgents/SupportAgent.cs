using Microsoft.SemanticKernel;
using AgentRouterTest.Common.Models;

namespace AgentRouterTest.Common.BaseAgents;

/// <summary>
/// Agent für technischen Support und Kundenservice
/// </summary>
public class SupportAgent : BaseAgent
{
    public override string Name => "SupportAgent";
    
    public override string Description => 
        "Spezialisiert auf technischen Support, Account-Probleme, Passwort-Resets, " +
        "Fehlerbehebung und allgemeine Kundenservice-Anfragen.";
    
    public SupportAgent(Kernel kernel) : base(kernel) { }
    
    protected override string GetSystemPrompt()
    {
        return @"
Du bist ein technischer Support-Agent mit Expertise in Kundenservice.
Deine Aufgaben:
- Technische Probleme lösen
- Account-bezogene Anfragen bearbeiten
- Passwort-Resets durchführen
- Software-Fehler diagnostizieren
- Anleitungen und Hilfestellungen geben
- Eskalationen managen

Gehe systematisch vor:
1. Verstehe das Problem genau
2. Stelle gezielte Rückfragen
3. Biete Schritt-für-Schritt-Lösungen
4. Überprüfe, ob das Problem gelöst wurde
5. Biete weitere Hilfe an

Sei geduldig, verständnisvoll und lösungsorientiert.
Erkläre technische Sachverhalte verständlich.";
    }
    
    protected override async Task<bool> ShouldKeepControlAsync(string userInput, string response, ConversationState state)
    {
        // Behalte Kontrolle bei aktivem Support-Fall
        if (state.WorkflowStage == WorkflowStage.SupportInProgress)
        {
            // Prüfe auf Abschluss-Signale
            var resolvedKeywords = new[] { "gelöst", "funktioniert", "danke", "erledigt", "hilft mir weiter" };
            if (resolvedKeywords.Any(keyword => userInput.ToLower().Contains(keyword)))
            {
                return false;
            }
            
            // Prüfe auf Themenwechsel
            var topicChangeKeywords = new[] { "andere frage", "noch etwas", "übrigens", "apropos" };
            if (topicChangeKeywords.Any(keyword => userInput.ToLower().Contains(keyword)))
            {
                return false;
            }
            
            return true; // Support-Fall noch aktiv
        }
        
        // Prüfe ob neue Support-Anfrage
        var supportKeywords = new[] { "problem", "fehler", "hilfe", "funktioniert nicht", "geht nicht", 
                                      "passwort", "account", "zugang", "anmelden", "error" };
        return await Task.FromResult(supportKeywords.Any(keyword => userInput.ToLower().Contains(keyword)));
    }
    
    protected override async Task<bool> IsWorkflowCompletedAsync(ConversationState state)
    {
        // Support-Workflow ist abgeschlossen, wenn Problem als gelöst markiert
        if (state.Context.ContainsKey("issue_resolved") && 
            state.Context["issue_resolved"] is bool resolved && 
            resolved)
        {
            return await Task.FromResult(true);
        }
        
        return await Task.FromResult(false);
    }
    
    public override async Task<AgentResponse> ProcessAsync(string userInput, ConversationState conversationState)
    {
        // Update Workflow-Stage wenn Support beginnt
        if (conversationState.WorkflowStage == WorkflowStage.Idle)
        {
            var supportKeywords = new[] { "problem", "fehler", "hilfe", "funktioniert nicht", "passwort" };
            if (supportKeywords.Any(keyword => userInput.ToLower().Contains(keyword)))
            {
                conversationState.WorkflowStage = WorkflowStage.SupportInProgress;
                conversationState.Context["support_started"] = DateTime.UtcNow;
            }
        }
        
        var response = await base.ProcessAsync(userInput, conversationState);
        
        // Tracke Support-Status
        UpdateSupportStatus(userInput, response.Message, conversationState);
        
        return response;
    }
    
    private void UpdateSupportStatus(string userInput, string response, ConversationState state)
    {
        // Prüfe ob Problem gelöst wurde
        var resolvedIndicators = new[] { "gelöst", "behoben", "funktioniert wieder", "vielen dank" };
        if (resolvedIndicators.Any(indicator => userInput.ToLower().Contains(indicator)))
        {
            state.Context["issue_resolved"] = true;
            state.WorkflowStage = WorkflowStage.Completed;
        }
        
        // Tracke Problemtyp
        if (!state.Context.ContainsKey("issue_type"))
        {
            if (userInput.ToLower().Contains("passwort"))
                state.Context["issue_type"] = "password_reset";
            else if (userInput.ToLower().Contains("account"))
                state.Context["issue_type"] = "account_issue";
            else if (userInput.ToLower().Contains("fehler") || userInput.ToLower().Contains("error"))
                state.Context["issue_type"] = "technical_error";
            else
                state.Context["issue_type"] = "general_support";
        }
        
        // Tracke Eskalationsstufe
        if (!state.Context.ContainsKey("escalation_level"))
        {
            state.Context["escalation_level"] = 1;
        }
        else if (state.CurrentAgentTurnCount > 5 && !state.Context.ContainsKey("issue_resolved"))
        {
            // Erhöhe Eskalation nach 5 Turns ohne Lösung
            state.Context["escalation_level"] = 2;
        }
    }
}