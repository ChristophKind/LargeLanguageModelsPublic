using Microsoft.SemanticKernel;
using AgentRouterTest.Common.Models;

namespace AgentRouterTest.Common.BaseAgents;

/// <summary>
/// Agent für Buchungsanfragen (Flüge, Hotels, etc.)
/// </summary>
public class BookingAgent : BaseAgent
{
    public override string Name => "BookingAgent";
    
    public override string Description => 
        "Spezialisiert auf Flug- und Hotelbuchungen, Verfügbarkeitsabfragen, " +
        "Preisvergleiche und Reiseplanung. Kann komplette Buchungsprozesse durchführen.";
    
    public BookingAgent(Kernel kernel) : base(kernel) { }
    
    protected override string GetSystemPrompt()
    {
        return @"
Du bist ein spezialisierter Buchungsagent für Reisen und Unterkünfte.
Deine Aufgaben:
- Flugbuchungen (Suche, Vergleich, Buchung)
- Hotelbuchungen (Verfügbarkeit, Preise, Reservierungen)
- Mietwagenbuchungen
- Reiseplanung und -beratung
- Umbuchungen und Stornierungen

Führe Buchungsprozesse schrittweise durch:
1. Sammle alle notwendigen Informationen (Reisedaten, Präferenzen, Budget)
2. Präsentiere Optionen
3. Bestätige die Auswahl
4. Führe die Buchung durch

Sei freundlich, professionell und führe den Benutzer durch den gesamten Prozess.
Stelle sicher, dass alle wichtigen Details geklärt sind, bevor du eine Buchung abschließt.";
    }
    
    protected override async Task<bool> ShouldKeepControlAsync(string userInput, string response, ConversationState state)
    {
        // Behalte Kontrolle bei aktiver Buchung
        if (state.WorkflowStage == WorkflowStage.BookingInProgress)
        {
            // Prüfe auf Abbruch-Signale
            var exitKeywords = new[] { "abbrechen", "stopp", "andere frage", "vergiss es", "danke, das war's" };
            if (exitKeywords.Any(keyword => userInput.ToLower().Contains(keyword)))
            {
                return false;
            }
            
            // Prüfe ob Buchung abgeschlossen
            var completionKeywords = new[] { "buchung abgeschlossen", "vielen dank", "perfekt, gebucht" };
            if (completionKeywords.Any(keyword => response.ToLower().Contains(keyword)))
            {
                return false;
            }
            
            return true; // Standardmäßig Kontrolle behalten
        }
        
        // Wenn Benutzer eine Buchungsanfrage stellt, Kontrolle übernehmen
        var bookingKeywords = new[] { "buchen", "flug", "hotel", "reise", "reservieren" };
        return await Task.FromResult(bookingKeywords.Any(keyword => userInput.ToLower().Contains(keyword)));
    }
    
    protected override async Task<bool> IsWorkflowCompletedAsync(ConversationState state)
    {
        // Workflow ist abgeschlossen, wenn Buchung bestätigt wurde
        if (state.Context.ContainsKey("booking_confirmed") && 
            state.Context["booking_confirmed"] is bool confirmed && 
            confirmed)
        {
            return await Task.FromResult(true);
        }
        
        return await Task.FromResult(false);
    }
    
    public override async Task<AgentResponse> ProcessAsync(string userInput, ConversationState conversationState)
    {
        // Update Workflow-Stage wenn Buchung beginnt
        if (conversationState.WorkflowStage == WorkflowStage.Idle)
        {
            var bookingKeywords = new[] { "buchen", "flug", "hotel", "reise", "reservieren" };
            if (bookingKeywords.Any(keyword => userInput.ToLower().Contains(keyword)))
            {
                conversationState.WorkflowStage = WorkflowStage.BookingInProgress;
            }
        }
        
        var response = await base.ProcessAsync(userInput, conversationState);
        
        // Simuliere Buchungsfortschritt
        if (!conversationState.Context.ContainsKey("booking_stage"))
        {
            conversationState.Context["booking_stage"] = "initial";
        }
        
        // Update Buchungsstatus basierend auf Konversation
        UpdateBookingStage(userInput, response.Message, conversationState);
        
        return response;
    }
    
    private void UpdateBookingStage(string userInput, string response, ConversationState state)
    {
        var currentStage = state.Context["booking_stage"]?.ToString() ?? "initial";
        
        // Vereinfachte Staging-Logik
        if (currentStage == "initial" && userInput.ToLower().Contains("flug"))
        {
            state.Context["booking_stage"] = "flight_search";
        }
        else if (currentStage == "flight_search" && response.ToLower().Contains("option"))
        {
            state.Context["booking_stage"] = "selection";
        }
        else if (currentStage == "selection" && userInput.ToLower().Contains("nehme"))
        {
            state.Context["booking_stage"] = "confirmation";
        }
        else if (currentStage == "confirmation" && userInput.ToLower().Contains("bestätig"))
        {
            state.Context["booking_stage"] = "completed";
            state.Context["booking_confirmed"] = true;
            state.WorkflowStage = WorkflowStage.Completed;
        }
    }
}