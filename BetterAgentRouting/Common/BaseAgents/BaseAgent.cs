using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using AgentRouterTest.Common.Interfaces;
using AgentRouterTest.Common.Models;
using System.Diagnostics;

namespace AgentRouterTest.Common.BaseAgents;

/// <summary>
/// Abstrakte Basisklasse für alle Agents
/// </summary>
public abstract class BaseAgent : IAgent
{
    protected readonly Kernel _kernel;
    protected readonly IChatCompletionService _chatService;
    
    public abstract string Name { get; }
    public abstract string Description { get; }
    
    protected BaseAgent(Kernel kernel)
    {
        _kernel = kernel;
        _chatService = kernel.GetRequiredService<IChatCompletionService>();
    }
    
    /// <summary>
    /// Verarbeitet eine Benutzeranfrage
    /// </summary>
    public virtual async Task<AgentResponse> ProcessAsync(string userInput, ConversationState conversationState)
    {
        var stopwatch = Stopwatch.StartNew();
        
        var systemPrompt = GetSystemPrompt();
        var contextPrompt = BuildContextPrompt(conversationState);
        
        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage($"{systemPrompt}\n\n{contextPrompt}");
        
        // Füge relevante Historie hinzu (letzte 5 Turns)
        var relevantHistory = conversationState.History.TakeLast(5);
        foreach (var turn in relevantHistory)
        {
            chatHistory.AddUserMessage(turn.UserInput);
            chatHistory.AddAssistantMessage(turn.AgentResponse);
        }
        
        chatHistory.AddUserMessage(userInput);
        
        var response = await _chatService.GetChatMessageContentAsync(chatHistory);
        
        stopwatch.Stop();
        
        return new AgentResponse
        {
            Message = response.Content ?? "Entschuldigung, ich konnte keine Antwort generieren.",
            KeepControl = await ShouldKeepControlAsync(userInput, response.Content ?? "", conversationState),
            WorkflowCompleted = await IsWorkflowCompletedAsync(conversationState),
            ResponseConfidence = await CalculateResponseConfidenceAsync(userInput, response.Content ?? ""),
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds
        };
    }
    
    /// <summary>
    /// Evaluiert die Eignung des Agents für eine Anfrage
    /// </summary>
    public virtual async Task<double> EvaluateSuitabilityAsync(string userInput, ConversationState conversationState)
    {
        var prompt = $@"
Bewerte auf einer Skala von 0 bis 1, wie gut der Agent '{Name}' für folgende Anfrage geeignet ist.
Agent-Beschreibung: {Description}
Aktueller Workflow-Status: {conversationState.WorkflowStage}
Benutzeranfrage: {userInput}

Antworte NUR mit einer Zahl zwischen 0 und 1 (z.B. 0.8).";

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("Du bist ein Routing-Evaluator. Antworte nur mit einer Dezimalzahl.");
        chatHistory.AddUserMessage(prompt);
        
        var response = await _chatService.GetChatMessageContentAsync(chatHistory);
        
        if (double.TryParse(response.Content?.Trim(), out var confidence))
        {
            return Math.Clamp(confidence, 0, 1);
        }
        
        return 0.5; // Fallback
    }
    
    /// <summary>
    /// Gibt den System-Prompt für den Agent zurück
    /// </summary>
    protected abstract string GetSystemPrompt();
    
    /// <summary>
    /// Baut den Kontext-Prompt basierend auf dem Konversationszustand
    /// </summary>
    protected virtual string BuildContextPrompt(ConversationState state)
    {
        return $@"
Konversations-Kontext:
- Session: {state.SessionId}
- Turn: {state.TurnCount}
- Workflow-Stadium: {state.WorkflowStage}
- Turns mit diesem Agent: {state.CurrentAgentTurnCount}";
    }
    
    /// <summary>
    /// Entscheidet, ob der Agent die Kontrolle behalten soll
    /// </summary>
    protected abstract Task<bool> ShouldKeepControlAsync(string userInput, string response, ConversationState state);
    
    /// <summary>
    /// Prüft, ob der Workflow abgeschlossen ist
    /// </summary>
    protected abstract Task<bool> IsWorkflowCompletedAsync(ConversationState state);
    
    /// <summary>
    /// Berechnet die Confidence für die gegebene Antwort
    /// </summary>
    protected virtual async Task<double> CalculateResponseConfidenceAsync(string userInput, string response)
    {
        // Basis-Implementierung: Immer hohe Confidence
        return await Task.FromResult(0.9);
    }
}