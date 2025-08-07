using AgentRouterTest.Common.Models;

namespace AgentRouterTest.Common.Interfaces;

/// <summary>
/// Basis-Interface für alle Agents im System
/// </summary>
public interface IAgent
{
    /// <summary>
    /// Name des Agents
    /// </summary>
    string Name { get; }
    
    /// <summary>
    /// Beschreibung der Fähigkeiten des Agents
    /// </summary>
    string Description { get; }
    
    /// <summary>
    /// Bearbeitet eine Benutzeranfrage und gibt eine Antwort zurück
    /// </summary>
    /// <param name="userInput">Die Eingabe des Benutzers</param>
    /// <param name="conversationState">Der aktuelle Konversationszustand</param>
    /// <returns>Die Antwort des Agents</returns>
    Task<AgentResponse> ProcessAsync(string userInput, ConversationState conversationState);
    
    /// <summary>
    /// Prüft, ob dieser Agent für die gegebene Anfrage geeignet ist
    /// </summary>
    /// <param name="userInput">Die Eingabe des Benutzers</param>
    /// <param name="conversationState">Der aktuelle Konversationszustand</param>
    /// <returns>Confidence-Score zwischen 0 und 1</returns>
    Task<double> EvaluateSuitabilityAsync(string userInput, ConversationState conversationState);
}