using AgentRouterTest.Common.Models;

namespace AgentRouterTest.Common.Interfaces;

/// <summary>
/// Interface für Router-Implementierungen
/// </summary>
public interface IRouter
{
    /// <summary>
    /// Name des Routing-Ansatzes
    /// </summary>
    string ApproachName { get; }
    
    /// <summary>
    /// Wählt den besten Agent für die gegebene Anfrage aus
    /// </summary>
    /// <param name="userInput">Die Eingabe des Benutzers</param>
    /// <param name="conversationState">Der aktuelle Konversationszustand</param>
    /// <param name="availableAgents">Liste der verfügbaren Agents</param>
    /// <returns>Das Routing-Ergebnis mit dem gewählten Agent</returns>
    Task<RoutingResult> RouteAsync(string userInput, ConversationState conversationState, List<IAgent> availableAgents);
}