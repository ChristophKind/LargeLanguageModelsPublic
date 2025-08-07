using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using AgentRouterTest.Approach1_StatefulRouting;
using AgentRouterTest.Approach2_AgentOwnership;
using AgentRouterTest.Approach3_StickySessions;

namespace AgentRouterTest;

/// <summary>
/// Hauptprogramm für die Agent-Router Test-Anwendung
/// Demonstriert drei verschiedene Router-Architekturen für persistente Agent-Workflows
/// </summary>
class Program
{
    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        
        // Lade .env Datei falls vorhanden
        DotNetEnv.Env.Load();
        
        // Lade Konfiguration (Environment Variables haben Priorität über appsettings.json)
        var configuration = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
            .AddEnvironmentVariables()
            .Build();
        
        // Setup Logging
        var serviceCollection = new ServiceCollection();
        serviceCollection.AddLogging(builder =>
        {
            builder.AddConsole();
            builder.SetMinimumLevel(LogLevel.Information);
        });
        
        var serviceProvider = serviceCollection.BuildServiceProvider();
        var loggerFactory = serviceProvider.GetRequiredService<ILoggerFactory>();
        
        // Erstelle Kernel mit OpenAI
        var kernel = CreateKernel(configuration);
        
        Console.Clear();
        PrintWelcomeBanner();
        
        while (true)
        {
            Console.WriteLine("\n" + new string('═', 80));
            Console.WriteLine("Wählen Sie einen Router-Ansatz:");
            Console.WriteLine("" + new string('─', 80));
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("1. Stateful Routing mit Context-Awareness");
            Console.ResetColor();
            Console.WriteLine("   → Dynamische Confidence-Schwellenwerte basierend auf Workflow-Status");
            
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n2. Agent-Ownership Pattern");
            Console.ResetColor();
            Console.WriteLine("   → Agents entscheiden selbst über Kontrollübergabe");
            
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n3. Sticky Sessions mit Exit-Detection");
            Console.ResetColor();
            Console.WriteLine("   → Router bleibt bei Agent bis explizites Exit-Signal");
            
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("\n0. Beenden");
            Console.ResetColor();
            Console.WriteLine(new string('═', 80));
            Console.Write("\nIhre Wahl: ");
            
            var choice = Console.ReadLine();
            
            switch (choice)
            {
                case "1":
                    await RunStatefulRouting(kernel, loggerFactory);
                    break;
                case "2":
                    await RunAgentOwnership(kernel, loggerFactory);
                    break;
                case "3":
                    await RunStickySessions(kernel, loggerFactory);
                    break;
                case "0":
                    Console.WriteLine("\nAuf Wiedersehen!");
                    return;
                default:
                    Console.WriteLine("Ungültige Eingabe. Bitte versuchen Sie es erneut.");
                    break;
            }
        }
    }
    
    /// <summary>
    /// Erstellt den Semantic Kernel mit OpenAI-Konfiguration
    /// </summary>
    static Kernel CreateKernel(IConfiguration configuration)
    {
        var builder = Kernel.CreateBuilder();
        
        // Priorität: Environment Variables > appsettings.json
        var apiKey = configuration["OPENAI_API_KEY"] ?? configuration["OpenAI:ApiKey"];
        var model = configuration["OPENAI_MODEL"] ?? configuration["OpenAI:Model"];
        
        if (string.IsNullOrEmpty(apiKey))
        {
            throw new InvalidOperationException("OpenAI API Key nicht gefunden. Bitte setzen Sie OPENAI_API_KEY in der .env Datei oder OpenAI:ApiKey in appsettings.json");
        }
        
        builder.AddOpenAIChatCompletion(
            modelId: model ?? "gpt-4o-mini",
            apiKey: apiKey
        );
        
        return builder.Build();
    }
    
    /// <summary>
    /// Führt Ansatz 1: Stateful Routing aus
    /// </summary>
    static async Task RunStatefulRouting(Kernel kernel, ILoggerFactory loggerFactory)
    {
        Console.Clear();
        PrintApproachHeader("Stateful Routing mit Context-Awareness", ConsoleColor.Yellow);
        
        var orchestrator = new StatefulRoutingOrchestrator(kernel, loggerFactory);
        var sessionId = orchestrator.StartNewSession();
        
        Console.WriteLine("\n📊 Dieser Ansatz verwendet dynamische Confidence-Schwellenwerte.");
        Console.WriteLine("Bei aktiven Workflows (z.B. Buchungen) sind höhere Schwellenwerte");
        Console.WriteLine("erforderlich, um den Agent zu wechseln.\n");
        
        await RunChatLoop(
            sessionId,
            async (input) =>
            {
                var (agentName, response, routingInfo) = await orchestrator.ProcessMessageAsync(sessionId, input);
                return (agentName, response, routingInfo.Reason);
            },
            () => orchestrator.GetSessionStatistics(sessionId)
        );
    }
    
    /// <summary>
    /// Führt Ansatz 2: Agent-Ownership Pattern aus
    /// </summary>
    static async Task RunAgentOwnership(Kernel kernel, ILoggerFactory loggerFactory)
    {
        Console.Clear();
        PrintApproachHeader("Agent-Ownership Pattern", ConsoleColor.Cyan);
        
        var orchestrator = new OwnershipOrchestrator(kernel, loggerFactory);
        var sessionId = orchestrator.StartNewSession();
        
        Console.WriteLine("\n🤝 Agents verwalten sich selbst und entscheiden über Kontrollübergabe.");
        Console.WriteLine("Jeder Agent kann die Kontrolle behalten (keep_control) oder");
        Console.WriteLine("einen Nachfolge-Agent vorschlagen.\n");
        
        await RunChatLoop(
            sessionId,
            async (input) =>
            {
                var (agentName, response, routingInfo, metadata) = 
                    await orchestrator.ProcessMessageAsync(sessionId, input);
                
                var details = $"{routingInfo.Reason}";
                if (metadata.ContainsKey("keep_control"))
                {
                    details += $" | Keep Control: {metadata["keep_control"]}";
                }
                if (metadata.ContainsKey("suggested_next") && metadata["suggested_next"]?.ToString() != "None")
                {
                    details += $" | Suggested: {metadata["suggested_next"]}";
                }
                
                return (agentName, response, details);
            },
            () => orchestrator.GetOwnershipInfo(sessionId)
        );
    }
    
    /// <summary>
    /// Führt Ansatz 3: Sticky Sessions aus
    /// </summary>
    static async Task RunStickySessions(Kernel kernel, ILoggerFactory loggerFactory)
    {
        Console.Clear();
        PrintApproachHeader("Sticky Sessions mit Exit-Detection", ConsoleColor.Green);
        
        var orchestrator = new StickyRoutingOrchestrator(kernel, loggerFactory);
        var sessionId = orchestrator.StartNewSession();
        
        Console.WriteLine("\n📌 Der Router bleibt \"klebrig\" beim aktuellen Agent.");
        Console.WriteLine("Nur bei expliziten Exit-Signalen (\"andere Frage\", \"fertig\", etc.)");
        Console.WriteLine("oder klaren Themenwechseln wird ein neuer Agent gewählt.\n");
        
        await RunChatLoop(
            sessionId,
            async (input) =>
            {
                var result = await orchestrator.ProcessMessageAsync(sessionId, input);
                
                var details = $"{result.RoutingInfo?.Reason}";
                if (result.SessionSticky)
                {
                    details = "📌 " + details;
                }
                if (result.ExitDetection?.ExitDetected == true)
                {
                    details += $" | Exit: {result.ExitDetection.ExitType}";
                }
                if (result.SuggestUserPrompt != null)
                {
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine($"\n💡 Tipp: {result.SuggestUserPrompt}");
                    Console.ResetColor();
                }
                
                return (result.AgentName, result.Response, details);
            },
            () => orchestrator.GetSessionMetrics(sessionId)
        );
    }
    
    /// <summary>
    /// Gemeinsame Chat-Loop für alle Ansätze
    /// </summary>
    static async Task RunChatLoop(
        string sessionId,
        Func<string, Task<(string agentName, string response, string details)>> processMessage,
        Func<Dictionary<string, object>> getStats)
    {
        Console.WriteLine("Chat gestartet. Geben Sie 'exit' ein zum Beenden, 'stats' für Statistiken.\n");
        Console.WriteLine(new string('─', 80));
        
        while (true)
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("\n👤 Sie: ");
            Console.ResetColor();
            var input = Console.ReadLine();
            
            if (string.IsNullOrWhiteSpace(input))
                continue;
            
            if (input.ToLower() == "exit")
            {
                Console.WriteLine("\nChat beendet.");
                break;
            }
            
            if (input.ToLower() == "stats")
            {
                ShowStatistics(getStats());
                continue;
            }
            
            try
            {
                var startTime = DateTime.Now;
                var (agentName, response, details) = await processMessage(input);
                var totalTime = (DateTime.Now - startTime).TotalMilliseconds;
                
                // Zeige Routing-Details
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"\n[{details}]");
                Console.ResetColor();
                
                // Zeige Performance-Metriken
                Console.ForegroundColor = ConsoleColor.DarkCyan;
                Console.WriteLine($"⏱️  Gesamtzeit: {totalTime:F0}ms");
                Console.ResetColor();
                
                // Zeige Agent-Antwort
                Console.ForegroundColor = GetAgentColor(agentName);
                Console.Write($"\n🤖 {agentName}: ");
                Console.ResetColor();
                Console.WriteLine(response);
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n❌ Fehler: {ex.Message}");
                Console.ResetColor();
            }
        }
    }
    
    /// <summary>
    /// Zeigt Session-Statistiken an
    /// </summary>
    static void ShowStatistics(Dictionary<string, object> stats)
    {
        Console.WriteLine("\n" + new string('─', 40));
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("📊 Session-Statistiken:");
        Console.ResetColor();
        
        foreach (var stat in stats)
        {
            Console.WriteLine($"  {stat.Key}: {stat.Value}");
        }
        Console.WriteLine(new string('─', 40));
    }
    
    /// <summary>
    /// Gibt die Farbe für einen Agent zurück
    /// </summary>
    static ConsoleColor GetAgentColor(string agentName)
    {
        return agentName switch
        {
            "BookingAgent" or "SelfManagedBookingAgent" => ConsoleColor.Blue,
            "SupportAgent" or "SelfManagedSupportAgent" => ConsoleColor.Magenta,
            "KnowledgeAgent" or "SelfManagedKnowledgeAgent" => ConsoleColor.Green,
            _ => ConsoleColor.Gray
        };
    }
    
    /// <summary>
    /// Zeigt das Willkommensbanner an
    /// </summary>
    static void PrintWelcomeBanner()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine(@"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     █████╗  ██████╗ ███████╗███╗   ██╗████████╗    ██████╗  ██████╗ ██╗   ██╗████████╗███████╗██████╗ 
║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝    ██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝██╔════╝██╔══██╗
║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║       ██████╔╝██║   ██║██║   ██║   ██║   █████╗  ██████╔╝
║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║       ██╔══██╗██║   ██║██║   ██║   ██║   ██╔══╝  ██╔══██╗
║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║       ██║  ██║╚██████╔╝╚██████╔╝   ██║   ███████╗██║  ██║
║    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝
║                                                                              ║
║               Persistente Agent-Workflows mit Microsoft Semantic Kernel      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝");
        Console.ResetColor();
        
        Console.WriteLine("\nWillkommen zum Agent Router Test System!");
        Console.WriteLine("Dieses System demonstriert drei verschiedene Ansätze für persistente");
        Console.WriteLine("Agent-Workflows mit Microsoft Semantic Kernel.\n");
        
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine("Verfügbare Agents:");
        Console.ForegroundColor = ConsoleColor.Blue;
        Console.WriteLine("  • BookingAgent - Flug- und Hotelbuchungen");
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine("  • SupportAgent - Technischer Support und Hilfe");
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("  • KnowledgeAgent - Allgemeine Wissensfragen");
        Console.ResetColor();
    }
    
    /// <summary>
    /// Zeigt den Header für einen Ansatz an
    /// </summary>
    static void PrintApproachHeader(string approachName, ConsoleColor color)
    {
        Console.ForegroundColor = color;
        Console.WriteLine("\n" + new string('═', 80));
        Console.WriteLine($"  {approachName}");
        Console.WriteLine(new string('═', 80));
        Console.ResetColor();
    }
}