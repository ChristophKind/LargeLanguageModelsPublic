# Router-Architekturen Visualisierung

## Übersicht: Die drei Router-Ansätze

```mermaid
graph TB
    User[User Input]
    
    subgraph "Ansatz 1: Stateful Routing"
        SR[Stateful Router]
        CS1[Conversation State]
        DT[Dynamic Threshold]
        SR --> CS1
        SR --> DT
    end
    
    subgraph "Ansatz 2: Agent Ownership"
        AO[Ownership Router]
        SM[Self-Managed Agents]
        OD[Ownership Decision]
        AO --> SM
        SM --> OD
    end
    
    subgraph "Ansatz 3: Sticky Sessions"
        SS[Sticky Router]
        ED[Exit Detection]
        TC[Topic Change Detection]
        SS --> ED
        SS --> TC
    end
    
    User --> SR
    User --> AO
    User --> SS
    
    subgraph "Verfügbare Agents"
        BA[BookingAgent]
        SA[SupportAgent]
        KA[KnowledgeAgent]
    end
    
    SR --> BA
    SR --> SA
    SR --> KA
    AO --> BA
    AO --> SA
    AO --> KA
    SS --> BA
    SS --> SA
    SS --> KA
```

## Ansatz 1: Stateful Routing Ablauf

```mermaid
flowchart TD
    Start([User Input]) --> GetState[Hole Conversation State]
    GetState --> CalcThreshold[Berechne Dynamic Threshold]
    
    CalcThreshold --> CheckWorkflow{Workflow aktiv?}
    CheckWorkflow -->|Ja| AddBonus[Füge Workflow-Bonus hinzu<br/>+0.3 für Booking<br/>+0.2 für Support]
    CheckWorkflow -->|Nein| EvalAgents
    
    AddBonus --> TurnBonus[Füge Turn-Count Bonus hinzu<br/>max +0.2]
    TurnBonus --> EvalAgents[Evaluiere alle Agents]
    
    EvalAgents --> CompareScores{Neuer Agent<br/>signifikant besser?}
    CompareScores -->|Ja, übersteigt Threshold| SwitchAgent[Wechsel zu neuem Agent]
    CompareScores -->|Nein| StayAgent[Bleibe bei aktuellem Agent]
    
    SwitchAgent --> UpdateState1[Update Workflow Stage]
    StayAgent --> UpdateState2[Update Turn Count]
    
    UpdateState1 --> Response([Agent Response])
    UpdateState2 --> Response
    
    style AddBonus fill:#ffeb3b
    style CalcThreshold fill:#4caf50
    style CompareScores fill:#ff9800
```

## Ansatz 2: Agent Ownership Flow

```mermaid
flowchart TD
    Start([User Input]) --> CheckCurrent{Aktueller Agent<br/>vorhanden?}
    
    CheckCurrent -->|Ja| AnalyzeOwnership[Agent analysiert Ownership]
    CheckCurrent -->|Nein| NormalRoute
    
    AnalyzeOwnership --> CheckPriority{Priority > 5?}
    
    CheckPriority -->|Ja| CheckKeep{Agent will<br/>Kontrolle behalten?}
    CheckPriority -->|Nein| NormalRoute[Normales Routing]
    
    CheckKeep -->|Ja| ForceControl[Agent erzwingt Kontrolle]
    CheckKeep -->|Nein| CheckSuggestion{Agent schlägt<br/>Nachfolger vor?}
    
    CheckSuggestion -->|Ja| UseSuggestion[Nutze vorgeschlagenen Agent]
    CheckSuggestion -->|Nein| NormalRoute
    
    ForceControl --> CriticalPhase[Kritische Phase<br/>z.B. Zahlungsbestätigung]
    CriticalPhase --> Response([Agent Response])
    
    UseSuggestion --> Response
    NormalRoute --> EvalAll[Evaluiere alle Agents]
    EvalAll --> SelectBest[Wähle besten Agent]
    SelectBest --> Response
    
    style ForceControl fill:#f44336
    style CriticalPhase fill:#ff5722
    style CheckPriority fill:#9c27b0
```

## Ansatz 3: Sticky Sessions Mechanismus

```mermaid
flowchart TD
    Start([User Input]) --> HasAgent{Aktueller Agent<br/>vorhanden?}
    
    HasAgent -->|Nein| NormalRoute[Normales Routing]
    HasAgent -->|Ja| DetectExit[Prüfe Exit-Intent]
    
    DetectExit --> CheckPhrases{Exit-Phrasen<br/>gefunden?}
    CheckPhrases -->|Ja<br/>'stop', 'fertig', 'danke'| ExitConfirmed[Exit bestätigt<br/>Confidence: 0.9]
    CheckPhrases -->|Nein| CheckTopic[Prüfe Themenwechsel]
    
    CheckTopic --> TopicChanged{Thema<br/>gewechselt?}
    TopicChanged -->|Ja| ExitPossible[Exit möglich<br/>Confidence: 0.7]
    TopicChanged -->|Nein| StaySticky[Bleibe kleben]
    
    ExitConfirmed --> CheckThreshold{Confidence ><br/>Exit-Threshold?}
    ExitPossible --> CheckThreshold
    
    CheckThreshold -->|Ja, > 0.6| AllowSwitch[Erlaube Wechsel]
    CheckThreshold -->|Nein| StaySticky
    
    StaySticky --> KeepAgent[Behalte aktuellen Agent]
    AllowSwitch --> NormalRoute
    
    KeepAgent --> Response([Agent Response<br/>'Sticky Session aktiv'])
    NormalRoute --> SelectNew[Wähle neuen Agent]
    SelectNew --> Response2([Agent Response<br/>'Neuer Agent'])
    
    style StaySticky fill:#4caf50
    style ExitConfirmed fill:#f44336
    style CheckThreshold fill:#ff9800
```

## Interaktion zwischen Komponenten

```mermaid
sequenceDiagram
    participant U as User
    participant R as Router
    participant CS as ConversationState
    participant A as Agent
    participant LLM as LLM Service
    
    U->>R: User Input
    R->>CS: Get Current State
    CS-->>R: State Info
    
    alt Stateful Routing
        R->>R: Calculate Dynamic Threshold
        R->>R: Add Workflow Bonus
        R->>A: Evaluate Suitability
        A->>LLM: Analyze Input
        LLM-->>A: Suitability Score
        A-->>R: Score + Bonus
    else Agent Ownership
        R->>A: Analyze Ownership
        A->>A: Check Critical Phase
        A->>LLM: Should Keep Control?
        LLM-->>A: Decision
        A-->>R: Ownership Decision
        Note over R: Respects Agent Decision
    else Sticky Sessions
        R->>R: Detect Exit Intent
        R->>LLM: Check Topic Change
        LLM-->>R: Exit Detection Result
        alt No Exit Detected
            R->>R: Stay with Current Agent
        else Exit Detected
            R->>R: Allow Agent Switch
        end
    end
    
    R->>CS: Update State
    R->>A: Process Input
    A->>LLM: Generate Response
    LLM-->>A: Response
    A-->>R: Final Response
    R-->>U: Agent Response
```

## Entscheidungsbaum für Router-Auswahl

```mermaid
graph TD
    Start([Welcher Router?]) --> Q1{Sind Workflows<br/>komplex und mehrstufig?}
    
    Q1 -->|Ja| Q2{Müssen Agents<br/>Kontrolle erzwingen können?}
    Q1 -->|Nein| Q3{Wechseln User<br/>oft das Thema?}
    
    Q2 -->|Ja| UseOwnership[Agent Ownership<br/>Pattern]
    Q2 -->|Nein| Q4{Sind Workflow-Prioritäten<br/>dynamisch?}
    
    Q3 -->|Ja| UseStateful[Stateful Routing]
    Q3 -->|Nein| UseSticky[Sticky Sessions]
    
    Q4 -->|Ja| UseStateful
    Q4 -->|Nein| UseSticky
    
    UseOwnership --> Features1[✓ Agents entscheiden selbst<br/>✓ Kritische Phasen geschützt<br/>✓ Domain-Expertise]
    UseStateful --> Features2[✓ Dynamische Schwellen<br/>✓ Workflow-Awareness<br/>✓ Flexible Anpassung]
    UseSticky --> Features3[✓ Maximale Stabilität<br/>✓ Einfach & vorhersagbar<br/>✓ Klare Exit-Signale]
    
    style UseOwnership fill:#9c27b0
    style UseStateful fill:#4caf50
    style UseSticky fill:#2196f3
```

## Performance-Vergleich

```mermaid
graph LR
    subgraph "Metriken"
        direction TB
        M1[Routing-Latenz]
        M2[Agent-Wechsel-Rate]
        M3[Workflow-Completion]
        M4[User-Satisfaction]
    end
    
    subgraph "Stateful Routing"
        direction TB
        SR1[Medium Latenz<br/>~150ms]
        SR2[Medium Wechsel<br/>15-20%]
        SR3[Hoch<br/>85%]
        SR4[Gut<br/>7.5/10]
    end
    
    subgraph "Agent Ownership"
        direction TB
        AO1[Höhere Latenz<br/>~200ms]
        AO2[Niedrig Wechsel<br/>5-10%]
        AO3[Sehr Hoch<br/>92%]
        AO4[Sehr Gut<br/>8.5/10]
    end
    
    subgraph "Sticky Sessions"
        direction TB
        SS1[Niedrig Latenz<br/>~100ms]
        SS2[Sehr Niedrig<br/>3-5%]
        SS3[Hoch<br/>88%]
        SS4[Gut<br/>7.8/10]
    end
    
    M1 --> SR1
    M1 --> AO1
    M1 --> SS1
    
    M2 --> SR2
    M2 --> AO2
    M2 --> SS2
    
    M3 --> SR3
    M3 --> AO3
    M3 --> SS3
    
    M4 --> SR4
    M4 --> AO4
    M4 --> SS4
```

## Hybrid-Router Konzept

```mermaid
flowchart TB
    Start([User Input]) --> Base[Sticky Session Base]
    
    Base --> CheckOwnership{Agent hat<br/>Ownership?}
    CheckOwnership -->|Ja| RespectOwnership[Respektiere Agent-Entscheidung]
    CheckOwnership -->|Nein| CheckExit{Exit-Signal?}
    
    CheckExit -->|Ja| UseStateful[Nutze Stateful Context]
    CheckExit -->|Nein| StaySticky[Bleibe bei Agent]
    
    RespectOwnership --> Priority{Priority > 8?}
    Priority -->|Ja| ForceAgent[Erzwinge Agent]
    Priority -->|Nein| UseStateful
    
    UseStateful --> DynamicThreshold[Berechne Dynamic Threshold]
    DynamicThreshold --> EvaluateAll[Evaluiere alle Agents]
    EvaluateAll --> SelectBest[Wähle besten Agent]
    
    ForceAgent --> Response([Agent Response])
    StaySticky --> Response
    SelectBest --> Response
    
    style Base fill:#2196f3
    style RespectOwnership fill:#9c27b0
    style UseStateful fill:#4caf50
    style Response fill:#ffc107
```

## Zusammenfassung

Die drei Router-Ansätze bieten unterschiedliche Strategien für konsistente Agent-Workflows:

1. **Stateful Routing**: Flexibel und anpassungsfähig durch dynamische Schwellenwerte
2. **Agent Ownership**: Robust und intelligent durch selbstverwaltete Agents
3. **Sticky Sessions**: Stabil und vorhersagbar durch "klebriges" Verhalten

Die Wahl des richtigen Ansatzes hängt vom konkreten Use-Case ab. In der Praxis zeigt sich oft, dass eine Kombination aller drei Ansätze die besten Ergebnisse liefert.