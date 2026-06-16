# Recommendation Engine v2.0 – Agentic Architecture Specification

## Overview

This specification describes the migration of the current single-LLM recommendation engine to a CrewAI-based multi-agent architecture capable of:
- Better model recommendations
- Stage-wise SDLC model selection
- Stage-wise token estimation
- Tool-assisted model evaluation
- Schema-compliant output generation

## Current Challenges

### Single LLM Decision Point
Questionnaire → Prompt → GPT-4o → Recommendation Output

Problems:
- Same models repeatedly recommended
- Weak differentiation between use cases
- No capability matching
- No model catalog intelligence
- No reasoning traceability
- Context window constraints frequently ignored

### Candidate Models Are Insufficient
Current implementation provides only model_id values.

Missing:
- Context length
- Pricing
- Tool support
- Vision support
- Reasoning support
- Modalities
- Provider constraints
- Completion limits

### Workload Profile Too Generic
Current output assumes one generic workload pattern.

Required SDLC phases:
1. Requirements
2. Architecture
3. Development
4. Code Review
5. Testing
6. Documentation
7. Deployment
8. Maintenance

## Proposed Architecture

Questionnaire
→ Analyzer Agent
→ Project Intelligence
→ Synthesizer Agent
→ Model Selection Draft
→ Summarizer Agent
→ Schema-Compliant Output

## Agent Design

### Analyzer Agent
Responsibilities:
- Understand questionnaire intent
- Classify application
- Determine capability requirements
- Create SDLC phase analysis
- Estimate workload complexity

Outputs:
- Application classification
- Capability requirements
- SDLC phase breakdown
- Complexity profile

### Synthesizer Agent
Responsibilities:
- Query model catalog
- Filter incompatible models
- Rank candidates
- Recommend models by phase

Outputs:
- Recommended model
- Budget model
- Premium model
- Stage-wise model allocations

### Summarizer Agent
Responsibilities:
- Produce schema-compliant output
- Ensure validation safety
- Assemble final response

## Tool Architecture

### ModelCatalogTool
Reads model catalog and exposes:
- model_id
- provider
- context_length
- pricing
- modalities
- reasoning support
- tool support
- vision support

### CapabilityFilterTool
Filters models by:
- Tool calling
- Reasoning support
- Vision support
- File support
- Structured output
- Context window

### FrameworkCompatibilityTool
Examples:
- Claude MCP → Anthropic only
- OpenAI Assistants → OpenAI only
- CrewAI → OpenAI-compatible APIs
- LangGraph → Provider agnostic

### ContextWindowValidatorTool
Ensures:
Required Context <= Supported Context

### ModelRankingTool
Suggested scoring:
- Capability Match = 35%
- Reasoning Quality = 20%
- Context Window = 15%
- Tool Support = 10%
- Cost Efficiency = 10%
- Latency = 10%

## Stage-Wise Model Recommendations

Each SDLC phase receives dedicated recommendations:

- Requirements
- Architecture
- Development
- Code Review
- Testing
- Documentation
- Deployment
- Maintenance

## Stage-Wise Token Estimation

Estimate separately for every SDLC phase:
- avg_input_tokens
- avg_output_tokens
- avg_reasoning_tokens
- avg_cached_tokens

## Costing Strategy

Future Pricing Engine:

Total Project Cost =
Requirements +
Architecture +
Development +
Code Review +
Testing +
Documentation +
Deployment +
Maintenance

## Model Catalog Strategy

Do not send all 350 models to the LLM.

Recommended flow:

350 Models
→ Capability Filter
→ Ranked Subset
→ Synthesizer Agent

Benefits:
- Lower token consumption
- Better recommendations
- Faster execution
- More deterministic output

## Schema Strategy

Keep existing APIs:
- RecommendationOutput
- RecommendationOutput_API

Add optional fields:
- stage_recommendations
- stage_workload_profiles

## Implementation Roadmap

### Phase 1
Build:
- ModelCatalogTool
- CapabilityFilterTool
- ContextWindowValidatorTool
- ModelRankingTool

### Phase 2
Introduce CrewAI:
- Analyzer Agent
- Synthesizer Agent
- Summarizer Agent

### Phase 3
Implement:
- Stage-wise token estimation
- Stage-wise model recommendations

### Phase 4
Upgrade pricing engine:
Project Cost = Sum(Stage Costs)

## Expected Outcomes

- Higher recommendation accuracy
- Better model differentiation
- SDLC-aware recommendations
- Realistic token projections
- Improved pricing accuracy
- Data-driven model selection
