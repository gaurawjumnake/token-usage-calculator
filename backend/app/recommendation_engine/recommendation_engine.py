from __future__ import annotations
import json
import math
import time
import logging
from datetime import datetime, timezone
from typing import Any

from backend.app.recommendation_engine.input_schema import QuestionnaireInput
from backend.app.recommendation_engine.output_schema_FINAL import (
    RecommendationOutput,
    SDLC_STAGES,
)
from backend.app.recommendation_engine.tools_CREWAI import _get_catalog, _build_capabilities
from backend.utilites.llm_models import llm_azure
from backend.utilites.app_logger import Logger
log = Logger()

llm = llm_azure

def _call_llm(system: str, user: str, label: str, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            log.log_info(f"[{label}] attempt {attempt}")
            response = llm.call(messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ])
            text = (
                response if isinstance(response, str)
                else getattr(response, "content", None)
                or (response.choices[0].message.content if hasattr(response, "choices") else "")
            )
            if text and text.strip():
                return text.strip()
            log.log_warning(f"[{label}] empty response, retrying…")
            time.sleep(2 ** attempt)
        except Exception as e:
            log.log_error(f"[{label}] error: {e}")
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)
    raise ValueError(f"[{label}] all retries exhausted")


def _parse_json(text: str, label: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = "\n".join(l for l in t.splitlines() if not l.startswith("```")).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        raise ValueError(f"[{label}] bad JSON: {e}\n\nRaw:\n{t}") from e

# STEP 1 — ANALYZER - Produces capability pointers for the overall project + each SDLC stage.

_ANALYZER_SYSTEM = """
You are an AI requirements analyst. Analyze the questionnaire and output ONE JSON
object describing what LLM capabilities are needed — for the OVERALL project and
for each of the 8 SDLC stages individually.

CAPABILITY VOCABULARY (use only these exact strings):
  "vision"             must understand images / diagrams
  "tools"              must call external APIs / functions
  "reasoning"          needs multi-step / chain-of-thought reasoning
  "structured_output"  must return JSON or structured data
  "long_context"       input routinely > 64 K tokens
  "very_long_context"  input routinely > 200 K tokens

CONTEXT SIZES → min_context_tokens:
  Short  (<10 K)  → 8000
  Medium (10-50K) → 32000
  Large  (50-200K)→ 96000
  Very large      → 200000

BUDGET FOCUS:
  cost-conscious | balanced | quality-first

OUTPUT FORMAT — return ONLY this JSON, no preamble or explanation:
{
  "app_summary": {
    "type": "<one of 7 app types>",
    "description": "2-3 sentences",
    "complexity": "low|medium|high",
    "budget_focus": "cost-conscious|balanced|quality-first",
    "min_context_tokens": 32000,
    "agentic_level": "Non-Agentic|Semi-Agentic|Fully Agentic",
    "scale": "Small|Medium|Enterprise",
    "latency": "low|medium|high"
  },
  "overall_capabilities": ["tools", "structured_output"],
  "stage_pointers": {
    "Requirements":  {
      "capabilities": ["long_context", "structured_output"],
      "min_context_tokens": 32000,
      "complexity": "low|medium|high",
      "rationale": "one sentence"
    },
    "Architecture":  { "capabilities": [...], "min_context_tokens": ..., "complexity": "...", "rationale": "..." },
    "Development":   { ... },
    "Code Review":   { ... },
    "Testing":       { ... },
    "Documentation": { ... },
    "Deployment":    { ... },
    "Maintenance":   { ... }
  }
}

STAGE DEFAULTS (override only if questionnaire implies otherwise):
  Requirements:  long_context, structured_output
  Architecture:  reasoning, structured_output
  Development:   tools, structured_output
  Code Review:   reasoning, structured_output
  Testing:       reasoning, structured_output, tools
  Documentation: long_context
  Deployment:    tools, structured_output
  Maintenance:   tools, reasoning
"""

def _run_analyzer(answers: dict, timestamp: str) -> dict:
    user = (
        f"QUESTIONNAIRE:\n{json.dumps(answers, indent=2)}\n\n"
        f"TIMESTAMP: {timestamp}\n\n"
        "Return ONLY the JSON object."
    )
    raw = _call_llm(_ANALYZER_SYSTEM, user, "Analyzer")
    return _parse_json(raw, "Analyzer")

# STEP 2 — CATALOG MATCHER - Reads the pointers from Step 1, queries the in-memory catalog for each stage,

# Price ceiling for cost normalisation (per-1K tokens).
# Real-world LLM prices range from ~$0.0001 to ~$0.75/1K.
# Using $1.0 as the ceiling so cost_score is spread meaningfully.
_COST_CEILING_PER_1K = 1.0


def _get_model_family(model_id: str) -> str:
    """Group model IDs into families for variety/diversity scoring."""
    mid = model_id.lower()
    if "gemini" in mid:
        if "pro" in mid:
            return "gemini-pro"
        if "flash" in mid:
            return "gemini-flash"
        return "gemini"
    if "claude" in mid:
        if "opus" in mid:
            return "claude-opus"
        if "sonnet" in mid:
            return "claude-sonnet"
        if "haiku" in mid:
            return "claude-haiku"
        return "claude"
    if "command" in mid:
        return "command"
    if "gpt-4" in mid:
        return "gpt-4"
    if "gpt-3.5" in mid:
        return "gpt-3.5"
    if "llama" in mid:
        return "llama"
    if "qwen" in mid:
        return "qwen"
    if "gemma" in mid:
        return "gemma"
    return mid.split("/")[-1].split(":")[0]


def _score_model(
    model: dict,
    caps_required: list[str],
    min_ctx: int,
    used_models: set | None = None,
    used_families: set | None = None,
    used_providers: set | None = None,
) -> float:
    """
    Score a model 0–100 for a given stage pointer.
    """
    model_caps  = _build_capabilities(model)
    n_required  = max(len(caps_required), 1)

    matched      = sum(1 for c in caps_required if c in model_caps)
    cap_score    = matched / n_required

    ctx          = model.get("context_length", 0)
    if min_ctx <= 0:
        ctx_score = 1.0
    else:
        ctx_score = min(1.0, math.log1p(ctx) / math.log1p(max(min_ctx * 4, ctx)))

    pricing      = model.get("pricing", {})
    inp_1k       = float(pricing.get("prompt",     0)) * 1_000
    out_1k       = float(pricing.get("completion", 0)) * 1_000
    total_1k     = inp_1k + out_1k
    cost_score   = max(0.0, 1.0 - min(1.0, total_1k / 1.0))

    tier         = _PROVIDER_TIER.get(model.get("provider", "").lower(), 3)
    tier_score   = {1: 1.0, 2: 0.75}.get(tier, 0.5)

    raw = (cap_score * 0.40 + ctx_score * 0.25 + cost_score * 0.25 + tier_score * 0.10)
    
    # Diversity penalties: heavily penalise repeating exact models or families
    penalty = 1.0
    model_id = model.get("model_id", "")
    provider = model.get("provider", "").lower()
    family = _get_model_family(model_id)

    if used_models and model_id in used_models:
        penalty *= 0.4
    elif used_families and family in used_families:
        penalty *= 0.6
        
    if used_providers and provider in used_providers:
        # Mild penalty to encourage provider variety across stages
        penalty *= 0.85

    return round(raw * 100 * penalty, 1)

# -- Provider quality tiers (used to filter catalog noise) ---------------------------
# Tier 1: well-known providers with documented regional endpoints
# Tier 2: known providers, good quality, may lack regional guarantees
# Tier 3: everything else (OpenRouter meta-models, unknown, zero-cost beta)

_PROVIDER_TIER: dict[str, int] = {
    # Tier 1 — regional hosting guaranteed
    "anthropic": 1,
    "google":    1,
    "azure":     1,
    "mistral":   1,
    "cohere":    1,

    # Tier 2 — good quality, no strong regional guarantee
    "openai":    2,
    "meta-llama":2,
    "qwen":      2,
    "deepseek":  2,
    "stepfun":   2,
    "nvidia":    2,
    "x-ai":      2,
    "amazon":    2,
    "microsoft": 2,
}

# OpenRouter meta / routing model_ids — never real deployable models
_META_MODEL_IDS = {"auto", "auto-router", "router"}

# Minimum price floor: zero-cost models are beta/internal, exclude from tiers
_MIN_PRICE_FLOOR = 1e-9   # per-1K; anything at or below this is treated as $0


def _is_zero_cost(model: dict) -> bool:
    return (
        model["pricing"]["input_per_1k"]  <= _MIN_PRICE_FLOOR
        and model["pricing"]["output_per_1k"] <= _MIN_PRICE_FLOOR
    )


def _is_meta_model(model: dict) -> bool:
    return model["model_id"].lower() in _META_MODEL_IDS


def _provider_tier(model: dict) -> int:
    """Return provider quality tier (1=best, 3=unknown)."""
    provider = model.get("provider", "").lower()
    return _PROVIDER_TIER.get(provider, 3)


def _pick_tier(
    ranked: list[dict],
    privacy_regional: bool = False,
) -> dict[str, dict]:
    """
    From a scored+ranked list return one model per tier:
      recommended / budget / premium — all distinct model_ids.

    Fixes applied vs previous version:
    - Meta/routing models (e.g. "auto") are excluded entirely.
    - Zero-cost models ($0 input+output) are excluded from all tiers.
    - privacy_regional=True further restricts to Tier-1 providers only.
    - budget  = real cheapest with score > 50 (not just any cheaper model).
    - premium = highest context_length among Tier-1/2 providers.
    - Graceful fallback: if not enough distinct models, reuses closest match
      and logs a warning rather than returning junk.
    """
    # ── Base pool: strip meta-models and zero-cost entries ────────────────────
    pool = [
        m for m in ranked
        if not _is_meta_model(m) and not _is_zero_cost(m)
    ]

    # ── Privacy filter: regional constraint → Tier-1 providers only ───────────
    if privacy_regional:
        regional_pool = [m for m in pool if _provider_tier(m) == 1]
        if len(regional_pool) >= 3:
            pool = regional_pool
        else:
            # Not enough Tier-1 models; fall back to Tier 1+2 and log
            pool = [m for m in pool if _provider_tier(m) <= 2]
            log.log_warning(
                "privacy_regional=True but fewer than 3 Tier-1 models found; "
                "including Tier-2 providers as fallback."
            )

    if not pool:
        log.log_warning("_pick_tier: pool is empty after filtering — returning empty tiers.")
        return {}

    # ── Recommended: highest score in the clean pool ──────────────────────────
    recommended = pool[0]

    # ── Budget: real cheapest with score > 50, different from recommended ──────
       # Budget: cheapest with score > 50, DIFFERENT PROVIDER than recommended
    budget_candidates = [
        m for m in pool
        if m["model_id"] != recommended["model_id"]
        and m["score"] > 50
        and m["provider"] != recommended["provider"]   # ← force different provider
    ]
    # fallback: same provider allowed if no cross-provider option exists
    if not budget_candidates:
        budget_candidates = [
            m for m in pool
            if m["model_id"] != recommended["model_id"] and m["score"] > 50
        ]
    budget_candidates.sort(
        key=lambda m: m["pricing"]["input_per_1k"] + m["pricing"]["output_per_1k"]
    )
    budget = budget_candidates[0] if budget_candidates else (pool[1] if len(pool) > 1 else recommended)

    # Premium: largest context, DIFFERENT PROVIDER than both recommended and budget
    premium_candidates = [
        m for m in pool
        if m["model_id"] not in {recommended["model_id"], budget["model_id"]}
        and _provider_tier(m) <= 2
        and m["provider"] not in {recommended["provider"], budget["provider"]}  # ← force different provider
    ]
    # fallback: relax provider constraint if not enough candidates
    if not premium_candidates:
        premium_candidates = [
            m for m in pool
            if m["model_id"] not in {recommended["model_id"], budget["model_id"]}
            and _provider_tier(m) <= 2
        ]
    premium = (
        max(premium_candidates, key=lambda m: m["context_length"])
        if premium_candidates else pool[-1]
    )

    return {
        "recommended": recommended,
        "budget":      budget,
        "premium":     premium,
    }


def _shortlist_for_pointer(
    pointer: dict,
    per_tier: int = 5,
    privacy_regional: bool = False,
    used_models: set | None = None,
    used_families: set | None = None,
    used_providers: set | None = None,
) -> dict:
    """
    Filter catalog for one stage pointer. Returns:
      {
        "candidates": [ ...top 15 clean models for LLM to reason over... ],
        "tiers":      { recommended, budget, premium }
      }
    """
    catalog = _get_catalog()
    caps    = pointer.get("capabilities", [])
    min_ctx = pointer.get("min_context_tokens", 8_000)

    def _query(caps_req: list[str], ctx: int) -> list[dict]:
        results = []
        for key, raw in catalog.items():
            model = {"model_id": key, **raw}

            # Context filter
            if model.get("context_length", 0) < ctx:
                continue

            # Capability filter
            model_caps = _build_capabilities(model)
            if not all(c in model_caps for c in caps_req):
                continue

            # Privacy regional filter: restrict to Tier 1 & 2 providers
            if privacy_regional and _provider_tier(model) > 2:
                continue

            # Pricing (per-token → per-1K)
            pricing = model.get("pricing", {})
            inp = float(pricing.get("prompt",     0)) * 1_000
            out = float(pricing.get("completion", 0)) * 1_000

            entry = {
                "model_id":       key,
                "name":           model.get("name", key),
                "provider":       model.get("provider", ""),
                "context_length": model.get("context_length", 0),
                "pricing":        {"input_per_1k": round(inp, 6), "output_per_1k": round(out, 6)},
                "capabilities":   sorted(model_caps),
                "score":          _score_model(model, caps_req, ctx, used_models, used_families, used_providers),
            }

            # Strip meta-models and zero-cost entries immediately
            if _is_meta_model(entry) or _is_zero_cost(entry):
                continue

            results.append(entry)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _diversify(results: list[dict], max_per_provider: int = 3) -> list[dict]:
        seen: dict[str, int] = {}
        out = []
        for m in results:
            p = m["provider"]
            if seen.get(p, 0) < max_per_provider:
                out.append(m)
                seen[p] = seen.get(p, 0) + 1
        return out

    # ── Fallback ladder: relax caps/ctx until ≥ 3 real candidates ─────────────
    results: list[dict] = []   # initialise so _diversify always has a value
    for attempt_caps, attempt_ctx in [
        (caps, min_ctx),
        ([c for c in caps if c not in ("long_context", "very_long_context")], min_ctx),
        ([c for c in caps if c in ("tools", "vision", "structured_output")], min_ctx // 2),
        ([], 0),
    ]:
        results = _query(attempt_caps, attempt_ctx)
        if len(results) >= 3:
            break

    # Diversify AFTER the ladder has produced a final result list
    results = _diversify(results, max_per_provider=3)

    candidates = results[: per_tier * 3]        # top 15
    tiers      = _pick_tier(results[:20], privacy_regional=privacy_regional)

    return {"candidates": candidates, "tiers": tiers}


def _run_catalog_matcher(stage_pointers: dict, privacy_regional: bool = False) -> dict:
    result: dict[str, dict] = {}
    used_models: set = set()
    used_families: set = set()
    used_providers: set = set()

    for stage in SDLC_STAGES:
        pointer = stage_pointers.get(stage, {})
        shortlist = _shortlist_for_pointer(
            pointer,
            privacy_regional=privacy_regional,
            used_models=used_models,
            used_families=used_families,
            used_providers=used_providers,
        )
        
        # Track selected models, families, and providers so future stages avoid repeating them
        for tier_model in shortlist.get("tiers", {}).values():
            if isinstance(tier_model, dict) and "model_id" in tier_model:
                mid = tier_model["model_id"]
                used_models.add(mid)
                used_families.add(_get_model_family(mid))
                prov = tier_model.get("provider", "").lower()
                if prov:
                    used_providers.add(prov)

        result[stage] = {
            "pointer": pointer,
            **shortlist,
        }

    all_caps = list({c for p in stage_pointers.values() for c in p.get("capabilities", [])})
    max_ctx  = max((p.get("min_context_tokens", 8_000) for p in stage_pointers.values()), default=32_000)
    result["overall"] = {
        "pointer": {"capabilities": all_caps, "min_context_tokens": max_ctx},
        **_shortlist_for_pointer(
            {"capabilities": all_caps, "min_context_tokens": max_ctx},
            privacy_regional=privacy_regional,
        ),
    }
    return result


# STEP 3 — SYNTHESIZER - Receives the shortlisted candidates per stage and produces the full output.

_SYNTHESIZER_SYSTEM = """
You are an expert LLM architect. Your task is to analyse the user's specific problem
statement and questionnaire answers to recommend the BEST-FIT models — not generic
defaults. Every recommendation must be justified by concrete evidence from the
questionnaire and the candidate model data provided.

You will receive:
  - app_summary      (analyzer output — app type, complexity, agentic level, etc.)
  - stage_catalog    (shortlisted candidate models + pre-scored tier suggestions per stage)
  - questionnaire    (original user answers)

═══════════════════════════════════════════════════════════════
SELECTION RULES (MANDATORY — violating any rule is an error):
═══════════════════════════════════════════════════════════════
1. USE ONLY model_ids that appear in the provided candidate list for each stage.
   Do NOT invent or hallucinate model IDs.

2. Within each stage, the three tier picks MUST be DIFFERENT model_ids AND from
   DIFFERENT providers where at least 3 providers are available.

3. Tier logic driven by questionnaire context:
   • recommended — highest capability-to-cost ratio for THIS specific use case.
     If the user is cost-conscious, prefer cheaper models; if quality-first, prefer
     high-reasoning models regardless of cost.
   • budget      — CHEAPEST model that still meets the stage's required capabilities
     (input_per_1k + output_per_1k should be lower than recommended).
   • premium     — BEST quality for the stage's core task (largest context if the
     stage is context-heavy; strongest reasoning if reasoning-heavy).

4. STAGE-SPECIFIC DIFFERENTIATION — each stage has different priorities:
   • Requirements, Documentation: prioritise long_context models.
   • Architecture, Code Review:   prioritise reasoning models.
   • Development, Testing:        prioritise tools + structured_output models.
   • Deployment, Maintenance:     prioritise tools + reliability (Tier-1 providers).
   Do NOT pick the same model for all 8 stages; vary selections based on stage role.

5. QUESTIONNAIRE-DRIVEN OVERRIDES:
   • If agentic_level = "Fully Agentic": prioritise tool-calling models across all stages.
   • If priority = "Cost-conscious" or budget < $1,000: choose cheapest that qualify.
   • If priority = "Quality-first" or budget > $10,000: choose highest-reasoning models.
   • If privacy includes "Data must stay in our region": restrict to Tier-1 providers
     (Anthropic, Google, Azure, Mistral, Cohere) only.
   • If context_size = "Large" or "Very Large": prefer models with context > 128K.
   • If scale = "Enterprise": prefer Tier-1 providers with SLA guarantees.

6. The pre-computed tier suggestions in stage_catalog.tiers are starting points.
   Override them if questionnaire context clearly warrants a different pick, and
   document why in the corresponding _why field.

7. The "why" fields must reference SPECIFIC factors: model name, price, capability,
   context window size, or provider. Generic text like "best for this stage" is
   NOT acceptable.

═══════════════════════════════════════════════════════════════
OUTPUT SCHEMA — return ONLY valid JSON, no preamble or markdown:
═══════════════════════════════════════════════════════════════
{
  "schema_version": "2.0",
  "generated_at": "<filled by system>",
  "input_hash": "<filled by system>",

  "questionnaire_summary": {
    "app_type": "...",
    "agentic_level": "Non-Agentic|Semi-Agentic|Fully Agentic",
    "scale": "Small|Medium|Enterprise",
    "priority": "...",
    "budget_range": "..."
  },

  "workload_profile": {
    "project_duration_months": 12,
    "active_users": 500,
    "requests_per_user_per_day": 10,
    "avg_input_tokens": 3000,
    "avg_output_tokens": 500,
    "avg_reasoning_tokens": 1000,
    "avg_cached_tokens": 0,
    "min_context_window": 32000,
    "recommended_context_window": 64000,
    "complexity": "low|medium|high",
    "latency_requirement": "low|medium|high",
    "batch_eligible": false,
    "cache_eligible": false
  },

  "single_model_recommendations": [
    {"category": "recommended", "model_id": "...", "why": "specific reason from questionnaire + model data", "tradeoffs": "..."},
    {"category": "budget",      "model_id": "...", "why": "specific reason", "tradeoffs": "..."},
    {"category": "premium",     "model_id": "...", "why": "specific reason", "tradeoffs": "..."}
  ],

  "stage_recommendations": [
    {
      "stage_name": "Requirements",
      "models": {
        "recommended_model_id": "...",
        "budget_model_id":      "...",
        "premium_model_id":     "...",
        "recommended_why": "model X chosen because [specific cap/price/context reason]",
        "budget_why":      "model Y is cheapest at $Z/1K while meeting long_context requirement",
        "premium_why":     "model Z has 1M-token context ideal for large requirements docs",
        "key_capability":  "long_context",
        "tradeoffs":       "..."
      },
      "rationale": "1-2 sentences specific to this stage and this questionnaire"
    },
    { "stage_name": "Architecture",  "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "reasoning", "tradeoffs": "..." }, "rationale": "..." },
    { "stage_name": "Development",   "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "tools", "tradeoffs": "..." }, "rationale": "..." },
    { "stage_name": "Code Review",   "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "reasoning", "tradeoffs": "..." }, "rationale": "..." },
    { "stage_name": "Testing",       "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "structured_output", "tradeoffs": "..." }, "rationale": "..." },
    { "stage_name": "Documentation", "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "long_context", "tradeoffs": "..." }, "rationale": "..." },
    { "stage_name": "Deployment",    "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "tools", "tradeoffs": "..." }, "rationale": "..." },
    { "stage_name": "Maintenance",   "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "tools", "tradeoffs": "..." }, "rationale": "..." }
  ],

  "architecture": {
    "pattern": "Single Model|RAG|Agentic|Multi-Agent",
    "hosting_strategy": "Managed API|Cloud Marketplace|Self-Hosted",
    "agent_framework_recommendation": null,
    "framework_constraints": [],
    "roles": [
      {
        "role": "...",
        "recommended_model_id": "...",
        "budget_model_id": "...",
        "premium_model_id": "...",
        "reason": "..."
      }
    ],
    "notes": []
  },

  "optimisation_tips": [
    {"impact": "high|medium|low", "title": "...", "detail": "specific actionable tip referencing model prices or capabilities"}
  ],

  "confidence": {
    "score": "high|medium|low",
    "reason": "specific reason e.g. 'strong match between stated requirements and catalog'",
    "assumptions": ["...", "..."]
  }
}

WORKLOAD ESTIMATION GUIDE (use to fill workload_profile):
  avg_input_tokens:       Short(<10K)→500, Medium(10-50K)→3000, Large(50-200K)→25000, Very Large→120000
  avg_output_tokens:      Chatbot→300, Docs→1200, Code→1000, Data→500, Content→1500
  avg_reasoning_tokens:   low complexity→0, medium→1000, high→5000
  complexity / latency_requirement must be exactly "low", "medium", or "high"
  batch_eligible: true if latency_requirement is high (async OK)
  cache_eligible: true if inputs are repetitive (RAG system prompts, etc.)
"""

def _run_synthesizer(
    app_summary: dict,
    stage_catalog: dict,
    questionnaire: dict,
    input_hash: str,
    timestamp: str,
) -> dict:
    # Build a compact catalog payload — tiers + top-5 candidates per stage
    compact_catalog: dict[str, Any] = {}
    for stage, data in stage_catalog.items():
        compact_catalog[stage] = {
            "pointer":    data["pointer"],
            "tiers":      data["tiers"],          # pre-computed R/B/P suggestions
            "candidates": data["candidates"][:5], # top 5 for LLM to reason over
        }

    user = (
        f"APP SUMMARY (from analyzer):\n{json.dumps(app_summary, indent=2)}\n\n"
        f"STAGE CATALOG (shortlisted models per stage):\n{json.dumps(compact_catalog, indent=2)}\n\n"
        f"ORIGINAL QUESTIONNAIRE:\n{json.dumps(questionnaire, indent=2)}\n\n"
        f"generated_at: {timestamp}\n"
        f"input_hash: {input_hash}\n\n"
        "Return ONLY the final JSON object."
    )
    raw = _call_llm(_SYNTHESIZER_SYSTEM, user, "Synthesizer")
    return _parse_json(raw, "Synthesizer")


# OUTPUT NORMALIZATION

_STAGE_ORDER  = {s: i for i, s in enumerate(SDLC_STAGES)}
_LATENCY_MAP  = {
    "fast": "low", "very fast": "low", "real-time": "low", "low latency": "low",
    "moderate": "medium", "normal": "medium", "standard": "medium",
    "slow": "high", "batch": "high", "relaxed": "high",
}
_COMPLEXITY_MAP = {
    "simple": "low", "easy": "low", "basic": "low",
    "moderate": "medium", "average": "medium",
    "complex": "high", "difficult": "high", "hard": "high",
}


def _normalize_stage_recommendations(raw: Any) -> list[dict]:
    if not isinstance(raw, list):
        raw = []
    by_stage = {
        item["stage_name"]: item
        for item in raw
        if isinstance(item, dict) and item.get("stage_name") in _STAGE_ORDER
    }
    result = []
    for stage in SDLC_STAGES:
        entry = by_stage.get(stage, {"stage_name": stage, "models": {}, "rationale": ""})
        models = entry.get("models") or {}
        for k in ("recommended_model_id", "budget_model_id", "premium_model_id"):
            models.setdefault(k, "unknown")
        for k in ("recommended_why", "budget_why", "premium_why", "key_capability", "tradeoffs"):
            models.setdefault(k, "")
        entry["models"]     = models
        entry["stage_name"] = stage
        entry.setdefault("rationale", "")
        result.append(entry)
    return result


def _normalize(data: dict, input_hash: str, original_answers: dict) -> dict:
    data["input_hash"] = input_hash

    qs = data.setdefault("questionnaire_summary", {})
    qs.setdefault("app_type",      original_answers.get("app_type", "Unknown"))
    qs.setdefault("agentic_level", original_answers.get("agentic_level", "Non-Agentic"))
    qs.setdefault("scale",         original_answers.get("scale", "Unknown"))
    qs.setdefault("priority",      original_answers.get("priority", "Balanced"))
    qs.setdefault("budget_range",  original_answers.get("budget", "Unknown"))

    wp = data.setdefault("workload_profile", {})
    lat = str(wp.get("latency_requirement", "")).strip().lower()
    if lat not in ("low", "medium", "high"):
        wp["latency_requirement"] = _LATENCY_MAP.get(lat, "medium")
    cmp = str(wp.get("complexity", "")).strip().lower()
    if cmp not in ("low", "medium", "high"):
        wp["complexity"] = _COMPLEXITY_MAP.get(cmp, "medium")

    smr = data.get("single_model_recommendations")
    if isinstance(smr, dict):
        data["single_model_recommendations"] = [
            {**(v if isinstance(v, dict) else {"model_id": v, "why": "", "tradeoffs": ""}),
             "category": cat}
            for cat, v in smr.items() if cat in ("recommended", "budget", "premium")
        ]

    arch = data.setdefault("architecture", {})
    arch.setdefault("hosting_strategy",      "Managed API")
    arch.setdefault("pattern",               "Single Model")
    arch.setdefault("framework_constraints", [])
    arch.setdefault("roles",                 [])
    arch.setdefault("notes",                 [])

    conf = data.setdefault("confidence", {})
    if "rationale" in conf and "reason" not in conf:
        conf["reason"] = conf.pop("rationale")
    conf.setdefault("reason", "")
    if not isinstance(conf.get("assumptions"), list):
        conf["assumptions"] = []
    if conf.get("score") not in ("high", "medium", "low"):
        conf["score"] = "medium"

    tips = data.get("optimisation_tips", [])
    data["optimisation_tips"] = [
        {**t, "impact": t.get("impact", "medium")}
        for t in (tips if isinstance(tips, list) else [])
        if isinstance(t, dict) and t.get("title") and t.get("detail")
    ]

    data["stage_recommendations"] = _normalize_stage_recommendations(
        data.get("stage_recommendations", [])
    )
    return data



# ENGINE


class RecommendationEngine:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)

    def run(self, questionnaire: QuestionnaireInput) -> RecommendationOutput:
        input_hash = questionnaire.stable_hash()
        timestamp  = datetime.now(timezone.utc).isoformat()
        answers    = questionnaire.answers
        t0         = time.time()

        privacy_regional = any(
            "region" in str(p).lower()
            for p in answers.get("privacy", [])
        )

        # -- Step 1: Analyzer (1 LLM call) ------------------------------------
        log.log_info("Step 1/3 — Analyzer")
        analysis       = _run_analyzer(answers, timestamp)
        app_summary    = analysis.get("app_summary", {})
        stage_pointers = analysis.get("stage_pointers", {})
        log.log_info(f"  done ({time.time()-t0:.1f}s)")

        # -- Step 2: Catalog matcher (pure Python) ------------------------------
        log.log_info("Step 2/3 — Catalog Matcher")
        stage_catalog = _run_catalog_matcher(stage_pointers, privacy_regional=privacy_regional)
        log.log_info(f"  done ({time.time()-t0:.1f}s)")

        # -- Step 3: Synthesizer (1 LLM call) ─────────────────────────────────
        log.log_info("Step 3/3 — Synthesizer")
        final = _run_synthesizer(app_summary, stage_catalog, answers, input_hash, timestamp)
        log.log_info(f"  done ({time.time()-t0:.1f}s total)")
        if final:
            log.log_info("Recommendation generated successfully")
            return _normalize(final, input_hash, answers)  # type: ignore
        else:
            log.log_warning("Failed to generate recommendation")
            return None #type:ignore
        


# if __name__ == "__main__":
#     from backend.app.recommendation_engine.input_schema import QuestionnaireInput
#     logging.basicConfig(level=logging.INFO)

#     sample = QuestionnaireInput(answers={
#         "app_type":          "Enterprise Knowledge Assistant",
#         "app_description":   "use openai and anthropic models only",
#         "context_size":      "Large",
#         "latency":           "Fast",
#         "scale":             "Enterprise",
#         "agentic_level":     "Semi-Agentic",
#         "agent_framework":   "LangChain / LangGraph",
#         "agent_structure":   "Single agent",
#         "capabilities":      ["Function / tool calling", "Structured output / JSON", "Long context / large docs"],
#         "uses_coding_agent": "No",
#         "coding_tool":       [],
#         "priority":          "Balanced",
#         "budget":            "$1,000–$10,000",
#         "privacy":           ["Data must stay in our region"],
#     })

#     print(sample)
#     output = RecommendationEngine(verbose=True).run(sample)
#     print(json.dumps(output, indent=2, ensure_ascii=False))