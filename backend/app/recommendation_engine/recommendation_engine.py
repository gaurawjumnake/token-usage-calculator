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

# STEP 1 — ANALYZER - Produces capability pointers for the overall project + each SDLC stage,
# AND extracts all structured constraints/overrides from the free-text app_description.

_ANALYZER_SYSTEM = """
You are an AI requirements analyst. Your job has TWO parts:

PART A — EXTRACT EVERYTHING FROM app_description (do this FIRST)
  app_description is a free-text field. The user may have written anything there:
  specific model names, usage metrics, team structures, token volumes, domain context,
  or constraints. You must read it carefully and extract ALL structured information
  into the "description_extracts" section of your output.

  Extract the following if present — leave fields null/empty if not mentioned:

  1. MODEL / PROVIDER CONSTRAINTS
     Look for any statement about which models or providers to use.
     This can be phrased many ways:
       "use gpt 4o and claude sonnet only"
       "only gemini models"
       "we want anthropic and openai"
       "restrict to azure-hosted models"
       "use claude-sonnet-4-6, gpt-4o and gemini-1.5-flash"
     If found → set has_model_constraint: true, list providers and model hints.
     If NOT found → has_model_constraint: false, empty lists.

  2. TOKEN VOLUME OVERRIDES
     Look for specific numbers about token usage per call/workflow/request.
     Examples:
       "average input tokens per workflow: 120,000"
       "each request sends ~50K tokens"
       "output is always under 2K tokens"
     If found → set has_token_override: true and fill avg_input_tokens / avg_output_tokens.
     These OVERRIDE the default token estimate table below.

  3. USAGE SCALE OVERRIDES
     Look for numbers about users, teams, requests, executions per day/month.
     Examples:
       "10 teams of 40 people each"
       "1,000 workflow executions per day"
       "22 working days per month → 22,000 executions/month"
     If found → set has_scale_override: true and fill the fields.

  4. DOMAIN / CAPABILITY SIGNALS
     Keywords that imply LLM capabilities:
       "images / diagrams / screenshots" → vision
       "calls external APIs / functions" → tools
       "chain-of-thought / reasoning"   → reasoning
       "JSON output / structured data"  → structured_output
       "large documents / PDFs / books" → long_context
       "very large corpora (>200K tks)" → very_long_context

PART B — ANALYZE CAPABILITIES AND TOKEN ESTIMATES
  Using BOTH the structured questionnaire answers AND the extracts from Part A,
  produce capability pointers for all 8 SDLC stages and token estimates.
  If description_extracts.has_token_override is true, use those token numbers as
  the base for all stages (scale per stage as appropriate) instead of the defaults.
  If description_extracts.has_scale_override is true, reflect scale in app_summary.

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

TOKEN ESTIMATE DEFAULTS (use only if description has no token override):
  Requirements:  input 4000–20000  output 500–2000
  Architecture:  input 3000–10000  output 800–3000
  Development:   input 2000–8000   output 500–2000
  Code Review:   input 3000–12000  output 500–1500
  Testing:       input 2000–6000   output 500–2000
  Documentation: input 5000–30000  output 1000–5000
  Deployment:    input 1000–4000   output 300–1000
  Maintenance:   input 2000–8000   output 300–1000
  Scale up if context_size = Large or Very Large.

OUTPUT FORMAT — return ONLY this JSON, no preamble or explanation:
{
  "description_extracts": {
    "has_model_constraint": false,
    "preferred_providers": [],
    "preferred_model_hints": [],
    "constraint_source": "exact quote from app_description that stated the constraint, or empty string",
    "has_token_override": false,
    "avg_input_tokens_override": null,
    "avg_output_tokens_override": null,
    "token_override_source": "exact quote, or empty string",
    "has_scale_override": false,
    "daily_executions": null,
    "monthly_executions": null,
    "active_users": null,
    "scale_override_source": "exact quote, or empty string",
    "domain_capability_signals": []
  },
  "app_summary": {
    "type": "<one of 7 app types>",
    "description": "2-3 sentences referencing app_description if present",
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
    "Development":   { "capabilities": [...], "min_context_tokens": ..., "complexity": "...", "rationale": "..." },
    "Code Review":   { "capabilities": [...], "min_context_tokens": ..., "complexity": "...", "rationale": "..." },
    "Testing":       { "capabilities": [...], "min_context_tokens": ..., "complexity": "...", "rationale": "..." },
    "Documentation": { "capabilities": [...], "min_context_tokens": ..., "complexity": "...", "rationale": "..." },
    "Deployment":    { "capabilities": [...], "min_context_tokens": ..., "complexity": "...", "rationale": "..." },
    "Maintenance":   { "capabilities": [...], "min_context_tokens": ..., "complexity": "...", "rationale": "..." }
  },
  "stage_token_estimates": {
    "Requirements":  { "avg_input_tokens": 8000,  "avg_output_tokens": 1200 },
    "Architecture":  { "avg_input_tokens": 5000,  "avg_output_tokens": 1500 },
    "Development":   { "avg_input_tokens": 4000,  "avg_output_tokens": 1000 },
    "Code Review":   { "avg_input_tokens": 6000,  "avg_output_tokens": 800  },
    "Testing":       { "avg_input_tokens": 3500,  "avg_output_tokens": 900  },
    "Documentation": { "avg_input_tokens": 12000, "avg_output_tokens": 2500 },
    "Deployment":    { "avg_input_tokens": 2000,  "avg_output_tokens": 500  },
    "Maintenance":   { "avg_input_tokens": 4000,  "avg_output_tokens": 600  }
  }
}

STAGE DEFAULTS (override only if questionnaire or app_description implies otherwise):
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
    app_desc = answers.get("app_description", "")
    desc_block = f"\nAPP DESCRIPTION:\n{app_desc}\n" if app_desc else ""
    user = (
        f"QUESTIONNAIRE:\n{json.dumps(answers, indent=2)}\n"
        f"{desc_block}"
        f"TIMESTAMP: {timestamp}\n\n"
        "Complete PART A first (extract everything from app_description into description_extracts),\n"
        "then complete PART B (capabilities and token estimates).\n"
        "Return ONLY the JSON object."
    )
    raw = _call_llm(_ANALYZER_SYSTEM, user, "Analyzer")
    result = _parse_json(raw, "Analyzer")

    de = result.get("description_extracts", {})
    if de.get("has_model_constraint"):
        log.log_info(
            f"Analyzer found model constraint: "
            f"providers={de.get('preferred_providers')} "
            f"hints={de.get('preferred_model_hints')} "
            f"source='{de.get('constraint_source')}'"
        )
    if de.get("has_token_override"):
        log.log_info(
            f"Analyzer found token override: "
            f"input={de.get('avg_input_tokens_override')} "
            f"output={de.get('avg_output_tokens_override')} "
            f"source='{de.get('token_override_source')}'"
        )
    if de.get("has_scale_override"):
        log.log_info(
            f"Analyzer found scale override: "
            f"daily={de.get('daily_executions')} "
            f"monthly={de.get('monthly_executions')} "
            f"users={de.get('active_users')}"
        )
    return result


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
    preferred_providers: list[str] | None = None,
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

    # Hard-filter to preferred providers so the synthesizer LLM only sees those models.
    # Fallback ladder: if filtering leaves < 3 candidates, progressively widen.
    if preferred_providers:
        preferred_set = {p.lower() for p in preferred_providers}
        filtered = [m for m in results if m["provider"].lower() in preferred_set]
        if len(filtered) >= 3:
            results = filtered
        else:
            # Not enough preferred-provider models — include all but put preferred first
            preferred_first = [m for m in results if m["provider"].lower() in preferred_set]
            others          = [m for m in results if m["provider"].lower() not in preferred_set]
            results = preferred_first + others
            if preferred_first:
                log.log_warning(
                    f"preferred_providers filter left only {len(preferred_first)} model(s); "
                    "including all providers as fallback but preferred models are ranked first."
                )
            else:
                log.log_warning(
                    "preferred_providers filter matched 0 models for this stage; "
                    "falling back to full catalog."
                )

    candidates = results[: per_tier * 3]        # top 15
    tiers      = _pick_tier(results[:20], privacy_regional=privacy_regional)

    return {"candidates": candidates, "tiers": tiers}


def _run_catalog_matcher(
    stage_pointers: dict,
    privacy_regional: bool = False,
    preferred_providers: list[str] | None = None,
) -> dict:
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
            preferred_providers=preferred_providers,
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
            preferred_providers=preferred_providers,
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
  - app_summary           (analyzer output — app type, complexity, agentic level, etc.)
  - stage_token_estimates (per-stage avg_input/output_tokens from the analyzer)
  - stage_catalog         (shortlisted candidate models + pre-scored tier suggestions per stage)
  - questionnaire         (original user answers including app_description free-text)

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

8. PROVIDER PREFERENCE (injected at runtime — check PROVIDER CONSTRAINT block below):
   If the user specified preferred providers, you MUST pick ONLY model_ids whose
   provider matches the list. This overrides rules 2–6 for provider diversity.
   If a stage's candidate list contains no model from a preferred provider, use the
   highest-scored available model and note the fallback in the corresponding _why field.
   Never invent a model_id to satisfy this rule — only use IDs present in candidates.

9. APP DESCRIPTION MODEL CONSTRAINTS (HIGHEST PRIORITY — overrides all other rules):
   Before selecting any model, READ app_description in full. If it explicitly names
   specific models or providers (e.g. "use gpt 4o and claude sonnet 4.6 only",
   "use only gemini flash", "restrict to anthropic and openai models"), this is a
   HARD CONSTRAINT that takes precedence over rules 1–8:
   • Every recommended_model_id, budget_model_id, and premium_model_id across ALL
     8 stages MUST come from the named models or providers only.
   • If multiple models from the same constraint set are available for a stage, vary
     within that set (e.g. different claude tiers: haiku/sonnet/opus).
   • If only one provider is available in the candidate list for a stage, use 3
     different models from that provider if available; otherwise repeat the best one
     and note the limitation in the _why field.
   • Always quote the exact phrase from app_description that defines the constraint
     in at least one _why field per stage so the user knows it was honoured.
   • NEVER pick a model outside the constraint set even if it scores higher.

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
      "workload_profile": {
        "avg_input_tokens":     8000,
        "avg_output_tokens":    1200,
        "avg_reasoning_tokens": 0
      },
      "rationale": "1-2 sentences specific to this stage and this questionnaire"
    },
    { "stage_name": "Architecture",  "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "reasoning", "tradeoffs": "..." }, "workload_profile": { "avg_input_tokens": 5000, "avg_output_tokens": 1500, "avg_reasoning_tokens": 1000 }, "rationale": "..." },
    { "stage_name": "Development",   "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "tools", "tradeoffs": "..." }, "workload_profile": { "avg_input_tokens": 4000, "avg_output_tokens": 1000, "avg_reasoning_tokens": 500 }, "rationale": "..." },
    { "stage_name": "Code Review",   "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "reasoning", "tradeoffs": "..." }, "workload_profile": { "avg_input_tokens": 6000, "avg_output_tokens": 800, "avg_reasoning_tokens": 1000 }, "rationale": "..." },
    { "stage_name": "Testing",       "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "structured_output", "tradeoffs": "..." }, "workload_profile": { "avg_input_tokens": 3500, "avg_output_tokens": 900, "avg_reasoning_tokens": 500 }, "rationale": "..." },
    { "stage_name": "Documentation", "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "long_context", "tradeoffs": "..." }, "workload_profile": { "avg_input_tokens": 12000, "avg_output_tokens": 2500, "avg_reasoning_tokens": 0 }, "rationale": "..." },
    { "stage_name": "Deployment",    "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "tools", "tradeoffs": "..." }, "workload_profile": { "avg_input_tokens": 2000, "avg_output_tokens": 500, "avg_reasoning_tokens": 0 }, "rationale": "..." },
    { "stage_name": "Maintenance",   "models": { "recommended_model_id": "...", "budget_model_id": "...", "premium_model_id": "...", "recommended_why": "...", "budget_why": "...", "premium_why": "...", "key_capability": "tools", "tradeoffs": "..." }, "workload_profile": { "avg_input_tokens": 4000, "avg_output_tokens": 600, "avg_reasoning_tokens": 500 }, "rationale": "..." }
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

STAGE TOKEN ESTIMATION RULES:
  Use the stage_token_estimates provided by the analyzer as your primary source.
  Override only if the questionnaire or app_description clearly implies different volumes.
  Each stage's workload_profile MUST contain:
    "avg_input_tokens"     — tokens consumed per LLM call at this stage
    "avg_output_tokens"    — tokens generated per LLM call at this stage
    "avg_reasoning_tokens" — additional chain-of-thought tokens (0 for non-reasoning stages)
  These values feed directly into cost projection calculations — be realistic, not minimal.

APP DESCRIPTION USAGE:
  Read app_description carefully. Extract domain, data volumes, and interaction patterns.
  Use this to justify any overrides of the analyzer's token estimates or capability choices.
  Reference specific phrases from app_description in your rationale and why fields.
"""

def _run_synthesizer(
    app_summary: dict,
    stage_catalog: dict,
    stage_token_estimates: dict,
    questionnaire: dict,
    input_hash: str,
    timestamp: str,
    preferred_providers: list[str] | None = None,
    description_extracts: dict | None = None,
) -> dict:
    de = description_extracts or {}

    # Build a compact catalog payload — tiers + top-5 candidates per stage
    compact_catalog: dict[str, Any] = {}
    for stage, data in stage_catalog.items():
        compact_catalog[stage] = {
            "pointer":    data["pointer"],
            "tiers":      data["tiers"],
            "candidates": data["candidates"][:5],
        }

    app_desc = questionnaire.get("app_description", "")
    desc_block = f"\nAPP DESCRIPTION:\n{app_desc}\n" if app_desc else ""

    # Provider constraint block — shown whenever the catalog was filtered by provider
    if preferred_providers:
        provider_block = (
            f"\nPROVIDER CONSTRAINT (MANDATORY — see Rules #8 and #9):\n"
            f"Model selection is restricted to these providers ONLY: "
            f"{json.dumps(preferred_providers)}\n"
            f"Every recommended_model_id, budget_model_id, and premium_model_id across all "
            f"8 stages MUST come from one of these providers. "
            f"If a stage candidate list has no match, use the best available model and "
            f"state 'no preferred-provider model available' in the _why field.\n"
        )
    else:
        provider_block = ""

    # Description extracts block — surfaces everything the Analyzer understood from
    # app_description as explicit, labelled constraints so the Synthesizer cannot miss them.
    extracts_parts: list[str] = []

    if de.get("has_model_constraint"):
        extracts_parts.append(
            f"  MODEL CONSTRAINT (HARD — highest priority, overrides all other rules):\n"
            f"    Source phrase : \"{de.get('constraint_source', '')}\"\n"
            f"    Providers     : {json.dumps(de.get('preferred_providers', []))}\n"
            f"    Model hints   : {json.dumps(de.get('preferred_model_hints', []))}\n"
            f"    → Every stage, every tier MUST use only these models/providers.\n"
            f"    → Quote the source phrase in at least one _why field per stage."
        )

    if de.get("has_token_override"):
        extracts_parts.append(
            f"  TOKEN VOLUMES (from app_description — use these, not defaults):\n"
            f"    avg_input_tokens  : {de.get('avg_input_tokens_override')}\n"
            f"    avg_output_tokens : {de.get('avg_output_tokens_override')}\n"
            f"    Source phrase     : \"{de.get('token_override_source', '')}\"\n"
            f"    → stage_token_estimates have already been updated with these values.\n"
            f"    → Use them as-is for all workload_profile fields."
        )

    if de.get("has_scale_override"):
        extracts_parts.append(
            f"  USAGE SCALE (from app_description):\n"
            f"    active_users        : {de.get('active_users')}\n"
            f"    daily_executions    : {de.get('daily_executions')}\n"
            f"    monthly_executions  : {de.get('monthly_executions')}\n"
            f"    Source phrase       : \"{de.get('scale_override_source', '')}\"\n"
            f"    → Use these numbers to populate workload_profile in the output."
        )

    extracts_block = (
        "\nDESCRIPTION EXTRACTS (structured facts the Analyzer read from app_description):\n"
        + "\n".join(extracts_parts)
        + "\n"
    ) if extracts_parts else ""

    user = (
        f"APP SUMMARY (from analyzer):\n{json.dumps(app_summary, indent=2)}\n\n"
        f"STAGE TOKEN ESTIMATES (from analyzer):\n{json.dumps(stage_token_estimates, indent=2)}\n\n"
        f"STAGE CATALOG (shortlisted models per stage):\n{json.dumps(compact_catalog, indent=2)}\n\n"
        f"ORIGINAL QUESTIONNAIRE:\n{json.dumps(questionnaire, indent=2)}\n"
        f"{desc_block}"
        f"{provider_block}"
        f"{extracts_block}"
        f"generated_at: {timestamp}\n"
        f"input_hash: {input_hash}\n\n"
        "Use stage_token_estimates to populate each stage's workload_profile.\n"
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
        # Preserve per-stage token estimates; default to empty dict if LLM omitted them
        wp = entry.get("workload_profile")
        if not isinstance(wp, dict):
            wp = {}
        wp.setdefault("avg_input_tokens", 0)
        wp.setdefault("avg_output_tokens", 0)
        wp.setdefault("avg_reasoning_tokens", 0)
        entry["workload_profile"] = wp
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

    # Overwrite top-level token fields with the sum across all 8 stages.
    # The LLM generates per-request averages for the overall profile which are
    # typically understated; the ground truth is the aggregate of stage-level values.
    stages = data["stage_recommendations"]
    total_input     = sum(s.get("workload_profile", {}).get("avg_input_tokens",     0) for s in stages)
    total_output    = sum(s.get("workload_profile", {}).get("avg_output_tokens",    0) for s in stages)
    total_reasoning = sum(s.get("workload_profile", {}).get("avg_reasoning_tokens", 0) for s in stages)
    if total_input > 0:
        wp["avg_input_tokens"]     = total_input
    if total_output > 0:
        wp["avg_output_tokens"]    = total_output
    if total_reasoning >= 0:
        wp["avg_reasoning_tokens"] = total_reasoning

    return data



# REASONING REPORT


def _fmt(v: Any, decimals: int = 6) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    return str(v)


def _build_reasoning_md(
    answers: dict,
    analysis: dict,
    stage_catalog: dict,
    output: dict,
) -> str:
    lines: list[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
    lines += [
        f"# LLM Recommendation Report",
        f"Generated: {output.get('generated_at', '')}  |  Hash: `{output.get('input_hash', '')}`",
        "",
    ]

    # ── 1. Questionnaire answers ──────────────────────────────────────────────
    lines += ["## 1. Questionnaire Answers", ""]
    ANSWER_LABELS: dict[str, str] = {
        "app_type":             "App type",
        "agentic_level":        "Agentic level",
        "scale":                "Scale",
        "priority":             "Priority",
        "budget":               "Budget",
        "context_size":         "Context size",
        "privacy":              "Privacy requirements",
        "provider_preferences": "Provider preferences",
        "latency":              "Latency requirement",
        "app_description":      "Free-text description",
    }
    lines.append("| Question | Answer |")
    lines.append("|---|---|")
    for key, label in ANSWER_LABELS.items():
        val = answers.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val) if val else "—"
        lines.append(f"| {label} | {val} |")
    lines.append("")

    # ── 2. Analyzer interpretation ────────────────────────────────────────────
    app_summary  = analysis.get("app_summary", {})
    de           = analysis.get("description_extracts", {})
    stage_ptrs   = analysis.get("stage_pointers", {})
    stage_toks   = analysis.get("stage_token_estimates", {})

    lines += ["## 2. How Answers Were Interpreted (Analyzer)", ""]

    lines += ["### App Summary", ""]
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for k, v in app_summary.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    if de:
        lines += ["### Extracts from Free-Text Description", ""]
        if de.get("has_model_constraint"):
            lines.append(
                f'- **Provider constraint** detected — source: *"{de.get("constraint_source", "")}"*'
            )
            lines.append(f'  - Providers enforced: `{", ".join(de.get("preferred_providers", []))}`')
            hints = de.get("preferred_model_hints", [])
            if hints:
                lines.append(f'  - Model hints: `{", ".join(hints)}`')
        if de.get("has_token_override"):
            lines.append(
                f'- **Token override** detected — source: *"{de.get("token_override_source", "")}"*'
            )
            lines.append(f'  - avg_input_tokens → `{de.get("avg_input_tokens_override")}`')
            lines.append(f'  - avg_output_tokens → `{de.get("avg_output_tokens_override")}`')
        if de.get("has_scale_override"):
            lines.append(
                f'- **Scale override** detected — source: *"{de.get("scale_override_source", "")}"*'
            )
            lines.append(f'  - active_users → `{de.get("active_users")}`')
            lines.append(f'  - daily_executions → `{de.get("daily_executions")}`')
            lines.append(f'  - monthly_executions → `{de.get("monthly_executions")}`')
        if not any(de.get(k) for k in ("has_model_constraint", "has_token_override", "has_scale_override")):
            lines.append("*No structured overrides extracted from description.*")
        lines.append("")

    lines += ["### Per-Stage Requirements", ""]
    lines.append("| SDLC Stage | Capabilities Required | Min Context Window |")
    lines.append("|---|---|---|")
    for stage in SDLC_STAGES:
        ptr = stage_ptrs.get(stage, {})
        caps = ", ".join(ptr.get("capabilities", [])) or "—"
        ctx  = f"{ptr.get('min_context_tokens', 0):,}"
        lines.append(f"| {stage} | {caps} | {ctx} |")
    lines.append("")

    lines += ["### Token Estimates from Analyzer", ""]
    lines.append("| SDLC Stage | Avg Input Tokens | Avg Output Tokens | Reasoning Tokens |")
    lines.append("|---|---|---|---|")
    for stage in SDLC_STAGES:
        te = stage_toks.get(stage, {})
        lines.append(
            f"| {stage} "
            f"| {te.get('avg_input_tokens', 0):,} "
            f"| {te.get('avg_output_tokens', 0):,} "
            f"| {te.get('avg_reasoning_tokens', 0):,} |"
        )
    lines.append("")

    # ── 3. Model scoring ──────────────────────────────────────────────────────
    lines += [
        "## 3. Model Scoring (Catalog Matcher)",
        "",
        "**Scoring formula** (0–100, before diversity penalty):",
        "",
        "```",
        "score = (cap_score × 0.40) + (ctx_score × 0.25) + (cost_score × 0.25) + (tier_score × 0.10)",
        "",
        "  cap_score   = capabilities_matched / capabilities_required",
        "  ctx_score   = log(model_context) / log(max(min_ctx × 4, model_context))   [capped at 1.0]",
        "  cost_score  = 1 − min(1, (input_per_1K + output_per_1K) / $1.00_ceiling)",
        "  tier_score  = 1.0 (Tier-1 provider) | 0.75 (Tier-2) | 0.50 (Tier-3)",
        "```",
        "",
        "Diversity penalties applied after scoring:",
        "- Same model repeated across stages: ×0.40",
        "- Same model family repeated: ×0.60",
        "- Same provider repeated: ×0.85",
        "",
    ]

    for stage in SDLC_STAGES:
        cat   = stage_catalog.get(stage, {})
        tiers = cat.get("tiers", {})
        candidates = cat.get("candidates", [])
        if not candidates:
            continue
        lines.append(f"### {stage}")
        lines.append("")
        lines.append("**Top candidates:**")
        lines.append("")
        lines.append("| Model | Score | Provider | Context |")
        lines.append("|---|---|---|---|")
        for c in candidates[:6]:
            lines.append(
                f"| `{c.get('model_id', '?')}` "
                f"| {c.get('score', 0)} "
                f"| {c.get('provider', '?')} "
                f"| {c.get('context_length', 0):,} |"
            )
        lines.append("")
        lines.append("**Tier picks:**")
        lines.append("")
        for tier_name, tm in tiers.items():
            if isinstance(tm, dict):
                lines.append(f"- **{tier_name}**: `{tm.get('model_id', '?')}` (score {tm.get('score', '?')})")
        lines.append("")

    # ── 4. Token calculations ─────────────────────────────────────────────────
    lines += ["## 4. Token Calculations", ""]
    stage_recs = output.get("stage_recommendations", [])
    total_in = total_out = total_reason = 0

    lines.append("| SDLC Stage | Avg Input | Avg Output | Avg Reasoning | Stage Total |")
    lines.append("|---|---|---|---|---|")
    for sr in stage_recs:
        wp    = sr.get("workload_profile", {})
        s_in  = wp.get("avg_input_tokens",     0)
        s_out = wp.get("avg_output_tokens",    0)
        s_rea = wp.get("avg_reasoning_tokens", 0)
        total = s_in + s_out + s_rea
        total_in     += s_in
        total_out    += s_out
        total_reason += s_rea
        lines.append(
            f"| {sr.get('stage_name', '?')} "
            f"| {s_in:,} | {s_out:,} | {s_rea:,} | {total:,} |"
        )
    grand_total = total_in + total_out + total_reason
    lines.append(
        f"| **TOTAL** "
        f"| **{total_in:,}** | **{total_out:,}** | **{total_reason:,}** | **{grand_total:,}** |"
    )
    lines.append("")
    lines.append(
        "> Total tokens per request = sum across all 8 SDLC stages. "
        "This overrides the LLM's top-level workload_profile estimate."
    )
    lines.append("")

    # ── 5. Cost calculations ──────────────────────────────────────────────────
    lines += ["## 5. Cost Calculations (Recommended Model per Stage)", ""]
    lines.append(
        "**Formula:** `cost = (input_tokens / 1000 × $/1K_in) + (output_tokens / 1000 × $/1K_out)`"
    )
    lines.append("")
    lines.append("| SDLC Stage | Model | $/1K in | $/1K out | Tokens in | Tokens out | Cost/request |")
    lines.append("|---|---|---|---|---|---|---|")

    total_cost_per_req = 0.0
    for sr in stage_recs:
        models    = sr.get("models", {})
        rec_id    = models.get("recommended_model_id", "?")
        cb        = sr.get("_cost_breakdown", {})
        in_toks   = cb.get("input_tokens",  sr.get("workload_profile", {}).get("avg_input_tokens",  0))
        out_toks  = cb.get("output_tokens", sr.get("workload_profile", {}).get("avg_output_tokens", 0))
        in_1k     = cb.get("input_per_1k",  0.0)
        out_1k    = cb.get("output_per_1k", 0.0)
        cost      = sr.get("estimated_cost_per_request", 0.0)
        total_cost_per_req += cost
        lines.append(
            f"| {sr.get('stage_name', '?')} "
            f"| `{rec_id}` "
            f"| ${_fmt(in_1k, 6)} "
            f"| ${_fmt(out_1k, 6)} "
            f"| {in_toks:,} "
            f"| {out_toks:,} "
            f"| ${_fmt(cost, 6)} |"
        )
    lines.append(
        f"| **TOTAL** | | | | | | **${_fmt(total_cost_per_req, 6)}** |"
    )
    lines.append("")

    # ── 5b. Projected usage costs ─────────────────────────────────────────────
    wp_global = output.get("workload_profile", {})
    active_users  = wp_global.get("active_users", 0)
    req_per_user  = wp_global.get("requests_per_user_per_day", 0)
    duration_mo   = wp_global.get("project_duration_months", 0)
    cached_toks   = wp_global.get("avg_cached_tokens", 0)
    cache_eligible = wp_global.get("cache_eligible", False)

    if active_users and req_per_user and total_cost_per_req > 0:
        daily_reqs  = active_users * req_per_user
        daily_cost  = daily_reqs * total_cost_per_req
        monthly_cost = daily_cost * 30
        project_cost = monthly_cost * duration_mo if duration_mo else None

        lines += ["### Projected Usage Costs", ""]
        lines.append(
            f"- **Daily requests:** {active_users:,} users × {req_per_user} req/user = **{daily_reqs:,} req/day**"
        )
        lines.append(f"- **Cost per request:** ${_fmt(total_cost_per_req, 6)}")
        lines.append(f"- **Daily cost:** {daily_reqs:,} × ${_fmt(total_cost_per_req, 6)} = **${daily_cost:,.4f}**")
        lines.append(f"- **Monthly cost (30 days):** **${monthly_cost:,.2f}**")
        if project_cost is not None:
            lines.append(f"- **Project total ({duration_mo} months):** **${project_cost:,.2f}**")

        if cache_eligible and cached_toks > 0:
            lines.append("")
            lines.append(f"### Cache Savings Estimate")
            lines.append("")
            lines.append(
                f"- {cached_toks:,} tokens cached per request (cache_eligible: true)"
            )
            lines.append(
                "- Cache reads are typically ~10× cheaper than regular input tokens."
            )
            lines.append(
                "- Implement prompt caching on repeated system-prompt / context prefixes "
                "to reduce the effective input cost."
            )
        lines.append("")

    # ── 6. Final selections ───────────────────────────────────────────────────
    lines += ["## 6. Final Model Selections (Synthesizer)", ""]

    smr = output.get("single_model_recommendations", [])
    if smr:
        lines += ["### Overall Single-Model Recommendations", ""]
        lines.append("| Tier | Model | Why | Tradeoffs |")
        lines.append("|---|---|---|---|")
        for rec in smr:
            lines.append(
                f"| {rec.get('category', '?')} "
                f"| `{rec.get('model_id', '?')}` "
                f"| {rec.get('why', '')} "
                f"| {rec.get('tradeoffs', '')} |"
            )
        lines.append("")

    arch = output.get("architecture", {})
    if arch:
        lines += ["### Architecture", ""]
        lines.append(f"- **Pattern:** {arch.get('pattern', '?')}")
        lines.append(f"- **Hosting:** {arch.get('hosting_strategy', '?')}")
        fw = arch.get("agent_framework_recommendation")
        if fw:
            lines.append(f"- **Framework:** {fw}")
        constraints = arch.get("framework_constraints", [])
        if constraints:
            lines.append(f"- **Constraints:** {', '.join(constraints)}")
        roles = arch.get("roles", [])
        if roles:
            lines.append("")
            lines.append("**Roles:**")
            lines.append("")
            lines.append("| Role | Recommended | Budget | Premium | Reason |")
            lines.append("|---|---|---|---|---|")
            for r in roles:
                lines.append(
                    f"| {r.get('role', '?')} "
                    f"| `{r.get('recommended_model_id', '?')}` "
                    f"| `{r.get('budget_model_id', '?')}` "
                    f"| `{r.get('premium_model_id', '?')}` "
                    f"| {r.get('reason', '')} |"
                )
        notes = arch.get("notes", [])
        if notes:
            lines.append("")
            lines.append("**Notes:**")
            for note in notes:
                lines.append(f"- {note}")
        lines.append("")

    # ── 7. Optimisation tips ──────────────────────────────────────────────────
    tips = output.get("optimisation_tips", [])
    if tips:
        lines += ["## 7. Optimisation Tips", ""]
        for tip in tips:
            impact = tip.get("impact", "medium").upper()
            lines.append(f"**[{impact}] {tip.get('title', '')}**")
            lines.append(f"> {tip.get('detail', '')}")
            lines.append("")

    # ── 8. Confidence ─────────────────────────────────────────────────────────
    conf = output.get("confidence", {})
    lines += [
        "## 8. Confidence",
        "",
        f"**Score:** {conf.get('score', '?').upper()}",
        "",
        f"**Reason:** {conf.get('reason', '')}",
        "",
    ]
    assumptions = conf.get("assumptions", [])
    if assumptions:
        lines.append("**Assumptions:**")
        for a in assumptions:
            lines.append(f"- {a}")
        lines.append("")

    return "\n".join(lines)


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
        analysis              = _run_analyzer(answers, timestamp)
        app_summary           = analysis.get("app_summary", {})
        stage_pointers        = analysis.get("stage_pointers", {})
        stage_token_estimates = analysis.get("stage_token_estimates", {})
        description_extracts  = analysis.get("description_extracts", {})
        log.log_info(f"  done ({time.time()-t0:.1f}s)")

        # -- Deterministic enforcement of description_extracts ------------------
        # The Analyzer (LLM) understood app_description and structured what it found.
        # We now enforce those extracts deterministically before passing to later steps.

        # 1. Provider / model constraints → feed catalog matcher so only matching
        #    models appear as candidates; also passed to Synthesizer as hard constraint.
        explicit_providers = [p.lower() for p in answers.get("provider_preferences", [])]
        desc_providers     = [p.lower() for p in description_extracts.get("preferred_providers", [])]
        seen: set          = set()
        preferred_providers: list[str] = []
        for p in explicit_providers + desc_providers:   # explicit takes priority
            if p not in seen:
                seen.add(p)
                preferred_providers.append(p)
        if preferred_providers:
            log.log_info(f"Provider filter active (enforced): {preferred_providers}")

        # 2. Token overrides → replace stage_token_estimates so Synthesizer uses
        #    real numbers from app_description, not generic defaults.
        if description_extracts.get("has_token_override"):
            inp_override = description_extracts.get("avg_input_tokens_override")
            out_override = description_extracts.get("avg_output_tokens_override")
            if inp_override or out_override:
                for stage in stage_token_estimates:
                    if inp_override:
                        stage_token_estimates[stage]["avg_input_tokens"]  = inp_override
                    if out_override:
                        stage_token_estimates[stage]["avg_output_tokens"] = out_override
                log.log_info(
                    f"Token estimates overridden from app_description: "
                    f"input={inp_override} output={out_override}"
                )

        # 3. Scale overrides → patch app_summary so Synthesizer workload_profile is accurate.
        if description_extracts.get("has_scale_override"):
            if description_extracts.get("active_users"):
                app_summary["active_users"] = description_extracts["active_users"]
            if description_extracts.get("daily_executions"):
                app_summary["daily_executions"] = description_extracts["daily_executions"]
            if description_extracts.get("monthly_executions"):
                app_summary["monthly_executions"] = description_extracts["monthly_executions"]

        # -- Step 2: Catalog matcher (pure Python) ------------------------------
        log.log_info("Step 2/3 — Catalog Matcher")
        stage_catalog = _run_catalog_matcher(
            stage_pointers,
            privacy_regional=privacy_regional,
            preferred_providers=preferred_providers if preferred_providers else None,
        )
        log.log_info(f"  done ({time.time()-t0:.1f}s)")

        # -- Step 3: Synthesizer (1 LLM call) ─────────────────────────────────
        log.log_info("Step 3/3 — Synthesizer")
        final = _run_synthesizer(
            app_summary, stage_catalog, stage_token_estimates, answers, input_hash, timestamp,
            preferred_providers=preferred_providers if preferred_providers else None,
            description_extracts=description_extracts if description_extracts else None,
        )
        log.log_info(f"  done ({time.time()-t0:.1f}s total)")
        if final:
            log.log_info("Recommendation generated successfully")
            normalized = _normalize(final, input_hash, answers)
            # Stash intermediate data so the API layer can build the full reasoning
            # report after pricing is infused (pricing rates aren't available here).
            normalized["_reasoning_inputs"] = {
                "answers":       answers,
                "analysis":      analysis,
                "stage_catalog": stage_catalog,
            }
            return normalized  # type: ignore
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