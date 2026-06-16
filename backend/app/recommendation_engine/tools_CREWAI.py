import json
from typing import List, Dict, Any, Optional
from crewai.tools import tool
from backend.utilites.app_logger import Logger

log = Logger()


# ── Module-level singleton ─────────────────────────────────────────────────────
_MODEL_CATALOG: Dict[str, Any] | None = None

def _get_catalog() -> Dict[str, Any]:
    """Load catalog once, return cached copy on subsequent calls."""
    global _MODEL_CATALOG
    if _MODEL_CATALOG is None:
        with open("backend/data/model_catalog.json", "r", encoding="utf-8") as f:
            _MODEL_CATALOG = json.load(f)
        log.log_info(f"Model catalog loaded with -{len(_MODEL_CATALOG)}- models") #type:ignore

    return _MODEL_CATALOG #type:ignore


def _build_capabilities(model: Dict[str, Any]) -> set:
    """
    Derive a canonical capability set from a raw catalog record.

    Capabilities recognised:
        vision          image in input_modalities
        video           video in input_modalities
        file            file in input_modalities
        tools           "tools" in supported_parameters
        structured_output "structured_outputs" in supported_parameters
        reasoning       "reasoning" OR "include_reasoning" in supported_parameters
        long_context    context_length >= 64 000 tokens
        very_long_context context_length >= 200 000 tokens
        multi_turn      assumed true for all chat models (no catalog signal to deny)
        real_time       assumed true for all API models
    """
    arch = model.get("architecture", {})
    supported_params = set(model.get("supported_parameters", []))
    ctx = model.get("context_length", 0)

    caps: set[str] = set()

    # Modality-based
    input_mods = arch.get("input_modalities", [])
    if "image" in input_mods:
        caps.add("vision")
    if "video" in input_mods:
        caps.add("video")
    if "file" in input_mods:
        caps.add("file")

    # Parameter-based
    if "tools" in supported_params or "tool_choice" in supported_params:
        caps.add("tools")
    if "structured_outputs" in supported_params or "response_format" in supported_params:
        caps.add("structured_output")
    if "reasoning" in supported_params or "include_reasoning" in supported_params:
        caps.add("reasoning")

    # Context-based
    if ctx >= 64_000:
        caps.add("long_context")
    if ctx >= 200_000:
        caps.add("very_long_context")

    # Assumed for all API models
    caps.add("multi_turn")
    caps.add("real_time")

    return caps


# ── Query Model Catalog ────────────────────────────────────────────────────────

@tool("Query Model Catalog")
def query_model_catalog(
    required_capabilities: List[str],
    min_context_length: int,
    max_cost_per_1k_input: float = 999,
    max_cost_per_1k_output: float = 999,
    framework_constraint: Optional[str] = None,
    limit: int = 25,
) -> Dict[str, Any]:
    """
    Query the model catalog and return compatible models.

    Args:
        required_capabilities: List of required capabilities.
            Recognised values: vision, video, file, tools, structured_output,
            reasoning, long_context, very_long_context, multi_turn, real_time
        min_context_length: Minimum context window in tokens
        max_cost_per_1k_input: Max cost per 1K input tokens  (default: no limit)
        max_cost_per_1k_output: Max cost per 1K output tokens (default: no limit)
        framework_constraint: Optional – "anthropic_only" | "openai_only"
        limit: Max models to return (default 25)

    Returns:
        Dict with 'models' list and 'count'. If no models match, automatically
        retries with relaxed constraints (drops long_context, then reasoning).
    """
    try:
        catalog = _get_catalog()

        def _run_query(caps_required: List[str], min_ctx: int) -> List[Dict]:
            candidates = []
            for model_key, raw_model in catalog.items():
                model = {"model_id": model_key, **raw_model}

                # Context filter
                if model.get("context_length", 0) < min_ctx:
                    continue

                # Pricing filter (catalog prices are per-token; convert to per-1K)
                pricing = model.get("pricing", {})
                input_price_per_tok  = float(pricing.get("prompt",     0))
                output_price_per_tok = float(pricing.get("completion",  0))
                input_price_per_1k   = input_price_per_tok  * 1_000
                output_price_per_1k  = output_price_per_tok * 1_000

                if input_price_per_1k > max_cost_per_1k_input:
                    continue
                if output_price_per_1k > max_cost_per_1k_output:
                    continue

                # Capability filter
                model_capabilities = _build_capabilities(model)
                if not all(cap in model_capabilities for cap in caps_required):
                    continue

                # Framework filter
                if framework_constraint:
                    provider = model.get("provider", "").lower()
                    if framework_constraint == "anthropic_only" and provider != "anthropic":
                        continue
                    if framework_constraint == "openai_only" and provider != "openai":
                        continue

                # Score
                cap_match   = (len([c for c in caps_required if c in model_capabilities])
                               / max(len(caps_required), 1))
                ctx_adequacy = min(1.0, model.get("context_length", 0) / max(min_ctx, 1))
                total_cost_1k = input_price_per_1k + output_price_per_1k
                cost_score   = 1.0 - min(1.0, total_cost_1k / 10.0)  # normalised to $10/1K ceiling

                score = (cap_match * 0.4) + (ctx_adequacy * 0.3) + (max(0.0, cost_score) * 0.3)

                candidates.append({
                    "model_id":       model.get("model_id"),
                    "name":           model.get("name"),
                    "provider":       model.get("provider"),
                    "context_length": model.get("context_length"),
                    "pricing": {
                        "input_per_1k":  round(input_price_per_1k,  6),
                        "output_per_1k": round(output_price_per_1k, 6),
                    },
                    "capabilities":    list(model_capabilities),
                    "relevance_score": round(score * 100, 1),
                })

            candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
            return candidates[:limit]

        # ── Attempt 1: strict ─────────────────────────────────────────────────
        results = _run_query(required_capabilities, min_context_length)

        # ── Attempt 2: drop non-critical virtual caps (long_context, real_time, multi_turn)
        if not results:
            relaxed = [c for c in required_capabilities
                       if c not in ("long_context", "very_long_context", "multi_turn", "real_time")]
            log.log_info("No results with strict caps — retrying without virtual caps")
            results = _run_query(relaxed, min_context_length)

        # ── Attempt 3: also drop reasoning ───────────────────────────────────
        if not results:
            relaxed2 = [c for c in required_capabilities
                        if c not in ("long_context", "very_long_context",
                                     "multi_turn", "real_time", "reasoning")]
            log.log_info("Still no results — retrying without reasoning")
            results = _run_query(relaxed2, min_context_length)

        # ── Attempt 4: halve min_context and drop all optional caps ───────────
        if not results:
            core_caps = [c for c in required_capabilities if c in ("vision", "tools")]
            log.log_info("Still no results — relaxing context to 50%% and keeping only core caps")
            results = _run_query(core_caps, min_context_length // 2)

        # ── Attempt 5: bare minimum — any model ──────────────────────────────
        if not results:
            log.log_info("Falling back to top models with no capability filter")
            results = _run_query([], 0)

        return {
            "models":  results,
            "count":   len(results),
            "filters_applied": {
                "min_context":           min_context_length,
                "max_input_cost":        max_cost_per_1k_input,
                "max_output_cost":       max_cost_per_1k_output,
                "required_capabilities": required_capabilities,
            },
        }

    except FileNotFoundError:
        return {"error": "Model catalog not found", "models": [], "count": 0}
    except Exception as e:
        return {"error": f"Query failed: {str(e)}", "models": [], "count": 0}


def refresh_catalog() -> None:
    """Force a reload of the model catalog on the next query."""
    global _MODEL_CATALOG
    _MODEL_CATALOG = None
    _get_catalog()


# ── Validate Context Window ────────────────────────────────────────────────────

@tool("Validate Context Window")
def validate_context_window(
    required_tokens: int,
    model_context_window: int,
    include_margin: bool = True,
) -> Dict[str, Any]:
    """
    Validate if model's context window meets requirement.

    Args:
        required_tokens: Total tokens needed
        model_context_window: Model's max context in tokens
        include_margin: Whether to require 20% safety margin

    Returns:
        Dict with validation result and margin analysis
    """
    try:
        if required_tokens < 0 or model_context_window < 0:
            return {"error": "Invalid token counts", "is_valid": False, "passes_validation": False}

        if include_margin:
            required_with_margin = required_tokens * 1.2
            is_valid = model_context_window >= required_with_margin
            margin_required = 20.0
        else:
            is_valid = model_context_window >= required_tokens
            margin_required = 0.0

        margin = ((model_context_window - required_tokens) / max(required_tokens, 1)) * 100

        return {
            "is_valid":                   is_valid,
            "required_tokens":            required_tokens,
            "model_window":               model_context_window,
            "margin_percentage":          round(margin, 1),
            "margin_required_percentage": margin_required,
            "passes_validation":          is_valid,
            "message": (
                f"Adequate context window with {margin:.1f}% margin" if is_valid
                else f"Insufficient context window (only {margin:.1f}% margin, need {margin_required}%)"
            ),
        }

    except Exception as e:
        return {"error": f"Validation failed: {str(e)}", "is_valid": False, "passes_validation": False}


# ── Rank Models by Fitness ─────────────────────────────────────────────────────

@tool("Rank Models by Fitness")
def rank_models(
    candidate_models: List[Dict[str, Any]],
    required_capabilities: List[str],
    complexity: str,
    budget_category: str,
) -> Dict[str, Any]:
    """
    Score and rank models based on fitness for use case.

    Args:
        candidate_models: Models from query_model_catalog
        required_capabilities: Capabilities needed
        complexity: "low", "medium", or "high"
        budget_category: "recommended", "budget", or "premium"

    Returns:
        List of ranked models with scores
    """
    try:
        if not candidate_models:
            return {"error": "No models to rank", "ranked_models": [], "weights_applied": {}}

        if budget_category == "budget":
            weights = {
                "capability_match": 0.25,
                "context_adequacy": 0.20,
                "complexity_fit":   0.15,
                "cost_efficiency":  0.35,
                "reasoning_quality":0.03,
                "latency":          0.02,
            }
        elif budget_category == "premium":
            weights = {
                "capability_match": 0.40,
                "context_adequacy": 0.20,
                "complexity_fit":   0.15,
                "cost_efficiency":  0.05,
                "reasoning_quality":0.15,
                "latency":          0.05,
            }
        else:  # recommended (balanced)
            weights = {
                "capability_match": 0.35,
                "context_adequacy": 0.20,
                "complexity_fit":   0.20,
                "cost_efficiency":  0.15,
                "reasoning_quality":0.05,
                "latency":          0.05,
            }

        ranked = []
        for i, model in enumerate(candidate_models):
            model_capabilities = set(model.get("capabilities", []))

            capability_match = (
                len([c for c in required_capabilities if c in model_capabilities])
                / max(len(required_capabilities), 1)
            )

            context_adequacy = min(1.0, model.get("context_length", 96_000) / 96_000)

            if complexity == "low":
                complexity_fit = 1.0
            elif complexity == "medium":
                complexity_fit = 0.9 if "reasoning" in model_capabilities else 0.7
            else:
                complexity_fit = 1.0 if "reasoning" in model_capabilities else 0.5

            pricing    = model.get("pricing", {})
            total_cost = pricing.get("input_per_1k", 0) + pricing.get("output_per_1k", 0)
            # total_cost is already per-1K here; normalise against $10 ceiling
            cost_efficiency  = max(0.0, 1.0 - min(1.0, total_cost / 10.0))
            reasoning_quality = 1.0 if "reasoning" in model_capabilities else 0.5

            score = (
                capability_match   * weights["capability_match"]  +
                context_adequacy   * weights["context_adequacy"]  +
                complexity_fit     * weights["complexity_fit"]     +
                cost_efficiency    * weights["cost_efficiency"]    +
                reasoning_quality  * weights["reasoning_quality"] +
                1.0                * weights["latency"]            # assume adequate
            ) * 100

            ranked.append({
                "rank":     i + 1,
                "model_id": model.get("model_id"),
                "score":    round(score, 1),
                "component_scores": {
                    "capability_match":   round(capability_match  * 100, 1),
                    "context_adequacy":   round(context_adequacy  * 100, 1),
                    "complexity_fit":     round(complexity_fit    * 100, 1),
                    "cost_efficiency":    round(cost_efficiency   * 100, 1),
                    "reasoning_quality":  round(reasoning_quality * 100, 1),
                    "latency":            100.0,
                },
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        for i, m in enumerate(ranked):
            m["rank"] = i + 1

        return {"ranked_models": ranked, "weights_applied": weights}

    except Exception as e:
        return {"error": f"Ranking failed: {str(e)}", "ranked_models": [], "weights_applied": {}}


# ── Check Framework Compatibility ──────────────────────────────────────────────

@tool("Check Framework Compatibility")
def check_framework_compatibility(
    model_provider: str,
    framework_name: str,
) -> Dict[str, Any]:
    """
    Check if model provider is compatible with framework.

    Args:
        model_provider: OpenAI, Anthropic, Google, Cohere, etc.
        framework_name: CrewAI, LangChain, LangGraph, etc.

    Returns:
        Dict with compatibility info and constraints
    """
    compatibility_matrix = {
        "CrewAI": {
            "OpenAI":    {"compatible": True,  "level": "full"},
            "Groq":      {"compatible": True,  "level": "full"},
            "Together":  {"compatible": True,  "level": "full"},
            "Anthropic": {"compatible": False, "level": "none"},
            "Google":    {"compatible": False, "level": "none"},
            "Cohere":    {"compatible": False, "level": "none"},
        },
        "LangChain": {
            "OpenAI":    {"compatible": True, "level": "full"},
            "Anthropic": {"compatible": True, "level": "full"},
            "Google":    {"compatible": True, "level": "full"},
            "Cohere":    {"compatible": True, "level": "full"},
            "Groq":      {"compatible": True, "level": "full"},
        },
        "LangGraph": {
            "OpenAI":    {"compatible": True, "level": "full"},
            "Anthropic": {"compatible": True, "level": "full"},
            "Google":    {"compatible": True, "level": "full"},
            "Cohere":    {"compatible": True, "level": "full"},
        },
        "OpenAI Assistants": {
            "OpenAI":    {"compatible": True,  "level": "full"},
            "Anthropic": {"compatible": False, "level": "none"},
            "Google":    {"compatible": False, "level": "none"},
        },
        "Claude MCP": {
            "Anthropic": {"compatible": True,  "level": "full"},
            "OpenAI":    {"compatible": False, "level": "none"},
            "Google":    {"compatible": False, "level": "none"},
        },
        "AutoGen": {
            "OpenAI":    {"compatible": True, "level": "full"},
            "Anthropic": {"compatible": True, "level": "partial"},
            "Google":    {"compatible": True, "level": "partial"},
        },
    }

    try:
        framework = framework_name.strip()
        provider  = model_provider.strip()

        if framework not in compatibility_matrix:
            return {
                "is_compatible":       True,
                "model_provider":      provider,
                "framework":           framework,
                "compatibility_level": "unknown",
                "message": f"Unknown framework '{framework}'. Assuming compatible.",
            }

        fw = compatibility_matrix[framework]
        if provider not in fw:
            return {
                "is_compatible":       True,
                "model_provider":      provider,
                "framework":           framework,
                "compatibility_level": "unknown",
                "message": f"Unknown provider '{provider}' for framework '{framework}'.",
            }

        info = fw[provider]
        return {
            "is_compatible":       info["compatible"],
            "model_provider":      provider,
            "framework":           framework,
            "compatibility_level": info["level"],
            "message": (
                f"{provider} works with {framework}."
                if info["compatible"]
                else f"{provider} is not compatible with {framework}."
            ),
        }

    except Exception as e:
        return {"error": f"Check failed: {str(e)}", "is_compatible": False}
