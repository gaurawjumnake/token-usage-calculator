import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query, status
from backend.utilites.app_logger import Logger

load_dotenv()

log = Logger()
router = APIRouter()

_env_path = "backend/processed_data/openrouter_models.json" # os.getenv("MODELS_JSON_PATH")
MODELS_JSON_PATH = (
    Path(_env_path).resolve()
    if _env_path
    else Path(__file__).resolve().parent.parent / "data" / "models.json"
)


ROUTER_TOKENIZERS  = {"Router"}
ROUTER_PRICING_FLAGS = {"-1"}          # prompt/completion == "-1"
FREE_PRICING_FLAG  = "0"


def _load_raw() -> list[dict]:
    if not MODELS_JSON_PATH.exists():
        log.log_error(f"Models file not found | path={MODELS_JSON_PATH}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Models JSON file not found. "
                f"Expected: {MODELS_JSON_PATH}. "
                f"Set MODELS_JSON_PATH env var to override."
            ),
        )
    try:
        with open(MODELS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        log.log_error(f"Invalid JSON in models file | error={e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Models file contains invalid JSON: {e}",
        )
    except OSError as e:
        log.log_error(f"Cannot read models file | error={e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Models file could not be read. Check file permissions.",
        )

    if not isinstance(data, list):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Models file must be a JSON array at the top level.",
        )
    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Models file is empty.",
        )

    return data


def _normalise(model: dict) -> dict:
    """
    Fix the typo in source data (provier → provider) and
    convert pricing strings to floats × 1,000,000 (per-million tokens).
    Also add derived helper fields for the calculator UI.
    """
    out = dict(model)

    # Normalise pricing to per-million
    raw_pricing = out.get("pricing") or {}
    per_million: dict[str, float | None] = {}
    is_free   = False
    is_router = False

    for key, val in raw_pricing.items():
        str_val = str(val).strip()
        if str_val in ROUTER_PRICING_FLAGS:
            is_router = True
            per_million[key] = None
        elif str_val == FREE_PRICING_FLAG:
            is_free = True
            per_million[key] = 0.0
        else:
            try:
                per_million[key] = round(float(str_val) * 1_000_000, 6)
            except (ValueError, TypeError):
                per_million[key] = None

    out["pricing_per_million"] = per_million
    out["is_free"]   = is_free
    out["is_router"] = is_router or (
        out.get("architecture", {}).get("tokenizer") in ROUTER_TOKENIZERS
    )

    return out


def _is_expired(model: dict) -> bool:
    expiry = model.get("expiration_date")
    if not expiry:
        return False
    try:
        from datetime import date
        return date.fromisoformat(expiry) < date.today()
    except (ValueError, TypeError):
        return False


def _extract_provider(model: dict) -> str:
    """Extract provider slug from model_id or provier field."""
    return model.get("provider") or model.get("provier") or ""



@router.get(
    "/models",
    status_code=status.HTTP_200_OK,
    summary="Get all LLM models",
    description=(
        "Returns the full list of LLM models from the local JSON file. "
        "Pricing is normalised to USD per 1 million tokens. "
        "Router/aggregator models are excluded by default."
    ),
)
def get_models(
    provider: Optional[str] = Query(
        None,
        description="Filter by provider slug (e.g. 'openai', 'anthropic', 'google'). Case-insensitive."
    ),
    include_free: bool = Query(
        True,
        description="Include models with zero pricing (free tier)."
    ),
    include_routers: bool = Query(
        False,
        description="Include OpenRouter aggregator/router models."
    ),
    include_expired: bool = Query(
        False,
        description="Include models with a past expiration_date."
    ),
    supports_vision: Optional[bool] = Query(
        None,
        description="Filter by vision/image input support."
    ),
    supports_tools: Optional[bool] = Query(
        None,
        description="Filter by function/tool calling support."
    ),
    supports_reasoning: Optional[bool] = Query(
        None,
        description="Filter by reasoning/thinking support."
    ),
    max_input_price: Optional[float] = Query(
        None,
        description="Maximum input (prompt) price per 1M tokens in USD."
    ),
    min_context_length: Optional[int] = Query(
        None,
        description="Minimum context window in tokens."
    ),
    search: Optional[str] = Query(
        None,
        description="Search by model name (case-insensitive substring match)."
    ),
):
    log.log_info(
        f"API: models requested | provider={provider} include_free={include_free} "
        f"include_routers={include_routers} include_expired={include_expired} "
        f"supports_vision={supports_vision} supports_tools={supports_tools} "
        f"max_input_price={max_input_price} min_context_length={min_context_length} "
        f"search={search}"
    )

    raw = _load_raw()
    models = [_normalise(m) for m in raw]

    # - Filters ----------------------------------------------------------------------

    if not include_routers:
        models = [m for m in models if not m["is_router"]]

    if not include_free:
        models = [m for m in models if not m["is_free"]]

    if not include_expired:
        models = [m for m in models if not _is_expired(m)]

    if provider:
        p = provider.lower()
        models = [m for m in models if _extract_provider(m).lower() == p]

    if search:
        s = search.lower()
        models = [m for m in models if s in m.get("name", "").lower()]

    if min_context_length is not None:
        models = [
            m for m in models
            if (m.get("context_length") or 0) >= min_context_length
        ]

    if max_input_price is not None:
        models = [
            m for m in models
            if (m["pricing_per_million"].get("prompt") or 0) <= max_input_price
        ]

    if supports_vision is not None:
        models = [
            m for m in models
            if ("image" in (m.get("architecture") or {}).get("input_modalities", []))
            == supports_vision
        ]

    if supports_tools is not None:
        params = m.get("supported_parameters", []) if (m := None) else []  # placeholder
        models = [
            m for m in models
            if ("tools" in (m.get("supported_parameters") or [])) == supports_tools
        ]

    if supports_reasoning is not None:
        models = [
            m for m in models
            if ("reasoning" in (m.get("supported_parameters") or [])
                or "include_reasoning" in (m.get("supported_parameters") or []))
            == supports_reasoning
        ]

    log.log_info(f"API: models returned | count={len(models)}")

    return {
        "total": len(models),
        "models": models,
    }


@router.get(
    "/models/{model_id}",
    status_code=status.HTTP_200_OK,
    summary="Get a single model by model_id",
    description="Returns a single model record by its model_id field. Pricing normalised to per-million USD.",
)
def get_model_by_id(model_id: str):
    log.log_info(f"API: model detail requested | model_id={model_id}")

    raw = _load_raw()
    for m in raw:
        if m.get("model_id") == model_id:
            log.log_info(f"API: model found | model_id={model_id}")
            return _normalise(m)

    log.log_warning(f"API: model not found | model_id={model_id}")
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Model with model_id='{model_id}' not found.",
    )


@router.get(
    "/models/providers/list",
    status_code=status.HTTP_200_OK,
    summary="Get list of unique providers",
    description="Returns a sorted list of all unique provider slugs in the models file.",
)
def get_providers():
    log.log_info("API: providers list requested")

    raw = _load_raw()
    providers = sorted(
        {_extract_provider(_normalise(m)) for m in raw if _extract_provider(_normalise(m))}
    )

    log.log_info(f"API: providers returned | count={len(providers)}")
    return {"total": len(providers), "providers": providers}
