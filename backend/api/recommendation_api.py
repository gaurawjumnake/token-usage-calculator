import asyncio
from fastapi import APIRouter, HTTPException, status

from backend.app.recommendation_engine.input_schema import QuestionnaireInput
from backend.app.recommendation_engine.output_schema_FINAL import RecommendationOutput_API
from backend.app.recommendation_engine.recommendation_engine import RecommendationEngine
from backend.app.recommendation_engine.model_data_extractor import ModelDataExtractor
from backend.utilites.app_logger import Logger

log = Logger()
router = APIRouter()
_REQUEST_TIMEOUT_SECONDS = 120 
_engine: RecommendationEngine | None = None
_extractor: ModelDataExtractor | None = None

def _get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine

def _get_extractor() -> ModelDataExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ModelDataExtractor()
    return _extractor

  

@router.post(
    "/recommend",
    response_model=RecommendationOutput_API,
    status_code=status.HTTP_200_OK,
    summary="Get LLM model recommendations",
    description=(
        "Accepts questionnaire answers and returns categorised model recommendations, "
        "token profiles, cost estimates, and architecture guidance."
    ),
)
async def get_recommendations(payload: QuestionnaireInput) -> RecommendationOutput_API:
    input_hash = payload.stable_hash()
    log.log_info(
        f"recommendation request | hash={input_hash} "
        f"app_type={payload.answers.get('app_type', 'unknown')} "
        f"scale={payload.answers.get('scale', 'unknown')}"
    )

    try:
        # Run the blocking engine in a thread so the event loop stays free.
        # asyncio.wait_for enforces a hard timeout on the whole pipeline.
        result = await asyncio.wait_for(
            asyncio.to_thread(_get_engine().run, payload),
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        log.log_info
        result = _get_extractor().infuse_model_pricing_data(result)
        
        if isinstance(result, dict) and "error" in result:
            log.log_error(f"extractor failed | hash={input_hash} | error={result['error']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process recommendation pricing data.",
            )

        log.log_info(f"recommendation generated | hash={input_hash}")
        return result #type:ignore

    except asyncio.TimeoutError:
        log.log_error(f"recommendation timed out | hash={input_hash}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Recommendation engine timed out after {_REQUEST_TIMEOUT_SECONDS}s. Please try again.",
        )

    except ValueError as e:
        log.log_error(f"schema validation failed | hash={input_hash} | error={e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    except Exception as e:
        log.log_error(f"engine failed | hash={input_hash} | error={e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation engine failed. Please try again.",
        )


@router.get(
    "/recommend/health",
    status_code=status.HTTP_200_OK,
    summary="Health check for recommendation engine",
)
def recommendation_health() -> dict:
    """
    Confirms the recommendation engine and catalog are reachable.
    Does not make an LLM call.
    """
    try:
        engine    = _get_engine()
        extractor = _get_extractor()
        return {
            "status":  "ok",
            "engine":  engine.__class__.__name__,
            "extractor": extractor.__class__.__name__,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Engine initialisation failed: {e}",
        )