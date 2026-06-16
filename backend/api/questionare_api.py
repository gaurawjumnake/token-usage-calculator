import json
from pathlib import Path
import os
from fastapi import APIRouter, HTTPException, status
from backend.utilites.app_logger import Logger

from dotenv import load_dotenv
load_dotenv()
print(f"Loaded environment variables: {os.getenv('QUESTIONNAIRE_PATH')}")

QUESTIONNAIRE_PATH = Path("backend/data/questionnaire_v2.json")
log = Logger()
router = APIRouter()


def _load_questionnaire() -> dict:
    if not QUESTIONNAIRE_PATH.exists():
        log.log_error(f"Questionnaire file not found | path={QUESTIONNAIRE_PATH}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Questionnaire file not found at expected path: {QUESTIONNAIRE_PATH}",
        )

    try:
        with open(QUESTIONNAIRE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        log.log_error(f"Questionnaire file is not valid JSON | error={e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Questionnaire file contains invalid JSON: {e}",
        )
    except OSError as e:
        log.log_error(f"Questionnaire file could not be read | error={e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Questionnaire file could not be read.",
        )

    # Basic structure validation
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Questionnaire file must be a JSON object at the top level.",
        )

    if "sections" not in data or not isinstance(data["sections"], list):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Questionnaire file is missing required 'sections' array.",
        )

    if len(data["sections"]) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Questionnaire file has an empty 'sections' array.",
        )

    return data


@router.get(
    "/questionnaire",
    status_code=status.HTTP_200_OK,
    summary="Get questionnaire",
    description="Returns the LLM selection questionnaire loaded from the local JSON file.",
)
def get_questionnaire():
    log.log_info(f"API: questionnaire requested | path={QUESTIONNAIRE_PATH}")
    data = _load_questionnaire()
    log.log_info(f"API: questionnaire returned | sections={len(data['sections'])}")
    return data