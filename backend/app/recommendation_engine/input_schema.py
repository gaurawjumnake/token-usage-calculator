# input_schema.py
from pydantic import BaseModel, Field
import hashlib, json
from typing import Any

class QuestionnaireInput(BaseModel):
    answers: dict[str, Any] = Field(
        description="Key-value map of questionnaire question_id → answer",
        json_schema_extra={
            "example": {
                "app_type": "Enterprise Knowledge Assistant",
                "app_description": "Internal HR assistant that answers policy questions and summarises documents.",
                "context_size": "Large",
                "latency": "Fast",
                "scale": "Enterprise",
                "agentic_level": "Semi-Agentic",
                "agent_framework": "LangChain / LangGraph",
                "agent_structure": "Single agent",
                "capabilities": ["Function / tool calling", "Structured output / JSON", "Long context / large docs"],
                "uses_coding_agent": "No",
                "coding_tool": [],
                "priority": "Balanced",
                "budget": "$1,000–$10,000",
                "privacy": ["Data must stay in our region"]
            }
        }
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "answers": {
                    "app_type": "Enterprise Knowledge Assistant",
                    "app_description": "Internal HR assistant that answers policy questions and summarises documents.",
                    "context_size": "Large",
                    "latency": "Fast",
                    "scale": "Enterprise",
                    "agentic_level": "Semi-Agentic",
                    "agent_framework": "LangChain / LangGraph",
                    "agent_structure": "Single agent",
                    "capabilities": ["Function / tool calling", "Structured output / JSON"],
                    "uses_coding_agent": "No",
                    "coding_tool": [],
                    "priority": "Balanced",
                    "budget": "$1,000–$10,000",
                    "privacy": ["Data must stay in our region"]
                }
            }
        }
    }

    def stable_hash(self) -> str:
        canonical = json.dumps(self.answers, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()