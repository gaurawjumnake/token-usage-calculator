import json
from typing import Any

from pydantic import BaseModel
from backend.utilites.app_logger import Logger
log = Logger()


class ModelDataExtractor:
    def __init__(self):
        with open("backend/processed_data/openrouter_models.json", "r") as f:
            self.model_catalog = json.load(f)
            log.log_info(f"Loaded model catalog with {len(self.model_catalog)} entries")  

    @staticmethod
    def _to_dict(recommendation_output: Any) -> dict[str, Any]:
        if isinstance(recommendation_output, BaseModel):
            return recommendation_output.model_dump()
        if isinstance(recommendation_output, dict):
            return recommendation_output
        raise TypeError("recommendation_output must be a dict or Pydantic model")

    def extract_model_ids(self,
        recommendation_output: Any
    ) -> set[str]:
        payload = self._to_dict(recommendation_output)
        model_ids = set()

        # Single model recommendations (top-level recommended/budget/premium)
        for model in payload.get("single_model_recommendations", []):
            if not isinstance(model, dict):
                continue
            mid = model.get("model_id", "")
            if mid and isinstance(mid, str):
                model_ids.add(mid)

        # Stage-level recommendations
        for stage in payload.get("stage_recommendations", []):
            if not isinstance(stage, dict):
                continue
            models = stage.get("models", {})
            if not isinstance(models, dict):
                continue
            for key in ("recommended_model_id", "budget_model_id", "premium_model_id"):
                mid = models.get(key, "")
                if mid and isinstance(mid, str) and mid != "unknown":
                    model_ids.add(mid)

        # Architecture role recommendations
        architecture = payload.get("architecture", {})
        if isinstance(architecture, dict):
            for role in architecture.get("roles", []):
                if not isinstance(role, dict):
                    continue
                for key in ("recommended_model_id", "budget_model_id", "premium_model_id"):
                    mid = role.get(key, "")
                    if mid and isinstance(mid, str):
                        model_ids.add(mid)

        return model_ids

    @staticmethod
    def _normalise_pricing(raw_pricing: dict) -> dict:
        """
        OpenRouter stores prices as per-token strings (e.g. "0.000005" = $5 per 1M tokens).
        Convert to human-readable per-1K and per-1M rates.
        """
        def to_float(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        prompt_per_tok      = to_float(raw_pricing.get("prompt", 0))
        completion_per_tok  = to_float(raw_pricing.get("completion", 0))
        cache_read_per_tok  = to_float(raw_pricing.get("input_cache_read", 0))
        cache_write_per_tok = to_float(raw_pricing.get("input_cache_write", 0))

        return {
            # Human-readable per-1K token prices (common billing unit)
            "input_per_1k_tokens":        round(prompt_per_tok     * 1_000, 8),
            "output_per_1k_tokens":       round(completion_per_tok * 1_000, 8),
            "cache_read_per_1k_tokens":   round(cache_read_per_tok  * 1_000, 8),
            "cache_write_per_1k_tokens":  round(cache_write_per_tok * 1_000, 8),
            # Per-1M token prices (used in most vendor pricing pages)
            "input_per_1m_tokens":        round(prompt_per_tok     * 1_000_000, 6),
            "output_per_1m_tokens":       round(completion_per_tok * 1_000_000, 6),
            "cache_read_per_1m_tokens":   round(cache_read_per_tok  * 1_000_000, 6),
            "cache_write_per_1m_tokens":  round(cache_write_per_tok * 1_000_000, 6),
            # Fields expected by the frontend ModelPricing structure
            "prompt":                     round(prompt_per_tok     * 1_000_000, 6),
            "completion":                 round(completion_per_tok * 1_000_000, 6),
            "input_cache_read":           round(cache_read_per_tok  * 1_000_000, 6),
            # Raw per-token retained for downstream calculations
            "_prompt_per_token":     prompt_per_tok,
            "_completion_per_token": completion_per_tok,
        }

    def infuse_model_pricing_data(self,
        recommendation_output: Any,
    ) -> dict[str, Any]:
        payload = self._to_dict(recommendation_output)
        try:
            model_ids = self.extract_model_ids(payload)
            pricing_information = []

            for model_id in model_ids:
                log.log_info(f"Extracted model_id: {model_id}")
                for model_details in self.model_catalog:
                    if model_id == model_details.get("model_id", ""):
                        log.log_info(f"Found pricing data for model_id: {model_id}")
                        raw_pricing = model_details.get("pricing", {})
                        pricing_information.append({
                            "model_id":     model_id,
                            "name":         model_details.get("name", model_id),
                            "provider":     model_details.get("provider", ""),
                            "pricing":      self._normalise_pricing(raw_pricing),
                            "top_provider": model_details.get("top_provider", {}),
                        })
                        break  # model found, no need to keep scanning

            payload["pricing_information"] = pricing_information

            # Calculate cost per stage using per-stage workload_profile token counts
            for stage in payload.get("stage_recommendations", []):
                if not isinstance(stage, dict):
                    continue

                models = stage.get("models", {})
                if not isinstance(models, dict):
                    continue

                rec_id = models.get("recommended_model_id")
                if not rec_id:
                    continue

                pricing = next((p["pricing"] for p in pricing_information if p["model_id"] == rec_id), None)
                if pricing:
                    stage_wp = stage.get("workload_profile", {})
                    in_toks  = float(stage_wp.get("avg_input_tokens",     0)) + float(stage_wp.get("avg_reasoning_tokens", 0))
                    out_toks = float(stage_wp.get("avg_output_tokens",    0))
                    in_cost  = (in_toks  / 1000.0) * pricing.get("input_per_1k_tokens",  0.0)
                    out_cost = (out_toks / 1000.0) * pricing.get("output_per_1k_tokens", 0.0)
                    stage["estimated_cost_per_request"] = round(in_cost + out_cost, 6)
                    stage["_cost_breakdown"] = {
                        "input_tokens":     int(in_toks),
                        "output_tokens":    int(out_toks),
                        "input_cost":       round(in_cost,  8),
                        "output_cost":      round(out_cost, 8),
                        "input_per_1k":     pricing.get("input_per_1k_tokens",  0.0),
                        "output_per_1k":    pricing.get("output_per_1k_tokens", 0.0),
                    }
                else:
                    stage["estimated_cost_per_request"] = 0.0

            return payload

        except Exception as e:
            log.log_error(f"Error while infusing data - {e}")
            return {"error": str(e)}
    
# if __name__ == "__main__":
#     extractor = ModelDataExtractor()
#     with open("backend/sample_data/sample_recommendation_output.json", "r") as f:
#         sample = json.load(f)
    
#     final_data = extractor.infuse_model_pricing_data(sample)
#     print(json.dumps(final_data, indent=2))
