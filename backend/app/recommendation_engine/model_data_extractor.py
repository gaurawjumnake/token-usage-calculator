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
        for model in payload.get(
            "single_model_recommendations", []
        ):
            model_ids.add(model["model_id"])
        architecture = payload.get(
            "architecture", {}
        )
        for role in architecture.get("roles", []):
            model_ids.add(
                role["recommended_model_id"]
            )
            model_ids.add(
                role["budget_model_id"]
            )
            model_ids.add(
                role["premium_model_id"]
            )
        return model_ids

    def infuse_model_pricing_data(self,
        recommendation_output: Any,
    ) -> dict[str, Any]:
        payload = self._to_dict(recommendation_output)
        try:
            model_ids = self.extract_model_ids(
                payload
            )
            pricing_information = []
            for model_id in model_ids:
                log.log_info(f"Extracted model_id: {model_id}") 
                for model_details in self.model_catalog:
                    if model_id == model_details.get("model_id", ""):
                        log.log_info(f"Found pricing data for model_id: {model_id}")  
                        pricing_information.append({'model_id':model_id,'pricing':model_details['pricing'], 'top_provider':model_details['top_provider']})

            payload["pricing_information"] = pricing_information

            return payload
        except Exception as e:
            log.log_error(f"Error while infusing data-{e}")
            return {'error':e}
    
# if __name__ == "__main__":
#     extractor = ModelDataExtractor()
#     with open("backend/sample_data/sample_recommendation_output.json", "r") as f:
#         sample = json.load(f)
    
#     final_data = extractor.infuse_model_pricing_data(sample)
#     print(json.dumps(final_data, indent=2))
