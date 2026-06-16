from typing import Any

SCRAPPER_OUTPUT_SCHEMA: dict[str, Any] = {
    "provider": "",
    "pricing_last_updated": "",
    "pricing_confidence": "",          # high | medium | low | estimated
    "models": [
        {
            "model_id": "",            # stable slug, e.g. "openai/gpt-5"
            "model_name": "",
            "category": "",            # chat | embedding | image | audio | reasoning
            "is_deprecated": False,
            "release_date": "",

            # Pricing 
            "input_cost_per_million": None,
            "output_cost_per_million": None,
            "cached_input_cost_per_million": None,
            "batch_input_cost_per_million": None,
            "batch_output_cost_per_million": None,

            # Context 
            "context_window": None,
            "max_output_tokens": None,
            "supports_context_caching": False,

            # Capabilities 
            "supports_reasoning": False,
            "supports_function_calling": False,
            "supports_structured_output": False,
            "supports_json_mode": False,
            "supports_vision": False,
            "supports_audio_input": False,
            "supports_audio_output": False,
            "supports_streaming": False,
            "supports_batch": False,
            "supports_fine_tuning": False,

            # Guidance 
            "pricing_tier": "",        # frontier | advanced | standard | economy
            "recommended_for": [],
            "avoid_for": [],
            "limitations": [],
        }
    ],
    "additional_costs": {
        "embedding": {},
        "image_generation": {},
        "audio": {},
        "search": {},
        "storage": {},
        "fine_tuning": {},
    },
    "rate_limits": {},
    "enterprise_options": {},
    "notes": [],
    "missing_information": [],
}

benchmarking_output_schema = {
  "provider": "",
  "last_updated": "",
  "models": [
    {
      "model_name": "",

      "benchmark_sources": [
        {
          "source_name": "",
          "source_url": "",
          "date": ""
        }
      ],

      "raw_benchmarks": {
        "gpqa": None,
        "humanitys_last_exam": None,
        "arc_agi": None,
        "mmlu": None,
        "mmlu_pro": None,

        "swe_bench_verified": None,
        "swe_bench": None,
        "humaneval": None,
        "livecodebench": None,

        "tau_bench": None,
        "agentbench": None,
        "gaia": None,
        "browsecomp": None,

        "mmmu": None,
        "mmmu_pro": None,
        "mathvista": None,
        "chartqa": None,
        "docvqa": None,

        "aime": None,
        "math": None,
        "gsm8k": None,

        "ifeval": None,

        "longbench": None,
        "infinitebench": None
      },

      "normalized_scores": {
        "reasoning_score": None,
        "coding_score": None,
        "agentic_score": None,
        "multimodal_score": None,
        "math_score": None,
        "instruction_following_score": None,
        "long_context_score": None,
        "overall_quality_score": None
      },

      "recommended_use_cases": [],

      "strengths": [],

      "limitations": [],

      "benchmark_confidence": "high"
    }
  ]
}