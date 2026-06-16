import httpx
import json
import pandas as pd

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/"
    "main/model_prices_and_context_window.json"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"

def fetch_litellm() -> dict:
    """
    500+ models. Costs in per-token USD.
    Multiply by 1_000_000 to match your schema.
    """
    r = httpx.get(LITELLM_URL, timeout=30)
    r.raise_for_status()
    return r.json()

# print("LiteLLM Data:",fetch_litellm())
data = fetch_litellm()
fname = "backend/scrapped_data/litelllm.json"
with open(fname, "w") as f:
    f.writelines(json.dumps(data, indent=2))
df = pd.read_json(fname)
# df.sort_values("id", inplace=True)
df.to_excel("backend/scrapped_data/litellm_models.xlsx", index=False)



def fetch_openrouter() -> list[dict]:
    """
    315+ models. Costs in per-token USD strings.
    Covers models LiteLLM misses (smaller open-source).
    """
    r = httpx.get(OPENROUTER_URL, timeout=30)
    r.raise_for_status()
    return r.json()["data"]

# # data = fetch_openrouter()
# fname = "backend/scrapped_data/openrouter.json"
# # with open(fname, "w") as f:
# #     f.writelines(json.dumps(data, indent=2))
# df = pd.read_json(fname)
# df.sort_values("id", inplace=True)
# df.to_excel("backend/scrapped_data/openrouter_models.xlsx", index=False)


def fetch_artificialanalysis(api_key: str) -> list[dict]:
    """
    Quality scores + pricing. Use for benchmark scores block.
    """
    r = httpx.get(
        "https://artificialanalysis.ai/api/v2/data/llms/models",
        headers={"x-api-key": api_key},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()