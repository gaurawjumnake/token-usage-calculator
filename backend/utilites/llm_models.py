from crewai import LLM
import os
from dotenv import load_dotenv
load_dotenv()

llm_azure = LLM(model="azure/gpt-4o",
          api_key=os.getenv("AZURE_OPENAI_API_KEY"),
          api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
          api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
          temperature=0.5,
          )
 
llm_gemini = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"))

def _check_llm(candidate: LLM) -> bool:
    try:
        candidate.call([{"role": "user", "content": "ping"}])
        return True
    except Exception:
        return False

def _resolve_llm() -> LLM:
    for candidate in [llm_gemini, llm_azure]:
        if _check_llm(candidate):
            print(f"[llm_models] Using model: {candidate.model}")
            return candidate
    raise RuntimeError("No LLM available — check API keys and connectivity.")

llm = _resolve_llm()

# llm = LLM(model="ollama/codellama:7b",
#           base_url="http://localhost:11434")

# Sample code to test llm ############################################




# from openai import OpenAI
 
# endpoint = "https://Nitor-Genai-Foundry.services.ai.azure.com/openai/v1"
# deployment_name = "gpt-4o"

 
# client = OpenAI(
#     base_url=endpoint,
#     api_key=api_key
# )
 
# completion = client.chat.completions.create(
#     model=deployment_name,
#     messages=[
#         {
#             "role": "user",
#             "content": "What is the capital of France?",
#         }
#     ],
# )
 
# print(completion.choices[0].message)


# from langchain_openai import AzureChatOpenAI 
# azure_llm = AzureChatOpenAI(
#     azure_deployment="gpt-4o",
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"), # type: ignore
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version="2024-02-15-preview",
# )