import json
from typing import Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Crew, Process, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from backend.app.scrapper.prompt import PromptTemplates
from backend.app.scrapper.output_schema import SCRAPPER_OUTPUT_SCHEMA
from backend.utilites.llm_models import llm

class UrlList(BaseModel):
    urls: list[str] = Field(
        description="Ranked list of URLs, most relevant first. Max 5 URLs."
    )

class ScrapperAgent:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._search_tool = SerperDevTool()
        self._scrape_tool = ScrapeWebsiteTool()
        self._prompts = PromptTemplates()
        self.agents = {}
        self.tasks = {}

    def planner(self) -> None:
        agent = Agent(
            role="AI Pricing Research Planner",
            goal=(
                "Find and return the 5 most relevant official URLs containing "
                "pricing and model capability data for {provider_name}. "
                "Return nothing but a JSON object with a 'urls' key."
            ),
            backstory=(
                "You are a precise web researcher. You only return official "
                "provider URLs — never third-party blogs or comparison sites. "
                "You always return a clean JSON object, never prose."
            ),
            tools=[self._search_tool],
            verbose=self.verbose,
            llm=llm,
            # max_iter=5,        
        )

        task = Task(
            description=self._prompts.PLANNER_PROMPT,
            expected_output=(
                'A JSON object with key "urls" containing an array of up to 5 '
                "official URL strings. Example: "
                '{"urls": ["https://openai.com/api/pricing"]}'
            ),
            agent=agent,
            output_pydantic=UrlList,    
        )

        self.agents["planner_agent"] = agent
        self.tasks["planner_task"] = task

    def extractor(self) -> None:
        agent = Agent(
            role="AI Pricing Intelligence Analyst",
            goal=(
                "Scrape the URLs from the planner and extract all pricing and "
                "capability data for {provider_name}. Output valid JSON only."
            ),
            backstory=(
                "You are a meticulous analyst who has read thousands of API pricing "
                "pages. You never guess — if a value isn't on the page, you mark it "
                "null and flag it. You always normalise costs to USD per 1M tokens. "
                "You always return pure JSON, never markdown or prose."
            ),
            tools=[self._scrape_tool],   # search tool removed: URLs already known
            verbose=self.verbose,
            llm=llm,
            # max_iter=10,
        )

        task = Task(
            description=self._prompts.EXTRACTOR_PROMPT.format(
                provider_name="{provider_name}",          # stays as runtime var
                output_schema=json.dumps(SCRAPPER_OUTPUT_SCHEMA, indent=2),
            ),
            expected_output=(
                "Valid JSON matching the output schema exactly. "
                "All costs in USD per 1M tokens. No markdown fences."
            ),
            agent=agent,
            context=[self.tasks["planner_task"]],
        )

        self.agents["extractor_agent"] = agent
        self.tasks["extractor_task"] = task

    def validator(self) -> None:          
        agent = Agent(
            role="Pricing Data Validator",
            goal=(
                "Validate the extracted pricing JSON against the schema rules. "
                "Annotate missing fields. Return the same JSON, corrected only "
                "where schema compliance is violated."
            ),
            backstory=(
                "You are a data quality engineer who enforces strict schema rules. "
                "You catch unit mismatches, missing required fields, and obvious "
                "pricing anomalies. You never invent values."
            ),
            tools=[],              
            verbose=self.verbose,
            llm=llm,
            max_iter=2,
        )

        task = Task(
            description=self._prompts.VALIDATOR_PROMPT,
            expected_output=(
                "The same JSON structure as the extractor output, with "
                "missing_information[] and notes[] arrays fully populated."
            ),
            agent=agent,
            context=[self.tasks["extractor_task"]],
        )

        self.agents["validator_agent"] = agent
        self.tasks["validator_task"] = task

    def run(
        self,
        provider_name: str,
        seed_urls: list[str] | None = None,
    ) -> dict[str, Any]:

        seed_urls = seed_urls or []

        self.planner()
        self.extractor()
        self.validator()


        crew = Crew(
            agents=[v for k,v in self.agents.items()],
            tasks=list(self.tasks.values()),
            process=Process.sequential,
            verbose=self.verbose,
        )

        raw_result = crew.kickoff(
            inputs={
                "provider_name": provider_name,
                "seed_urls": json.dumps(seed_urls),
            }
        )

        return self._parse_output(raw_result)

    @staticmethod
    def _parse_output(raw: Any) -> dict[str, Any]:
        """parse crew output to a dict. Handles both string JSON and already-parsed objects."""

        if isinstance(raw, dict):
            return raw

        text = str(raw).strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_output": text, "parse_error": True}



if __name__ == "__main__":
    scrapper = ScrapperAgent(verbose=True)

    result = scrapper.run(
        provider_name="Anthropic",
        seed_urls=["https://www.anthropic.com/pricing"],
    )
    with open("backend/scrapped_data/anthropic.json", "w") as f:       
        f.writelines(json.dumps(result, indent=2))