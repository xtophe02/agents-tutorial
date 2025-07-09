from agents import Agent, function_tool, ModelSettings
from tavily import TavilyClient
import os

from dotenv import load_dotenv

load_dotenv(override=True)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succintly, no need to have complete sentences or good "
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary itself."
)


@function_tool
def tavily_search(query: str) -> str:
    """Search the web using Tavily and return a summary string."""
    result = tavily_client.search(query=query)
    # You may want to process result['results'] or result['answer'] depending on Tavily's response
    return result.get("answer", "")


search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    tools=[tavily_search],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
