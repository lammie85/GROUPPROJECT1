"""STAGE 7: THE AGENT

`create_agent` is LangChain's newer, higher-level way to build an agent:
it wires the model, tools, and system prompt into a LangGraph graph.
"""

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

SYSTEM_PROMPT = (
    "You are a friendly Mechanical bot assistant. Always use the "
    "search_policy tool to look up facts before answering. If the answer "
    "isn't in the search results, say you don't know."
)


def build_agent(llm: BaseChatModel, tools: list) -> CompiledStateGraph:
    """MECHANICAL BOT assistant."""
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
