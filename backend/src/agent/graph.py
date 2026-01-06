import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq

from agent.tools_and_schemas import SearchQueryList
from agent.state import (
    OverallState,
    QueryGenerationState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    answer_instructions,
)
from agent.utils import (
    get_research_topic,
)
from local_search import search_in_directory

def local_research(state, config):
    queries = state.get("search_query", [])
    configurable = Configuration.from_runnable_config(config)
    search_dir = getattr(configurable, "local_search_dir", None)
    summaries = search_in_directory(search_dir, queries)
    return {"local_research_result": summaries}
dotenv_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".env")
)
load_dotenv(dotenv_path)

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set")

def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """
    Generate search queries given the user's question.
    Uses Groq (llama-3.3-70b-versatile) as the query-generation LLM.
    Returns a dict containing 'search_query' list.
    """
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    structured_llm = llm.with_structured_output(SearchQueryList)

    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )

    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def finalize_answer(state: OverallState, config: RunnableConfig):
    """
    Finalize the research answer.
    Since web search is disabled in this minimal pipeline,
    the node synthesizes an answer from available material
    (generated queries or any provided summaries).
    """
    configurable = Configuration.from_runnable_config(config)

    
    summaries_list = state.get("web_research_result") or []
    if not summaries_list:
        queries = state.get("search_query", [])
        summaries_text = "\n".join(f"- {q}" for q in queries)
    else:
        summaries_text = "\n---\n\n".join(summaries_list)

    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries=summaries_text,
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    result = llm.invoke(formatted_prompt)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": state.get("sources_gathered", []),
    }


builder = StateGraph(OverallState, config_schema=Configuration)

builder.add_node("generate_query", generate_query)
builder.add_node("finalize_answer", finalize_answer)

builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "finalize_answer")
builder.add_edge("finalize_answer", END)
graph = builder.compile(name="pro-search-agent-minimal")
#Замінено етап веб-дослідження на  Groq з використанням llama-3.3-70b-versatile для забезпечення детермінованого виконання та спрощення архітектури агента. Відокремлення міркувань від пошуку, щоб уникнути використання LLM як транспортного шару для зовнішніх API.Відновлено зручність використання CLI для локального виконання
