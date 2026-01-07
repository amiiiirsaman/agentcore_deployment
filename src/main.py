# src/main.py
import os
import json
from pathlib import Path
from datetime import datetime, UTC

from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.utilities import SerpAPIWrapper


SYSTEM_PROMPT = """You are a helpful research assistant.
You are methodical and always think step-by-step.

Current date (UTC): {current_date}

Goal:
- Answer the user's query accurately, preferring recent, trustworthy sources.
- First, use web search to gather relevant, up-to-date information (when available).
- Then, synthesize the information into a clear, concise summary in markdown.
- Write the summary to a file in the /tmp directory.
- Finally, confirm that the file has been written and state its location.

When thinking, you can reason step by step, but the final answer should be a clean summary plus a short confirmation about the file.
"""


def web_search(query: str, num_results: int = 5):
    """SerpAPI web search via LangChain wrapper; returns a list of result dicts."""
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        # Degrade gracefully if the key is not configured in the environment.
        return []

    serp = SerpAPIWrapper()
    results = serp.results(query)

    if isinstance(results, dict) and "organic_results" in results:
        organic = results["organic_results"] or []
        return organic[:num_results]
    if isinstance(results, list):
        return results[:num_results]
    return []


def build_llm() -> ChatBedrock:
    """Nova Pro chat model via Bedrock."""
    return ChatBedrock(
        model_id="amazon.nova-pro-v1:0",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        temperature=0.2,
    )


def write_tmp_file(filename: str, content: str) -> str:
    """Write a file under /tmp and return its absolute path."""
    base = Path("/tmp")
    base.mkdir(parents=True, exist_ok=True)
    path = (base / filename).resolve()
    path.write_text(content, encoding="utf-8")
    return str(path)


def run_agent(query: str) -> dict:
    """Core agent logic used by HTTP server and Lambda handler."""
    current_date = datetime.now(UTC).strftime("%Y-%m-%d")

    # 1) Web search (may be empty if SERPAPI_API_KEY is not set)
    results = web_search(query, num_results=5)
    snippets = []
    for i, r in enumerate(results, start=1):
        title = r.get("title") or ""
        snippet = r.get("snippet") or ""
        link = r.get("link") or r.get("url", "")
        snippets.append(f"{i}. {title}\n   {snippet}\n   Source: {link}")
    search_context = "\n\n".join(snippets) if snippets else "No search results found."

    # 2) Ask Nova Pro for summary
    llm = build_llm()
    system_text = SYSTEM_PROMPT.format(current_date=current_date)
    messages = [
        SystemMessage(content=system_text),
        HumanMessage(
            content=(
                f"User question: {query}\n\n"
                f"Here are web search results:\n\n{search_context}\n\n"
                "Please:\n"
                "1. Synthesize the key points into a clear markdown summary.\n"
                "2. At the end, add a short note confirming that this will be written "
                "to a file in /tmp.\n"
            )
        ),
    ]
    llm_response = llm.invoke(messages)
    summary_markdown = llm_response.content

    # 3) Write to /tmp
    file_path = write_tmp_file("summary.txt", summary_markdown)

    # 4) Final response body (Python dict)
    return {
        "query": query,
        "summary": summary_markdown,
        "file_path": file_path,
    }


# --- Lambda handler (kept for compatibility) ---

def handler(event, context):
    """
    Lambda-style handler. Expects:
      event = {"query": "..."}
    """
    query = event.get("query") if isinstance(event, dict) else None
    if not query:
        return {"statusCode": 400, "body": "Query not provided"}

    try:
        result = run_agent(query)  # dict
        return {"statusCode": 200, "body": json.dumps(result)}
    except Exception as e:
        return {"statusCode": 500, "body": f"Error: {e}"}
