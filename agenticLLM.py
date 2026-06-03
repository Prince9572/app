import ast
import os
import subprocess
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

def _load_local_knowledge() -> str:
    """Load a tiny local knowledge base, creating it on first run."""
    filename = "knowledge_base.txt"
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("Foundational Knowledge: AI is rapidly transforming modern industries through automation and predictive analytics.")

    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

# --- 1. THE CUSTOM KNOWLEDGE TOOL (ZERO PIP INSTALLS) ---
@tool
def read_local_knowledge(query: str) -> str:
    """Use this tool to search the local knowledge base for foundational information."""
    return _load_local_knowledge()

def _extract_key_findings(raw_search_results: str):
    """Turn DuckDuckGo tool output into short, readable bullet points."""
    try:
        parsed = ast.literal_eval(raw_search_results)
    except (SyntaxError, ValueError):
        parsed = []

    findings = []
    if isinstance(parsed, list):
        for item in parsed[:5]:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "Untitled result")
            snippet = item.get("snippet", "No summary available.")
            link = item.get("link")
            line = f"{title}: {snippet}"
            if link:
                line += f" ({link})"
            findings.append(line)

    if findings:
        return findings

    fallback = raw_search_results.strip() or "No recent web results were available."
    return [fallback[:500]]

def _build_fallback_report(topic: str, raw_search_results: str, llm_error: str) -> str:
    """Create a deterministic report when Ollama or the configured model is unavailable."""
    local_knowledge = _load_local_knowledge()
    key_findings = _extract_key_findings(raw_search_results)
    web_search_available = not raw_search_results.startswith("Web search unavailable:")

    findings_text = "\n".join(f"- {item}" for item in key_findings)
    challenges = [
        "The local Ollama model could not be used for synthesis during this run.",
        "The topic is evolving quickly, so conclusions should be refreshed with a working model for richer analysis.",
    ]
    if not web_search_available:
        challenges.append(raw_search_results)

    challenges_text = "\n".join(f"- {item}" for item in challenges)

    return f"""=========================================
COVER PAGE
Title: Fallback Research Brief on {topic}
Topic: {topic}
Author: Autonomous Research Agent
=========================================

# Fallback Research Brief on {topic}

## Introduction
This report was generated with the local knowledge base and direct web-search results because the Ollama model was unavailable during execution.

## Key Findings
{findings_text}

## Challenges
{challenges_text}

## Future Scope
Future runs can generate a more comprehensive synthesis once a local Ollama model is available. The foundational context remains: {local_knowledge}

## Conclusion
This run completed with a fallback path so the project could still produce an output file. Ollama reported: {llm_error}
"""

def _ollama_model_available(model_name: str) -> bool:
    """Check whether the requested Ollama model is already installed locally."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    return model_name in result.stdout

def run_local_research_agent(topic: str):
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")

    # 2. Define Tools (DuckDuckGo + Our Custom No-Install Tool)
    search = DuckDuckGoSearchResults()
    tools = [search, read_local_knowledge]

    # 3. Strict System Prompt
    system_instructions = f"""
    You are an autonomous research agent. Your task is to research: '{topic}'
    
    Use your tools to gather comprehensive information. 
    You must use the web search tool for recent data and the local knowledge tool for foundational knowledge.
    
    Once you have gathered enough information, you MUST format your final response EXACTLY according to the structure below:
    
    =========================================
    COVER PAGE
    Title: [Insert Catchy Title Here]
    Topic: {topic}
    Author: Autonomous Research Agent
    =========================================
    
    # [Insert Title]
    
    ## Introduction
    [Write a comprehensive introduction to the topic]
    
    ## Key Findings
    [Extract and bullet point the most critical data and insights]
    
    ## Challenges
    [Detail the current obstacles or limitations]
    
    ## Future Scope
    [Discuss what the future holds for this topic]
    
    ## Conclusion
    [Provide a final summary wrapping up the report]
    """

    print(f"Starting local research on: '{topic}' using model '{model_name}'...\n")
    if not _ollama_model_available(model_name):
        reason = f"Ollama model '{model_name}' is not installed locally."
        print(f"{reason} Falling back to direct report generation.\n")
        try:
            raw_search_results = search.invoke(topic)
        except Exception as search_error:
            raw_search_results = f"Web search unavailable: {search_error}"
        return _build_fallback_report(topic, raw_search_results, reason)

    try:
        llm = ChatOllama(model=model_name, temperature=0, request_timeout=120)
        agent = create_react_agent(llm, tools=tools)
        response = agent.invoke({"messages": [HumanMessage(content=system_instructions)]})
        return response["messages"][-1].content
    except Exception as e:
        print(f"Ollama agent unavailable. Falling back to direct report generation.\nReason: {e}\n")
        try:
            raw_search_results = search.invoke(topic)
        except Exception as search_error:
            raw_search_results = f"Web search unavailable: {search_error}"
        return _build_fallback_report(topic, raw_search_results, str(e))

# --- Execution ---
if __name__ == "__main__":
    user_topic = input("Enter a research topic (e.g., 'Impact of AI in Healthcare'): ")
    
    if not user_topic:
        user_topic = "Impact of AI in Healthcare" 
        
    final_report = run_local_research_agent(user_topic)
    
    print("\n\n" + "="*50 + " FINAL REPORT " + "="*50 + "\n")
    print(final_report)
    
    safe_filename = user_topic.replace(" ", "_").replace("/", "_") + "_local_output.txt"
    with open(safe_filename, "w", encoding="utf-8") as file:
        file.write(final_report)
        
    print(f"\n✅ Report saved as '{safe_filename}'.")
