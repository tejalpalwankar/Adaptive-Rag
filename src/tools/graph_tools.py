"""
Tools for graph routing and document grading.
"""

from typing import Literal

from langchain_core.prompts import PromptTemplate

from src.config.settings import Config
from src.llms.openai import llm
from src.models.state import State
from src.models.verification_result import VerificationResult

config = Config()


def routing_tool(state: State) -> Literal["retriever", "general_llm", "web_search"]:
    """
    Route the graph to the appropriate node based on query classification.

    Args:
        state (State): The current state of the graph.

    Returns:
        The next node to execute: "retriever", "general_llm", or "web_search".
    """
    if state["route"] == "index":
        return "retriever"
    elif state["route"] == "general":
        return "general_llm"
    else:
        return "web_search"


# Maximum number of query rewrites before falling back to web search. This caps
# the grade -> rewrite -> retrieve loop so it cannot run indefinitely.
MAX_REWRITES = 2


def doc_tool(state: State) -> Literal["rewrite", "generate", "web_search"]:
    """
    Route after grading, with a bounded rewrite loop.

    - If the retrieved context is relevant ("yes"), generate the answer.
    - If not, rewrite and retry, up to MAX_REWRITES times.
    - Once the rewrite budget is exhausted, fall back to web search so the user
      still gets an answer instead of looping forever.

    Args:
        state (State): The current state of the graph.

    Returns:
        The next node: "generate", "rewrite", or "web_search".
    """
    score = state["binary_score"]
    retry_count = state.get("retry_count", 0)
    print(f"[doc_tool] Routing based on score: {score}, retries: {retry_count}")

    if score == "yes":
        return "generate"
    if retry_count < MAX_REWRITES:
        return "rewrite"
    print("[doc_tool] Rewrite budget exhausted; falling back to web search.")
    return "web_search"


# Maximum number of answer generations (initial + regenerations) before the
# graph ends regardless of the faithfulness check. Caps the
# generate -> verify -> generate loop so it cannot run indefinitely.
MAX_GENERATIONS = 2


def verify_answer(state: State) -> Literal["__end__", "generate"]:
    """
    Verify whether the final answer is faithful to the retrieved context.

    Runs after ``generate``. If the answer is supported by the context, the
    graph ends; otherwise it regenerates, up to MAX_GENERATIONS times.

    Args:
        state (State): The current state of the graph.

    Returns:
        "__end__" if the answer is faithful or the retry budget is exhausted,
        otherwise "generate" to try again.
    """
    # General-knowledge answers have no retrieved context to verify against.
    if state.get("route") == "general":
        return "__end__"

    context = state.get("context")
    if not context:
        # Nothing to verify against; accept the answer.
        return "__end__"

    # Stop regenerating once the generation budget is spent.
    count = state.get("verify_count", 0) or 0
    if count >= MAX_GENERATIONS:
        print("[verify_answer] Generation budget exhausted; ending.")
        return "__end__"

    question = state["latest_query"]
    final_answer = state["messages"][-1].content

    verify_prompt = PromptTemplate(
        template=config.prompt("verify_prompt"),
        input_variables=["question", "context", "final_answer"]
    )
    llm_with_verification = llm.with_structured_output(VerificationResult)

    verify_chain = verify_prompt | llm_with_verification

    result = verify_chain.invoke({
        "question": question,
        "context": context,
        "final_answer": final_answer
    })

    if result.faithful:
        return "__end__"
    else:
        print("Generating again as answer is not faithful.")
        return "generate"
