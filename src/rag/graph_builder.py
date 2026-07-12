"""
Graph builder module for the adaptive RAG system.
"""

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph

from src.rag.retriever_setup import get_retriever
from src.config.settings import Config
from src.llms.openai import llm
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.tools.graph_tools import routing_tool, doc_tool, verify_answer

config = Config()


# Node implementations
def query_classifier(state: State):
    """
    Classify the query to determine if it's related to indexed documents.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with route and latest_query.
    """
    question = state["messages"][-1].content
    retriever = get_retriever(state.get("session_id", "default"))
    context = retriever.invoke(question)
    print("docs received from Qdrant")
    print(context)

    llm_with_structured_output = llm.with_structured_output(RouteIdentifier)
    classify_prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=["question", "context"]
    )
    chain = classify_prompt | llm_with_structured_output
    result = chain.invoke({"question": question, "context": context})
    print("result received is in query classifier")
    print(result.route)

    # Store the retrieved context so retriever_node can reuse it on the first
    # pass instead of retrieving the same query a second time.
    return {
        "messages": state["messages"],
        "route": result.route,
        "latest_query": question,
        "context": context,
    }


def general_llm(state: State):
    """
    Fetch general common knowledge result from the LLM.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages from LLM.
    """
    result = llm.invoke(state["messages"])
    print("inside general llm")
    print(result)
    return {"messages": result}


def retriever_node(state: State):
    """
    Retrieve context for the current query from the session's vector store.

    On the first pass the context retrieved during classification is reused, so
    the same query is not embedded and retrieved twice. After a rewrite the
    query has changed, so a fresh retrieval is performed for it.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages and context with the retrieved content.
    """
    retry_count = state.get("retry_count", 0) or 0
    reused_context = state.get("context")

    if retry_count == 0 and reused_context:
        # Reuse the context already retrieved by query_classifier.
        content = reused_context
    else:
        # Query was rewritten (or no prior context); retrieve for it now.
        retriever = get_retriever(state.get("session_id", "default"))
        content = retriever.invoke(state["latest_query"])

    new_message = AIMessage(content=content)

    return {
        "messages": [new_message],
        "context": content,
    }


def grade(state: State):
    """
    Grade the results retrieved from vector stores.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with binary_score.
    """
    grading_prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=["question", "context"]
    )
    context = state["messages"][-1].content
    question = state["latest_query"]

    llm_with_grade = llm.with_structured_output(Grade)

    chain_graded = grading_prompt | llm_with_grade
    result = chain_graded.invoke({"question": question, "context": context})

    print(result)
    return {"messages": state["messages"], "binary_score": result.binary_score}


def rewrite_query(state: State):
    """
    Rewrite the query to get better retrieval results.

    Args:
        state (State): State of the question.

    Returns:
        dict: Updated latest_query.
    """
    query = state["latest_query"]
    rewrite_prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=["query"]
    )
    chain = rewrite_prompt | llm
    result = chain.invoke({"query": query})
    print(result)

    return {
        "latest_query": result.content,
        "retry_count": state.get("retry_count", 0) + 1
    }


def generate(state: State):
    """
    Generate the final answer for the user.

    Args:
        state (State): State of the question.

    Returns:
        dict: Generated response.
    """
    count = state.get("verify_count", 0) or 0
    # On the first generation the last message is the retrieved/searched
    # context. On a regeneration it is the previous answer, so reuse the
    # context captured on the first pass.
    if count == 0:
        context = state["messages"][-1].content
    else:
        context = state.get("context") or state["messages"][-1].content

    generate_prompt = PromptTemplate(
        template=config.prompt("generate_prompt"),
        input_variables=["context"]
    )

    generate_chain = generate_prompt | llm
    result = generate_chain.invoke({"context": context})

    return {
        "messages": [{"role": "assistant", "content": result.content}],
        "context": context,
        "verify_count": count + 1
    }


def web_search(state: State):
    """
    Search the web for the rewritten query.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Search results as messages.
    """
    # Initialize the Tavily tool
    search_tool = TavilySearchResults()

    # Search a query
    result = search_tool.invoke(state["latest_query"])

    contents = [item["content"] for item in result if "content" in item]
    print(contents)

    return {
        "messages": [{"role": "assistant", "content": "\n\n".join(contents)}]
    }


# Build the graph
graph = StateGraph(State)

graph.add_node("query_analysis", query_classifier)
graph.add_node("retriever", retriever_node)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("general_llm", general_llm)

graph.add_edge(START, "query_analysis")
graph.add_edge("web_search", "generate")
graph.add_edge("retriever", "grade")
graph.add_edge("rewrite", "retriever")
graph.add_conditional_edges("query_analysis", routing_tool)
graph.add_conditional_edges("grade", doc_tool)
# After generating, verify the answer is faithful to the context; regenerate if
# not (bounded by MAX_GENERATIONS inside verify_answer), otherwise end.
graph.add_conditional_edges(
    "generate",
    verify_answer,
    {"__end__": END, "generate": "generate"}
)
graph.add_edge("general_llm", END)

builder = graph.compile()

