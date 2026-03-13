"""
LangGraph workflow definition for the terminal chatbot.
State machine: receive_question → generate_answer → return_answer.
"""

from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from chatbot import get_llm, invoke_llm


# ---------------------------------------------------------------------------
# State definition
# LangGraph passes this state between nodes; each node can read and update it.
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    """State for the chatbot graph: current question, generated answer, and message history."""
    question: str
    answer: str
    # Annotated with add_messages so LangGraph can append new messages to history
    messages: Annotated[list[BaseMessage], add_messages]


def receive_question(state: ChatState) -> ChatState:
    """
    Node 1: Accepts user input.
    The question is already in state (set by the caller before invoking the graph).
    This node simply forwards the state so the flow explicitly includes 'receive_question'.
    """
    return state


def generate_answer(state: ChatState) -> ChatState:
    """
    Node 2: Uses the LangChain LLM to generate an answer from the current question
    and conversation history. Updates state with the new answer and appends
    human/assistant messages to history for future turns.
    """
    question = state["question"]
    history: list[BaseMessage] = state.get("messages") or []
    llm = get_llm()
    answer = invoke_llm(question, history, llm=llm)
    # Append this turn to history so the next invocation has context
    new_messages = [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]
    return {
        **state,
        "answer": answer,
        "messages": new_messages,
    }


def return_answer(state: ChatState) -> ChatState:
    """
    Node 3: Prints the answer to the terminal and returns the state.
    The actual print happens here so the graph is responsible for output.
    """
    answer = state.get("answer") or ""
    print(f"\nBot: {answer}\n")
    return state


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph workflow.
    Flow: START → receive_question → generate_answer → return_answer → END.
    """
    # Create the graph with our state type
    graph = StateGraph(ChatState)

    # Add the three nodes
    graph.add_node("receive_question", receive_question)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("return_answer", return_answer)

    # Define the flow
    graph.add_edge(START, "receive_question")
    graph.add_edge("receive_question", "generate_answer")
    graph.add_edge("generate_answer", "return_answer")
    graph.add_edge("return_answer", END)

    return graph.compile()
