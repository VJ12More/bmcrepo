"""
LangChain LLM logic for the terminal chatbot.
Uses OpenAI Chat model (gpt-4o-mini) with configurable temperature.
"""

import os
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


# Default model and temperature (OpenAI)
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatOpenAI:
    """
    Create and return a LangChain OpenAI Chat model instance.
    API key is read from the OPENAI_API_KEY environment variable (via .env).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def invoke_llm(
    question: str,
    chat_history: List[BaseMessage],
    llm: ChatOpenAI | None = None,
) -> str:
    """
    Send the user question (and optional history) to the LLM and return the answer.
    Builds the message list from history + new question, then invokes the model.
    """
    if llm is None:
        llm = get_llm()

    messages: List[BaseMessage] = list(chat_history)
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)
