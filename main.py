"""
Terminal-based AI chatbot - entry point.
Run with: python main.py
Type 'exit' to quit. Uses LangGraph for the conversation workflow and OpenAI via LangChain.
"""

import os
import sys

from dotenv import load_dotenv

from graph import build_graph
from graph import ChatState


def load_env() -> None:
    """Load environment variables from .env file. Required for OPENAI_API_KEY."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Create a .env file with OPENAI_API_KEY=your_key")
        sys.exit(1)


def run_chat_loop() -> None:
    """
    Main loop: read user input from the terminal, run the LangGraph workflow,
    and print the bot's response. Keeps conversation history for context.
    """
    graph = build_graph()
    # Conversation history (list of HumanMessage / AIMessage) for context
    messages: list = []


    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        # Initial state for this turn: current question and accumulated history
        initial_state: ChatState = {
            "question": user_input,
            "answer": "",
            "messages": messages,
        }

        try:
            # Run the LangGraph workflow; return_answer node prints the bot reply
            result = graph.invoke(initial_state)
            # Update history so the next turn has context
            messages = result.get("messages") or messages
        except Exception as e:
            print(f"\nBot: Error calling the model: {e}\n")
            # Optionally log e for debugging


def main() -> None:
    load_env()
    run_chat_loop()


if __name__ == "__main__":
    main()
