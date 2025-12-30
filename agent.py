# agent.py

from strands import Agent
from strands.models.ollama import OllamaModel

from tools.retrieval_tool import retrieve_docs


# Configure your local Ollama model
# Make sure `ollama run llama3.1 "hi"` works in a normal terminal
model = OllamaModel(
    model_id="llama3.1",              # must match an installed ollama model
    host="http://localhost:11434",    # default ollama host
)


SYSTEM_PROMPT = """
You are a Retrieval-Augmented Generation (RAG) assistant running inside a Strands agent.

You have access to a tool called `retrieve_docs` that searches over a set of
indexed documents and returns relevant passages.

Your behavior:

- For each user question, first think if you need external context.
- If the answer might depend on the documents, CALL `retrieve_docs` with
  an appropriate search query (which can be the user question or a refined version).
- Use ONLY the returned passages as your knowledge source when forming the answer.
- If the context is not enough to answer reliably, clearly say you don't know.
- At the end of your response, ALWAYS include:

  1. A short "Tool usage summary" section explaining:
     - Did you call `retrieve_docs`?
     - If yes, what you searched for and how many passages you used.
  2. A "Citations" section that references the numbered passages like [1], [2].

Example format:

Answer:
<your answer here>

Tool usage summary:
- Called retrieve_docs with query: "<query>"
- Retrieved N passages and used them to construct the answer.

Citations:
- [1] Source: ...
- [2] Source: ...
"""


agent = Agent(
    model=model,
    tools=[retrieve_docs],
    system_prompt=SYSTEM_PROMPT,
)


def chat_loop():
    print("🧠 Strands RAG Agent (Ollama + Chroma)")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("❓ Question: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break

        if not question:
            continue

        print("\n🤖 Thinking...\n")
        try:
            # Strands will decide when to call retrieve_docs
            answer = agent(question)
            print(answer)
        except Exception as e:
            print(f"❌ Error during agent execution: {e}")


if __name__ == "__main__":
    chat_loop()
