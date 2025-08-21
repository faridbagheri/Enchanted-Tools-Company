"""
CLI Chatbot with RAG (Hospital Dataset)
- Context-aware assistant (Hospital / Concierge roles).
- Retrieval-Augmented Generation (RAG) with hospital JSON.
- Commands: /reset /exit /help
"""

import os
import sys
import argparse
from typing import List, Dict

from dotenv import load_dotenv

# ---- LangChain / RAG imports ----
import json
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ---------------- Config ----------------
load_dotenv(override=True)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.4"))
MAX_TURNS_MEMORY = 8
DB_NAME = "hospital_vector_db"

# ---------------- Role Prompts ----------------
HOSPITAL_SYSTEM_PROMPT = """You are Mirokaï, a courteous hospital assistant robot.
Goals:
- Greet warmly, keep responses concise and reassuring.
- Clarify requests (who/what/where/when) before acting.
- NEVER give medical advice; defer to medical staff for diagnoses/treatments.
- You can provide directions, general info, hospital facility info, and polite small talk.
Tone: calm, supportive, professional, accessible to non-experts."""

CONCIERGE_SYSTEM_PROMPT = """You are Mirokaï, a friendly concierge robot in a hotel lobby.
Goals:
- Greet guests, answer questions, give directions and recommendations.
- Ask short clarifying questions before giving long answers.
- Be specific when possible (times, locations, steps).
Tone: upbeat, helpful, slightly playful but always professional."""

ROLE_TO_PROMPT = {
    "hospital": HOSPITAL_SYSTEM_PROMPT,
    "concierge": CONCIERGE_SYSTEM_PROMPT,
}


# -------------- Build RAG Pipeline --------------
def build_rag_chain(json_file: str):
    # Load JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    # Flatten hospital JSON into documents
    def flatten_dict(d, prefix=""):
        text_parts = []
        if isinstance(d, dict):
            for k, v in d.items():
                text_parts.extend(flatten_dict(v, f"{prefix}{k} "))
        elif isinstance(d, list):
            for item in d:
                text_parts.extend(flatten_dict(item, prefix))
        else:
            text_parts.append(f"{prefix}{d}")
        return text_parts

    flat_text = " ".join(flatten_dict(data))
    documents.append(Document(page_content=flat_text, metadata={"source": "hospital_dataset"}))

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    # Fresh vector DB
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_NAME)

    llm = ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

    return chain


# -------------- Chatbot Wrapper --------------
class Chatbot:
    def __init__(self, role: str, rag_chain, verbose: bool = False):
        self.role = role
        self.system_prompt = ROLE_TO_PROMPT[role]
        self.verbose = verbose
        self.chain = rag_chain

    def ask(self, user_text: str) -> str:
        query = f"{self.system_prompt}\nUser: {user_text}"
        result = self.chain.invoke({"question": query})
        return result["answer"]

    def reset(self):
        # Reset conversation memory
        self.chain.memory.clear()


# -------------- CLI Loop --------------
def print_header(role: str):
    print("=" * 64)
    print(f"Mirokaï CLI Chatbot  •  Role: {role}")
    print("Commands: /reset  /exit  /help")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["hospital", "concierge"], default="hospital")
    parser.add_argument("--data", default="california_general_hospital.json", help="Path to hospital JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rag_chain = build_rag_chain(args.data)
    bot = Chatbot(role=args.role, rag_chain=rag_chain, verbose=args.verbose)

    print_header(args.role)

    greeting = {
        "hospital": "Hello, I'm Mirokaï. How can I help you today? (I can give directions and hospital info; I don't provide medical advice.)",
        "concierge": "Welcome! I'm Mirokaï at your service. What can I help you with—directions, recommendations, or check-in info?"
    }[args.role]
    print(f"Mirokaï: {greeting}")

    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nMirokaï: Goodbye! 👋")
            break

        if not user_text:
            continue
        if user_text.lower() in ("/exit", "/quit"):
            print("Mirokaï: Goodbye! 👋")
            break
        if user_text.lower() == "/help":
            print("Mirokaï: Commands → /reset (clear context), /exit (quit), /help (this help)")
            continue
        if user_text.lower() == "/reset":
            bot.reset()
            print("Mirokaï: Context cleared. How can I help now?")
            continue

        reply = bot.ask(user_text)
        print(f"Mirokaï: {reply}")


if __name__ == "__main__":
    main()
