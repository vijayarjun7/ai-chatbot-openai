"""
LangChain chat logic
  - GPT-4o streaming via OpenAI API
  - ConversationBufferWindowMemory  -> short-term (last 8 turns in prompt)
  - ChromaDB VectorStoreRetrieverMemory -> long-term (semantically relevant past context)
  - CombinedMemory merges both into a single prompt
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferWindowMemory,
    VectorStoreRetrieverMemory,
    CombinedMemory,
)
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

CHROMA_DIR = "./chroma_db"

# Streaming callback (token-by-token into a Streamlit container)
class StreamHandler(BaseCallbackHandler):
    """Streams LLM tokens live into a Streamlit markdown container."""

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, *args, **kwargs):
        self.container.markdown(self.text)


# Prompt template (uses both memory keys)
PROMPT = PromptTemplate(
    input_variables=["chat_history", "relevant_history", "input"],
    template=(
        "You are a helpful, knowledgeable AI assistant built by Vijaya Kumari "
        "using LangChain and OpenAI GPT-4o.\n\n"
        "=== Relevant past context (from long-term memory) ===\n"
        "{relevant_history}\n\n"
        "=== Recent conversation ===\n"
        "{chat_history}\n\n"
        "Human: {input}\n"
        "Assistant:"
    ),
)


def _build_memory(embeddings: OpenAIEmbeddings):
    """Create CombinedMemory: short-term buffer + ChromaDB long-term store."""
    buffer_memory = ConversationBufferWindowMemory(
        k=8,
        memory_key="chat_history",
        input_key="input",
    )
    vectorstore = Chroma(
        collection_name="chat_memory",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    vector_memory = VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="relevant_history",
        input_key="input",
    )
    return CombinedMemory(memories=[buffer_memory, vector_memory])


def get_chain() -> ConversationChain:
    """Build and return the streaming ConversationChain."""
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )
    memory = _build_memory(embeddings)
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=PROMPT,
        verbose=False,
    )
    return chain


def get_response(chain: ConversationChain, user_input: str, stream_container=None) -> str:
    """Send a message and return the full response."""
    callbacks = [StreamHandler(stream_container)] if stream_container else []
    result = chain.predict(input=user_input, callbacks=callbacks)
    return result.strip()


def clear_memory(chain: ConversationChain) -> None:
    """Wipe the short-term buffer memory (ChromaDB history is preserved)."""
    for mem in chain.memory.memories:
        if hasattr(mem, "clear"):
            mem.clear()
