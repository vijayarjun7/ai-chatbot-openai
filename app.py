import streamlit as st
from utils.chat import get_response, get_chain, clear_memory

st.set_page_config(
    page_title="AI Chatbot - GPT-4o + LangChain",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 AI Chatbot")
st.caption("GPT-4o · LangChain · ChromaDB long-term memory · Built by Vijaya Kumari")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    with st.spinner("Loading model & memory..."):
        st.session_state.chain = get_chain()

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Model:** GPT-4o (streaming)")
    st.markdown("**Short-term:** Last 8 turns")
    st.markdown("**Long-term:** ChromaDB (persists across sessions)")
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        clear_memory(st.session_state.chain)
        st.success("Conversation cleared!")
        st.rerun()
    st.divider()
    st.markdown(
        "**Stack**\n\n"
        "`LangChain` · `OpenAI GPT-4o` · `ChromaDB` · `Streamlit`\n\n"
        "Long-term memory stored in `./chroma_db/` — the bot remembers "
        "relevant past conversations even after restart."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        stream_box = st.empty()
        response = get_response(st.session_state.chain, prompt, stream_container=stream_box)
    st.session_state.messages.append({"role": "assistant", "content": response})
