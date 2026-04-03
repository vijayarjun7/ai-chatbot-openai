# 🤖 AI Chatbot — GPT-4o + LangChain + ChromaDB

A multi-turn AI chatbot with conversation memory, built with LangChain, OpenAI GPT-4o, ChromaDB, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-purple)

---

## ✨ Features

- 💬 **Multi-turn conversations** — remembers your last 8 messages
- ⚡ **GPT-4o streaming** — token-by-token response with live cursor
- 🧠 **LangChain ConversationChain** — structured memory management
- 🗄️ **ChromaDB long-term memory** — persists relevant context across sessions
- 🎨 **Streamlit UI** — clean chat interface with sidebar controls
- 🗑️ **Clear history** — reset conversation with one click

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o |
| Orchestration | LangChain |
| Short-term Memory | ConversationBufferWindowMemory (8 turns) |
| Long-term Memory | ChromaDB VectorStoreRetrieverMemory |
| UI | Streamlit |
| Language | Python 3.11 |

---

## 🚀 Quick Start

```bash
git clone https://github.com/vijayarjun7/ai-chatbot-openai.git
cd ai-chatbot-openai
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
streamlit run app.py
```

---

## 📁 Project Structure

```
ai-chatbot-openai/
├── app.py              # Streamlit UI + streaming
├── utils/
│   └── chat.py         # LangChain + ChromaDB memory logic
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key from platform.openai.com |

---

## 🌐 Deploy on Streamlit Cloud

1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repo → add `OPENAI_API_KEY` in Secrets → Deploy 🚀

---

## 👩‍💻 Author

**Vijaya Kumari** — Gen AI Engineer | LangChain · RAG · Azure OpenAI · GitHub Copilot Agents

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vijayakumari007-blue)](https://linkedin.com/in/vijayakumari007)
[![GitHub](https://img.shields.io/badge/GitHub-vijayarjun7-black)](https://github.com/vijayarjun7)
