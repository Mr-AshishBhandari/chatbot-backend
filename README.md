# 🤖 Chatbot Backend

A powerful AI chatbot backend built with **FastAPI**, **LangChain**, and **LangGraph**. This project provides an intelligent conversational API that leverages agent-based workflows, tool calling, and Retrieval-Augmented Generation (RAG). Users can chat naturally, retrieve real-time weather and stock information, and upload documents to query their own knowledge base.

---

## ✨ Features

### 💬 Conversational AI
- Natural language conversations powered by Large Language Models (LLMs).
- Agentic workflows built with **LangGraph**.
- Context-aware responses with conversation memory.

### 🛠️ Tool Calling
The chatbot can intelligently invoke external tools when required:

- 🌦️ **Weather Tool** – Fetches real-time weather information.
- 📈 **Stock Price Tool** – Retrieves live stock market prices.
- 🔧 Easily extensible with additional custom tools.

### 📚 Retrieval-Augmented Generation (RAG)

- Upload documents and chat with your own data.
- Automatic document parsing and chunking.
- Semantic search using vector embeddings.
- Context-aware answers generated from uploaded documents.

### 📄 Document Upload

- Upload PDFs or other supported documents.
- Automatically indexes uploaded files into a vector database.
- Enables personalized document-based question answering.

### ⚡ FastAPI Backend

- High-performance asynchronous REST APIs.
- Interactive API documentation with Swagger.
- Modular and scalable project architecture.

---

# 🏗️ Tech Stack

| Technology | Purpose |
|------------|---------|
| FastAPI | REST API Framework |
| LangGraph | Agent Workflow Orchestration |
| LangChain | LLM Integration & Tool Calling |
| Python | Backend Language |
| Vector Database | Document Embeddings for RAG |
| Pydantic | Data Validation |
| Uvicorn | ASGI Server |

---

# 🚀 Getting Started

## Prerequisites

- Python 3.11+
- pip
- Virtual Environment

---

## Installation

Clone the repository

```bash
git clone https://github.com/Mr-AshishBhandari/chatbot-backend.git

cd chatbot-backend
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

### Linux/macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙️ Environment Variables

Create a `.env` file in the project root.

```env
HUGGING_FACE_API_KEY=your_openai_api_key

WEATHER_API_KEY=your_weather_api_key

STOCK_API_KEY=your_stock_api_key

```

---

# ▶️ Running the Application

Development

```bash
uvicorn app.main:app --reload
```

Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

# 📖 API Documentation

Once the server is running, visit:

Swagger UI

```
http://localhost:8000/docs
```


# 💬 How It Works

```text
                User Message
                     │
                     ▼
               FastAPI Endpoint
                     │
                     ▼
             LangGraph Agent
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Weather Tool   Stock Tool   RAG Retriever
        │            │            │
        └────────────┼────────────┘
                     ▼
              Large Language Model
                     │
                     ▼
              Generated Response
```

---

# 📚 Retrieval-Augmented Generation (RAG)

The backend supports document-based question answering.

### Workflow

1. User Uploads document.
2. Documents are parsed and split into chunks.
3. Chunks are converted into vector embeddings.
4. Embeddings are stored in a vector database(FAISS).
5. During conversations, relevant document chunks are retrieved.
6. Retrieved context is passed to the LLM for accurate responses.

This allows users to ask questions about their own documents in natural language.

---

# 🛠️ Available Tools

| Tool | Description |
|------|-------------|
| 🌦️ Weather Tool | Retrieves current weather information. |
| 📈 Stock Price Tool | Fetches live stock prices. |
| 📄 RAG Tool | Answers questions using uploaded documents. |

---


# 🔮 Future Enhancements

- Multi-user authentication
- Persistent chat history
- Streaming responses
- Additional AI tools
- Multi-agent workflows
- Multiple LLM provider support
- Docker & Kubernetes deployment
- CI/CD integration

---


⭐ If you found this project helpful, consider giving it a star on GitHub!
