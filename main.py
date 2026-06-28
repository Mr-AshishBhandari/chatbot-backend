import asyncio
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from chatbot import chatbot
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["POST"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    message: str
    session_id: str


@app.post("/ingest")
async def chat(message: ChatMessage):
    inital_state = {"messages": [message.message], "thread_id": message.session_id}
    final_state = await chatbot.ainvoke(
        inital_state, config={"configurable": {"thread_id": message.session_id}}
    )
    return final_state["messages"][-1].content


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile, thread_id: str = Form()):
    try:
        contents = await file.read()
        path = f"temp/{file.filename}"
        with open(path, "wb") as f:
            f.write(contents)

        # document loader
        loader = PyPDFLoader(path)
        docs = loader.load()

        # chunking docs
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # embedding model
        embedding = HuggingFaceEndpointEmbeddings(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )

        # vector store and retriever
        vector_store = FAISS.from_documents(chunks, embedding=embedding)
        vector_store.save_local(f"vectorstores/{thread_id}")

        return {"file": file.filename, "docs": len(docs), "chunks": len(chunks)}
    finally:
        try:
            os.remove(path=path)
        except:
            pass


@app.get("/")
async def home_page():
    inital_state = {"messages": ["show welcome message to user"],"thread_id": "1"}
    final_state = await chatbot.ainvoke(
        inital_state, config={"configurable": {"thread_id": "1"}}
    )
    return final_state["messages"][-1].content
