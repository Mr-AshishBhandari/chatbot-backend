from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEndpointEmbeddings,
)
from langchain_core.messages import BaseMessage
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition,InjectedState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import requests
import os


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
stock_api_key = os.getenv("stock_api_key")
weather_api_key = os.getenv("weather_api_key")


model = ChatHuggingFace(llm=llm)
checkpointer = MemorySaver()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    print("_______Tool calling ________")
    return a * b


@tool
def get_stock_price(symbol: str):
    """
    Fetches the latest intraday stock price data for a given stock symbol
    using the Alpha Vantage API.

    This function queries the TIME_SERIES_INTRADAY endpoint with a 5-minute
    interval and returns the JSON response containing recent stock price
    movements, including open, high, low, close, and volume data.

    Parameters:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "GOOGL").

    Returns:
        dict: A JSON dictionary containing intraday time series stock data
              as returned by the Alpha Vantage API. Despite the type hint
              showing float, the actual return value is a dict.

    Raises:
        requests.exceptions.RequestException:
            If there is an issue with the HTTP request (network error, timeout, etc.).

        KeyError / ValueError:
            If the API response is malformed or missing expected fields.
    """
    print("_______Tool calling ________")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_api_key}"
    response = requests.get(url)
    return response.json()


@tool
def get_weather_info(location: str):
    """
    Fetches current weather data for a given location name.

    Args:
        location (str): City or place name (e.g., "Kathmandu").

    Returns:
        dict: Weather data including coordinates, temperature (Kelvin),
        humidity, wind details, cloud coverage, and weather description.

    """
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}"
    response = requests.get(url)
    return response.json()


@tool
def rag_tool(state: Annotated[dict, InjectedState], query: str):
    """
Search and retrieve information from the user's uploaded document.

Use this tool to answer any questions related to the uploaded file, including:
- Resume / CV content
- Education, skills, projects, or experience
- Any information contained in the document

The document is already stored in a FAISS vector store and associated with a
`thread_id` from the conversation state. The tool automatically uses this
`thread_id` to access the correct document.

Do NOT ask the user for the thread_id. It is managed internally by the system.

Args:
    query: A natural language question about the uploaded document.

Returns:
    Relevant document chunks retrieved from the vector store.
"""
    print("________rag tool ________")
    
    embedding = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    vector_store = FAISS.load_local(
        f"vectorstores/{state['thread_id']}",
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )
    reteriver = vector_store.as_retriever()
    docs = reteriver.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"answer": context}


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id : str

def build_graph():

    # mcp_tools = await client.get_tools()
    tools = [multiply, get_stock_price, get_weather_info, rag_tool]
    model_with_tools = model.bind_tools(tools=tools)

    async def chatnode(MessageState: MessageState):
        messages = MessageState["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(MessageState)

    graph.add_node("chatnode", chatnode)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chatnode")
    graph.add_conditional_edges("chatnode", tools_condition)
    graph.add_edge("tools", "chatnode")
    graph.add_edge("chatnode", END)

    chatbot = graph.compile(checkpointer=checkpointer)

    return chatbot

print(rag_tool.args)
chatbot = build_graph()
# async def main():
#     chatbot =  await build_graph()
#     inital_state = {
#         "messages": [
#             HumanMessage(
#                 content="How many shares of google can i buy with  $50,000?"
#             )
#         ]
#     }

#     final_state = await chatbot.ainvoke(
#         inital_state, config={"configurable": {"thread_id": str(uuid.uuid4())}}
#     )
#     print(final_state["messages"][-1].content)

# if __name__=='__main__':
#     asyncio.run(main())
