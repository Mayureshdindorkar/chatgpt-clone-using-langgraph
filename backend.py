from __future__ import annotations
import os
import sqlite3
import tempfile
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.sqlite import SqliteSaver  # Saves the LangGraph 'State' in Sqlite DB
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests


load_dotenv()


# PDF retriever store (per thread)
_THREAD_RETRIEVER_MAPPING: Dict[str, Any] = {}
_THREAD_METADATA_MAPPING: Dict[str, dict] = {}


# ------ Create a database in sqlite ------#
connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# To support persistance
checkpointer = SqliteSaver(conn=connection)
# -----------------------------------------#



# ------------ Utility function -----------# (Used in frontend.py)
def retrieve_all_unique_threads_from_db():
    all_thread_ids = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        if thread_id:
            all_thread_ids.add(thread_id)
    return list(all_thread_ids)

def get_thread_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVER_MAPPING

def get_thread_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA_MAPPING.get(str(thread_id), {})
# -----------------------------------------#



# ------------ LLM & Embedding ------------#
llm = ChatOpenAI()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
# -----------------------------------------#



# -------------- RAG ----------------------#
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embedding_model)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVER_MAPPING[str(thread_id)] = retriever
        metedata = {
            "filename": filename or os.path.basename(temp_path),
            "number_of_documents": len(docs),
            "number_of_chunks": len(chunks),
        }
        _THREAD_METADATA_MAPPING[str(thread_id)] = metedata

        return metedata
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

# To get retriver of particular thread
def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVER_MAPPING:
        return _THREAD_RETRIEVER_MAPPING[thread_id]
    return None
# -----------------------------------------#



# ------------------ Tools ----------------#
# Tool 1
search_tool = DuckDuckGoSearchRun(region="us-en")

# Tool 2
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

# Tool 3
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=MZMEVDB5A7A6IDUW"
    r = requests.get(url)
    return r.json()

# Tool 4
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA_MAPPING.get(str(thread_id), {}).get("filename"),
    }

# Creating tools list
tools_list = [search_tool, get_stock_price, calculator, rag_tool]

# Binding tools with llm, so that llm knows about the tools
llm_with_tools = llm.bind_tools(tools_list)
# -----------------------------------------#



#------------ Create State schema ---------#
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
# -----------------------------------------#



# ---------- Create node function ---------#
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and calculator tools when helpful."
        )
    )

    # take entire chat history from state
    messages = [system_message, *state['messages']]
    # send to llm
    response = llm_with_tools.invoke(messages)
    # response store state
    return {'messages': [response]}

tool_node = ToolNode(tools_list)
# -----------------------------------------#



# ------------ Create graph ---------------#
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node", tools_condition) # tools_condition: is a inbuilt function, which decides whether to go to tool_node or END node
graph.add_edge('tools', 'chat_node')

# Compile graph
chatbot = graph.compile(checkpointer=checkpointer)
# -----------------------------------------#