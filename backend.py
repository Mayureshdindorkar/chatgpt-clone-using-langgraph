from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver  # Saves the State in Sqlite DB
load_dotenv()

# Get LLM
llm = ChatOpenAI()

# Create State schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Create node function
def chat_node(state: ChatState):
    # take entire chat history from state
    messages = state['messages']
    # send to llm
    response = llm.invoke(messages)
    # response store state
    return {'messages': [response]}

# create a database in sqlite
connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# To support persistance
checkpointer = SqliteSaver(conn=connection)

# Create graph
graph = StateGraph(ChatState)
# add nodes
graph.add_node('chat_node', chat_node)
# Add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)
# Compile graph
chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_unique_threads_from_db():
    all_thread_ids = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        if thread_id:
            all_thread_ids.add(thread_id)
    return list(all_thread_ids)