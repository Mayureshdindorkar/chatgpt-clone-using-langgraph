from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver   # Saves the State in RAM
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

# To support persistance
checkpointer = InMemorySaver()

# Create graph
graph = StateGraph(ChatState)
# add nodes
graph.add_node('chat_node', chat_node)
# Add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)
# Compile graph
chatbot = graph.compile(checkpointer=checkpointer)
