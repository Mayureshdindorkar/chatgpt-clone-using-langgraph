import streamlit as st
# When we press enter, the whole python script is rerun by sreamlit from top to bottom.
# st.session_state is a dictionary, which dont get erase when we press enter.
# It gets erased only when we manually refresh the page manually.
from backend import chatbot, retrieve_all_unique_threads_from_db
from langchain_core.messages import HumanMessage
import uuid

############## Utility functions ############
def generate_unique_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_unique_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread_id_to_threads_list(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread_id_to_threads_list(thread_id):
    if thread_id not in st.session_state['threads_list']:
        st.session_state['threads_list'].append(thread_id)

def load_chat_history_based_on_thread_id(thread_id):
    state = chatbot.get_state({'configurable': {'thread_id': thread_id}})
    if state and state.values:
        return state.values.get('messages', [])
    return []
#############################################

######## Storing the session state ##########
# Checking whether session_state dict contains a kay of name 'message_history'
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_unique_thread_id()

if 'threads_list' not in st.session_state:
    st.session_state['threads_list'] = retrieve_all_unique_threads_from_db()

add_thread_id_to_threads_list(st.session_state['thread_id'])
#############################################

################## Side bar ##################
st.sidebar.title('LangGraph UI')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['threads_list'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_chat_history_based_on_thread_id(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state['message_history'] = temp_messages

##############################################

############# Chat section ###################
# Showing the past chat history on page
for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
user_input = st.chat_input("Type your message here")
if user_input:
    # Showing user provided input on the page
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # Without streaming
    # Getting AI response from backend
    # response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    # ai_message = response['messages'][-1].content
    # Showing the AI response on the page
    # st.session_state['message_history'].append({"role": "assistant", "content": ai_message})
    # with st.chat_message("assistant"):
    #     st.markdown(ai_message)

    # With streaming
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )
    st.session_state['message_history'].append({"role": "assistant", "content": ai_message})
##############################################
