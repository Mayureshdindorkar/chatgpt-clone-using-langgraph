import streamlit as st
# When we press enter, the whole python script is rerun by sreamlit from top to bottom.
# st.session_state is a dictionary, which dont get erase when we press enter.
# It gets erased only when we manually refresh the page manually.
from backend import chatbot
from langchain_core.messages import HumanMessage
CONFIG = {'configurable': {'thread_id': 'thread_1'}}

# Checking whether session_state dict contains a kay of name 'message_history'
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Showing the conversation history on page
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