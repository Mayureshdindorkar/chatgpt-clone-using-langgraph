import streamlit as st
# When we press enter, the whole python script is rerun by sreamlit from top to bottom.
# st.session_state is a dictionary, which dont get erase when we press enter.
# It gets erased only when we manually refresh the page manually.
from backend import chatbot, retrieve_all_unique_threads_from_db, ingest_pdf, get_thread_metadata
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# st.session_state = {

#     "thread_id" : <current thread id>,
#     "sidebar_threads_list" : [list of all thread ids],                                    # To show all past threads in sidebar
#     "current_thread_message_history" : [list of all messages in current thread],          # To show previous chat messages of current thread (current chat session)

#     "uploaded_docs" : {
#         "<thread_id_1>" : {
#             "<filename_1>" : {
#                 "filename": "<filename_1>",
#                 "number_of_chunks": <num_chunks>,
#                 "number_of_documents": <num_pages>
#             },
#             "<filename_2>" : {
#                 "filename": "<filename_2>",
#                 "number_of_chunks": <num_chunks>,
#                 "number_of_documents": <num_pages>
#             }
#         },
#         "<thread_id_2>" : {
#             "<filename_1>" : {
#                 "filename": "<filename_1>",
#                 "number_of_chunks": <num_chunks>,
#                 "number_of_documents": <num_pages>
#             }
#         }
#     },
# }

#--------------- Utility functions -----------------#
def _generate_unique_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def add_thread_id_to_sidebar_threads_list(thread_id):
    if thread_id not in st.session_state['sidebar_threads_list']:
        st.session_state['sidebar_threads_list'].append(thread_id)

# This gets called when 'New Chat' button is pressed in sidebar
def reset_chat():
    thread_id = _generate_unique_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread_id_to_sidebar_threads_list(st.session_state['thread_id'])
    st.session_state['current_thread_message_history'] = []

def load_chat_history_based_on_thread_id(thread_id):
    state = chatbot.get_state({'configurable': {'thread_id': thread_id}})
    if state and state.values:
        return state.values.get('messages', [])
    return []
#--------------------------------------------------#



#--------- Session state Initialization -----------#
# Current session thread id
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = _generate_unique_thread_id()

# Checking whether session_state dict contains a key of name 'current_thread_message_history'
if 'current_thread_message_history' not in st.session_state:
    st.session_state['current_thread_message_history'] = []

# sidebar_threads_list is used to populate the sidebar, to show all the past threads (i.e, all past chat sessions)
if 'sidebar_threads_list' not in st.session_state:
    st.session_state['sidebar_threads_list'] = retrieve_all_unique_threads_from_db()

if "uploaded_docs" not in st.session_state:
    st.session_state["uploaded_docs"] = {}

# Append the current_chat_thread_id to sidebar_threads_list, so that current_chat_thread_id gets shown in sidebar.
add_thread_id_to_sidebar_threads_list(st.session_state['thread_id'])

# For RAG: To keep track of ingested documents per thread
thread_id = str(st.session_state["thread_id"])
uploaded_docs_of_current_thread = st.session_state["uploaded_docs"].setdefault(thread_id, {})
sidebar_threads_list = st.session_state["sidebar_threads_list"][::-1]
selected_thread = None
#--------------------------------------------------#



#---------------- Side bar ------------------------#
st.sidebar.title('User Chats')

# Show the 'New Chat' button
if st.sidebar.button("Create new chat", use_container_width=True):
    reset_chat()
    st.rerun()
st.sidebar.divider()

# Print current chat thread_id
st.sidebar.markdown(f"**Current chat:** `{thread_id}`")

# File upload section
if uploaded_docs_of_current_thread:
    latest_doc = list(uploaded_docs_of_current_thread.values())[-1]
    st.sidebar.success(f"Using `{latest_doc.get('filename')}` ({latest_doc.get('number_of_chunks')} chunks from {latest_doc.get('number_of_documents')} pages)")
else:
    st.sidebar.info("No PDF uploaded yet!")
# To upload the pdf
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in uploaded_docs_of_current_thread:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Uploading PDF....", expanded=True) as status_box:
            metedata = ingest_pdf(
                file_bytes=uploaded_pdf.getvalue(),
                thread_id=thread_id,
                filename=uploaded_pdf.name,
            )
            uploaded_docs_of_current_thread[uploaded_pdf.name] = metedata
            status_box.update(label="✅ PDF uploaded", state="complete", expanded=False)
st.sidebar.divider()


# List past chats
st.sidebar.subheader("Past conversations")
if not sidebar_threads_list:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in sidebar_threads_list:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):   # This line will display hte buttons with thread_id in sidebar. & If we click on particular thread_id button in sidebar, it will the code in if block.
            selected_thread = thread_id
#--------------------------------------------------#



#------------------ Chat section ------------------#
st.title("Multi Utility Chatbot")

# Showing the past chat history of the current thread in the chat section
for message in st.session_state['current_thread_message_history']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
user_input = st.chat_input("Type your message here")
if user_input:
    # Showing user provided input on the page
    st.session_state['current_thread_message_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # metadata.thread_id show the chats in the thread section of LangSmith
    CONFIG = {
        'configurable': {'thread_id': st.session_state['thread_id']},
        'metadata': {
            'thread_id': st.session_state['thread_id'],
        },
        'run_name': 'chat_turn'
    }

    # Without streaming
    # Getting AI response from backend
    # response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    # ai_message = response['messages'][-1].content
    # Showing the AI response on the page
    # st.session_state['message_history'].append({"role": "assistant", "content": ai_message})
    # with st.chat_message("assistant"):
    #     st.markdown(ai_message)

    # With streaming [chatbot.stream() returns us a generator. We extract messages from it one by one using yield method]
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}
        def show_AI_and_Tool_messages_separately_using_streaming():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` ....", expanded=False)
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=False,
                        )
                    with status_holder["box"]:
                        st.markdown(message_chunk.content)
                elif isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        ai_message = st.write_stream(show_AI_and_Tool_messages_separately_using_streaming())
        # If the tool was called, then close the tool call dropdown
        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool call finished", state="complete", expanded=False)
    st.session_state['current_thread_message_history'].append({"role": "assistant", "content": ai_message})

    doc_meta = get_thread_metadata(thread_id)
    if doc_meta:
        st.caption(f"References: {doc_meta.get('filename')} (chunks: {doc_meta.get('number_of_chunks')}, pages: {doc_meta.get('number_of_documents')})")
st.divider()


# If the user selects one particular thread_id from side bar, the show the chat history of that thread in the chat area
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_chat_history_based_on_thread_id(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["current_thread_message_history"] = temp_messages
    st.session_state["uploaded_docs"].setdefault(str(selected_thread), {})
    st.rerun()
#--------------------------------------------------#