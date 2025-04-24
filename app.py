import streamlit as st
from dotenv import load_dotenv
from chatbot.core.graph import Graph
from resources.vector_store_manager import VectorDBManager
from resources.utils import read_excel, build_documents
load_dotenv()

db_manager = VectorDBManager()


def main():
    """
    Main function to run the Streamlit app.
    Sets up the page layout and calls other functions.
    """
    st.set_page_config(layout="wide", page_title="AI Medical Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        handle_file_upload()
    
    with col2:
        st.header("Hi, I'm your Medical Assitant Chatbot")
        display_chat_interface()


def handle_file_upload():
    """
    Handles file upload functionality in the sidebar.
    Includes upload button, processing with spinner, and status indicator.
    """
    st.header("Upload you QA Excel File")
    
    if st.session_state.uploaded_file is not None:
        st.success(f"Current file: {st.session_state.uploaded_file.name}")
    
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
    
    if uploaded_file is not None and st.button("Submit and Process"):
        with st.spinner("Processing file..."):
            try:
                raw_text = read_excel(uploaded_file)
                documents = build_documents(raw_text)
                db_manager.initialize_db(documents)
                st.session_state.uploaded_file = uploaded_file
                st.session_state.processing_complete = True
            except Exception as e:
                st.error(f"Processing failed: {e}")
    
    if st.session_state.processing_complete:
        st.success("DONE!")


def display_chat_interface():
    """
    Displays the chatbot interface with message history and input field.
    Uses a fixed height container with native scrolling for messages.
    """
    
    chat_container = st.container(height=700)
    
    if prompt := st.chat_input("Write here your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        graph = Graph()
        response = graph.execute_agent(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


if __name__ == "__main__":
    main()