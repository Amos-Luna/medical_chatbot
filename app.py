import streamlit as st
from dotenv import load_dotenv
from chatbot.core.graph import Graph
from resources.vector_store_manager import VectorDBManager
from resources.utils import read_excel, build_documents
from resources.state_initializer import initialize_session_state
from chatbot.core.utils import (
    qualify_answer_relevance, 
    qualify_context_relevance, 
    qualify_groundedness
)
load_dotenv()

db_manager = VectorDBManager()

def main():
    """
    Main function to run the Streamlit app.
    Sets up the page layout and calls other functions.
    """
    st.set_page_config(layout="wide", page_title="AI Medical Assistant")
    initialize_session_state()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        handle_file_upload()
        display_rag_metrics()
    
    with col2:
        st.header("👨‍⚕️Hi, I'm your Medical Assitant Chatbot 🏥")
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
        st.session_state.user_message_p1 = prompt
        
        graph = Graph()
        response, chunks_retrieved = graph.execute_agent(prompt)
    
        st.session_state.ai_response_p3 = response
        st.session_state.chunks_retrived_p2 = chunks_retrieved
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        get_rag_metrics()
        st.rerun()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def get_rag_metrics():
    """Get metrics using trulens for rag processing"""
    user_message = st.session_state.user_message_p1
    chunks_retrieved = st.session_state.chunks_retrived_p2
    ai_response = st.session_state.ai_response_p3

    score_answer_relevance = float(qualify_answer_relevance(user_message, ai_response))
    score_context_relevance = float(qualify_context_relevance(user_message, chunks_retrieved))
    score_groundedness = float(qualify_groundedness(chunks_retrieved, ai_response))
    
    score_answer_relevance = min(10.0, max(0.0, score_answer_relevance))
    score_context_relevance = min(10.0, max(0.0, score_context_relevance))
    score_groundedness = min(10.0, max(0.0, score_groundedness))
    
    score_final = (
        (score_answer_relevance / 10.0) * 0.3 +
        (score_context_relevance / 10.0) * 0.2 +
        (score_groundedness / 10.0) * 0.5
    )
    
    score_final = min(1.0, max(0.0, score_final))
    
    st.session_state.score_answer_relevance = score_answer_relevance
    st.session_state.score_context_relevance = score_context_relevance
    st.session_state.score_groundedness = score_groundedness
    st.session_state.score_final = score_final


def display_rag_metrics():
    """
    Muestra las métricas RAG con barras horizontales de Streamlit
    """
    st.subheader("RAG Performance Metrics")
    
    metrics_container = st.container()
    with metrics_container:
        col1, col2 = st.columns([2, 8])
        with col1:
            st.markdown("**Answer Relevance:**")
        with col2:
            raw_score = float(st.session_state.score_answer_relevance)
            normalized_score = float(raw_score / 10.0)
            normalized_score = min(1.0, max(0.0, normalized_score))
            st.progress(value=normalized_score, text=f"{raw_score:.1f}/10")
            
        col1, col2 = st.columns([2, 8])
        with col1:
            st.markdown("**Context Relevance:**")
        with col2:
            raw_score = float(st.session_state.score_context_relevance)
            normalized_score = float(raw_score / 10.0)
            normalized_score = min(1.0, max(0.0, normalized_score))
            st.progress(value=normalized_score, text=f"{raw_score:.1f}/10")
            
        col1, col2 = st.columns([2, 8])
        with col1:
            st.markdown("**Groundedness:**")
        with col2:
            raw_score = float(st.session_state.score_groundedness)
            normalized_score = float(raw_score / 10.0)
            normalized_score = min(1.0, max(0.0, normalized_score))
            st.progress(value=normalized_score, text=f"{raw_score:.1f}/10")
        
        col1, col2 = st.columns([2, 8])
        with col1:
            st.markdown("**Overall Score:**")
        with col2:
            score = float(st.session_state.score_final)
            score = min(1.0, max(0.0, score))
            st.progress(value=score, text=f"{int(score*100)}%")


if __name__ == "__main__":
    main()