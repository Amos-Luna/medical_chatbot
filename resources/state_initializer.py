import streamlit as st


def initialize_session_state():
    """Inicializa las variables en session_state si no existen"""
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
        
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    
    if "user_message_p1" not in st.session_state:
        st.session_state.user_message_p1 = None
        
    if "chunks_retrived_p2" not in st.session_state:
        st.session_state.chunks_retrived_p2 = None
        
    if "ai_response_p3" not in st.session_state:
        st.session_state.ai_response_p3 = None
        
    if "score_answer_relevance" not in st.session_state:
        st.session_state.score_answer_relevance = 0.0
    
    if "score_context_relevance" not in st.session_state:
        st.session_state.score_context_relevance = 0.0
    
    if "score_groundedness" not in st.session_state:
        st.session_state.score_groundedness = 0.0
    
    if "score_final" not in st.session_state:
        st.session_state.score_final = 0.0