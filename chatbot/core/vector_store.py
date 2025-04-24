from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings


def vector_store(text_chunks):
    
    embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")