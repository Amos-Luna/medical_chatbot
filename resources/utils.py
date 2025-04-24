import pandas as pd
from langchain_core.documents import Document


def read_excel(file_path):
    """Read an Excel file and return a DataFrame."""
    
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
        df["mixed_text"] = "user question ->: " + df["question"] + "\n" + "doctor answer ->: " + df["answer"]

        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None
    

def build_documents(
    df: pd.DataFrame
) -> list[Document]:
    
    """Convert DataFrame to a list of dictionaries."""
    try:
        documents = [
            Document(
                page_content=row["mixed_text"], 
                metadata={
                    "question": row["question"], 
                    "answer": row["answer"]
                }
            ) for _, row in df.iterrows()
        ]
        return documents
    except Exception as e:
        print(f"Error building documents: {e}")
        return None