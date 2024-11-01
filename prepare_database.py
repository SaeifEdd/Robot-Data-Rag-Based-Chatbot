import json
import argparse
import shutil
import os
from langchain_core.documents import Document
#from langchain_community.document_loaders import JSONLoader
#from langchain_text_splitters import RecursiveJsonSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
#from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from embeddings import get_embeddings

chroma_path = "./chroma"
file_path = 'transformed_data.json'

# Open and read the JSON file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def lists_to_documents(data: list):
    documents = []

    # Map each list to a document with appropriate metadata
    for i, log_list in enumerate(data):
        # Determine document type based on content patterns
        doc_type = "general"
        if any("Device ID:" in entry for entry in log_list):
            doc_type = "device_status"
        elif any("Error" in entry for entry in log_list):
            doc_type = "errors"
        elif any("Warning" in entry for entry in log_list):
            doc_type = "warnings"
        elif any("Connection Status" in entry for entry in log_list):
            doc_type = "connection"

        # Join the list entries with newlines to create the document content
        content = "\n".join(log_list)

        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "type": doc_type,
                "list_index": i
            }
        )
        documents.append(doc)

    return documents


def create_database(splits):
    db = Chroma.from_documents(documents=splits,
                                        embedding=get_embeddings(),
                                        persist_directory=chroma_path)



def clear_database():
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
# test


if __name__ == "__main__":
    clear_database()
    data = load_data(file_path)
    splits = lists_to_documents(data)

    if not os.path.exists('chroma_path'):
        create_database(splits)
    else:
        print("Database already exists. Skipping creation.")



