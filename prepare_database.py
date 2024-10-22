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


def split_data(data):
    documents = [Document(page_content=' '.join(item)) for item in data]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    all_splits = text_splitter.split_documents(documents)

    return all_splits


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
    data  = load_data(file_path)
    splits = split_data(data)

    if not os.path.exists('chroma_path'):
        create_database(splits)
    else:
        print("Database already exists. Skipping creation.")



