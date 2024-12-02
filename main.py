import os
import argparse
import numpy as np
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
#from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from embeddings import get_embeddings
from sentence_transformers import CrossEncoder


chroma_path = "chroma"
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
PROMPT_TEMPLATE = """
You are a helpful chatbot of patrol and security robot.
You are asked about robot data logs.
Answer the question with clear sentences based on the context. 
For errors or warnings questions provide a list of the results.

Context: {context}
Question: {question}
"""
def rerank_docs(query, retrieved_documents):
    document_texts = [doc.page_content for doc in retrieved_documents]
    pairs = [[query, doc] for doc in document_texts]
    scores = cross_encoder.predict(pairs)
    indexes = np.argsort(scores)[::-1]
    return [retrieved_documents[i] for i in indexes[:1]]

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=2)
    retrieved_documents = [doc for doc, _score in results]
    top_document = rerank_docs(query_text, retrieved_documents)

    context_text = "\n\n".join([doc.page_content for doc in top_document])
    print(f"this is the context the model is using: \n {context_text}")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # get answer generation model
    model = ChatOllama(
        model="llama3.2:latest",
        temperature= 0.8
    )
    response_text = model.invoke(prompt)

    formatted_response = f"\n\nResponse: {response_text}\n"
    print(formatted_response)
    return response_text


def clean_response(response: str) -> str:
    # Remove leading/trailing spaces and ensure consistent newlines
    response = response.strip()

    # Standardize bullet points and indentation
    response = response.replace("* ", "• ")  # Replace '*' with '•'
    response = response.replace("+ ", "  - ")  # Replace '+' with a nested dash

    # Handle newlines for consistent formatting
    response = '\n'.join(line.strip() for line in response.splitlines())

    return response

# def main():
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#     query_rag(query_text)


if __name__ == "__main__":
    question = "give the errors the robot is facing?"
    answer = query_rag(question)
    answer = answer.get("content", "")
    clean_response(answer)
