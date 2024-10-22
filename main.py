import os
import argparse
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
#from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from embeddings import get_embeddings

chroma_path = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""



def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=4)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(f"this is the context the model is using: \n {context_text}")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # get answer generation model
    model = ChatOllama(
        model="llama3.2",
    )
    response_text = model.invoke(prompt)

    formatted_response = f"\n\nResponse: {response_text}\n"
    print(formatted_response)
    return response_text


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


if __name__ == "__main__":
    question = "is there any errors?"
    query_rag(question)
