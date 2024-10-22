from langchain_ollama import OllamaEmbeddings


def get_embeddings():
    local_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    #local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return local_embeddings
