from langchain_ollama import OllamaEmbeddings


def get_embeddings():
    local_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    #local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    #local_embeddings = OllamaEmbeddings(model="pankajrajdeo/sentence-transformers_all-minilm-l6-v2:latest")
    return local_embeddings
