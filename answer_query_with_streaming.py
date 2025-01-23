import os
import json
import numpy as np
from typing import List, Optional, Generator
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.data_structs import Node
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from get_secret_openai import get_secret

os.environ["OPENAI_API_KEY"] = get_secret()


def initialize_settings(language_model: str, embedding_model: str):
    """Initialize global settings for LLM and embeddings"""
    Settings.llm = OpenAI(model=language_model, streaming=True)
    Settings.embed_model = OpenAIEmbedding(model=embedding_model)


def load_embeddings_from_json(json_file_path: str, embedding_model: str) -> Optional[List[Node]]:
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    filtered_data = [
        entry for entry in data
        if entry.get("embedding_model") == embedding_model
    ]
    if not filtered_data:
        return None
    return [
        Node(
            doc_id=str(idx),
            text=entry["chunk"],
            embedding=np.array(entry["embedding"])
        )
        for idx, entry in enumerate(filtered_data)
    ]


def build_and_query_index(nodes: List[Node], query: str, top_k: int) -> Generator[str, None, None]:
    index = VectorStoreIndex(nodes, show_progress=True)
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=top_k
    )
    response = query_engine.query(query)

    # Yield streaming response
    for text in response.response_gen:
        yield text


def query_embedding_with_streaming(
        query: str = "What are the key points in these documents?",
        json_embedding: str = "embedding_folder/8c3951f6e15a4c148dba9b13e0fa4786.json",
        language_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5
) -> Optional[str]:
    try:
        initialize_settings(language_model, embedding_model)
        nodes = load_embeddings_from_json(json_embedding, embedding_model)

        if nodes is None:
            return "No embeddings found for the selected model."

        stream_generator = build_and_query_index(nodes, query, top_k)
        full_response = ""
        for chunk in stream_generator:
            full_response += chunk

        return full_response

    except Exception as e:
        return f"Error processing query: {str(e)}"


if __name__ == "__main__":
    query_embedding_with_streaming(
        query="What are the key points of the portal",
        json_embedding="embedding_folder/8c3951f6e15a4c148dba9b13e0fa4786.json",
        language_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        top_k=3
    )