import os
import json
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from get_secret_openai import get_secret
import uuid


# ========================
# 1. Set up OpenAI API
# ========================
os.environ["OPENAI_API_KEY"] = get_secret()

# ========================
# 2. Load Documents
# ========================
def load_documents(directory_path):
    """Load documents from a folder"""
    loader = SimpleDirectoryReader(directory_path)
    return loader.load_data()

# ========================
# 3. Chunk Documents with Overlap
# ========================
def chunk_document_with_overlap(document, chunk_size=200, overlap=10):
    """
    Split a document into smaller chunks with overlap.

    Parameters:
    - document (str): The text of the document to chunk.
    - chunk_size (int): The maximum size of each chunk in words.
    - overlap (int): The number of overlapping words between consecutive chunks.

    Returns:
    - List[str]: A list of text chunks.
    """
    words = document.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    return chunks

# ========================
# 4. Generate Embeddings with OpenAIEmbedding
# ========================
def generate_embeddings_with_llama(model_name, text_chunks):
    """Generate embeddings for a list of text chunks using OpenAIEmbedding"""
    openai_embed = OpenAIEmbedding(model=model_name)
    embeddings = []
    for chunk in text_chunks:
        embedding = openai_embed.get_text_embedding(chunk)
        embeddings.append({
            "chunk": chunk,
            "embedding": embedding,
            "embedding_model": model_name  # Adding embedding model information
        })
    return embeddings

# ========================
# 5. Save Embeddings to JSON
# ========================
def save_embeddings_to_json(embeddings, file_path="embedding_folder"):
    """Ensure the directory exists"""
    os.makedirs(file_path, exist_ok=True)

    """Generate file name"""
    file_name = uuid.uuid4().hex + ".json"
    full_path = os.path.join(file_path, file_name)

    """Save embeddings to a JSON file"""
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)
    # print(f"Embeddings saved to {full_path}")

    return full_path

# ========================
# Main Process
# ========================

def embedding_files_multiple_dirs(directory_paths, model_name, chunk_size, overlap=10):
    """
    Process multiple directories of files and save their embeddings.

    Parameters:
    - directory_paths (list): List of directories containing files.
    - model_name (str): the model of embedding: text-embedding-3-large, text-embedding-ada-002, text-embedding-3-small
    - chunk_size: size of every chunk
    - output_file (str): Path to save the embeddings JSON file.
    """
    all_embeddings = []

    documents = load_documents(directory_paths)

    # Process each document
    for document in documents:
        # Split document into chunks with overlap
        chunks = chunk_document_with_overlap(document.text, chunk_size=chunk_size, overlap=overlap)
        # Generate embeddings for each chunk
        embeddings = generate_embeddings_with_llama(model_name, chunks)

        # Add document name to each embedding
        for embedding in embeddings:
            embedding["document_name"] = document.doc_id

        all_embeddings.extend(embeddings)

    # Save all embeddings to a JSON file
    output_file = save_embeddings_to_json(all_embeddings)

    print(f"Embeddings saved to {output_file}.")

    import glob

    files = glob.glob('./temp_uploads/*')
    for f in files:
        os.remove(f)

    return output_file


# Example usage:
# embedding_files_multiple_dirs([r"C:\Users\yosef\Downloads\Inention", r"C:\Users\yosef\Downloads\Portal"], "text-embedding-3-small", 200)
# "C:\Users\yosef\Downloads\INTENTION BEYOND - V2.pdf"
