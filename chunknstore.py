import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility
)
import numpy as np

PDF_DIR = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_DIM = 384
COLLECTION_NAME = "pdf_chunks"

def extract_text_from_pdfs(pdf_dir):
    texts = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_dir, file))
            text = ""
            for page in doc:
                text += page.get_text()
            texts.append(text)
    return texts

def chunk_texts(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def store_embeddings_milvus(chunks, embeddings):
    connections.connect("default", host="localhost", port="19530")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=2048)
    ]
    schema = CollectionSchema(fields)
    # Drop if exists
    if COLLECTION_NAME in utility.list_collections():
        Collection(COLLECTION_NAME).drop()
    # Create collection
    collection = Collection(COLLECTION_NAME, schema)
    # Create index
    collection.create_index("embedding", {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    })
    # Insert data (auto_id means no need for id field input)
    collection.insert([embeddings, chunks])
    collection.load()

def main():
    print("🔍 Extracting text...")
    texts = extract_text_from_pdfs(PDF_DIR)
    print("✂️ Chunking...")
    chunks = chunk_texts(texts)
    print("🧬 Embedding...")
    embeddings = embed_chunks(chunks)
    print("🧠 Storing in Milvus...")
    store_embeddings_milvus(chunks, embeddings)
    print("✅ Documents processed and stored in Milvus.")

if __name__ == "__main__":
    main()
