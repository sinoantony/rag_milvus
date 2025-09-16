import os
import fitz  # PyMuPDF
import numpy as np
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType
)

# === CONFIG ===
PDF_DIR = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_DIM = 384  # for MiniLM
COLLECTION_NAME = "pdf_chunks"
OLLAMA_MODEL = "tinyllama"
OLLAMA_URL = "http://localhost:11434/api/generate"

# === STEP 1: Extract Text from PDFs ===
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

# === STEP 2: Chunk Text ===
def chunk_texts(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

# === STEP 3: Embed Chunks ===
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings, model

# === STEP 4: Store in Milvus ===
def store_embeddings_milvus(embeddings):
    connections.connect("default", host="localhost", port="19530")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]
    schema = CollectionSchema(fields)
    collection = Collection(COLLECTION_NAME, schema)
    collection.create_index("embedding", {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    })
    collection.insert([embeddings])
    collection.load()
    return collection

# === STEP 5: Retrieve Relevant Chunks ===
def retrieve_milvus(query, embed_model, collection, chunks):
    query_embedding = embed_model.encode([query])
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["id"]
    )

    indices = [hit.id for hit in results[0]]
    valid_indices = [i for i in indices if i < len(chunks)]

    if not valid_indices:
        return "No relevant context found."

    return "\n".join([chunks[i] for i in valid_indices])

# === STEP 6: Generate Answer with Ollama (Streaming) ===
def generate_answer(context, query):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt},
            stream=True
        )
        answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    answer += data.get("response", "")
                except json.JSONDecodeError:
                    continue
        return answer.strip() if answer else "⚠️ Ollama returned no usable output."
    except Exception as e:
        return f"❌ Failed to connect to Ollama: {e}"

# === MAIN ===
if __name__ == "__main__":
    print("🔍 Extracting text...")
    texts = extract_text_from_pdfs(PDF_DIR)

    print("✂️ Chunking...")
    chunks = chunk_texts(texts)

    print("🧬 Embedding...")
    embeddings, embed_model = embed_chunks(chunks)

    print("🧠 Storing in Milvus...")
    collection = store_embeddings_milvus(embeddings)

    print("💬 Ready for queries.")
    query = input("Enter your question: ")
    context = retrieve_milvus(query, embed_model, collection, chunks)

    print("🦙 Generating answer...")
    answer = generate_answer(context, query)
    print("\nAnswer:\n", answer)