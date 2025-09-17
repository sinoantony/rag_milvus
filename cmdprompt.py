from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
import requests
import json

EMBEDDING_DIM = 384
COLLECTION_NAME = "pdf_chunks"
OLLAMA_MODEL = "tinyllama"
OLLAMA_URL = "http://localhost:11434/api/generate"

def retrieve_milvus(query, embed_model, collection):
    query_embedding = embed_model.encode([query])
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["chunk"]
    )
    if not results[0]:
        return "No relevant context found."
    return "\n".join([hit.entity.get("chunk", "") for hit in results[0]])

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

def main():
    # Load Milvus and embedding model ONCE
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(COLLECTION_NAME)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Main loop
    while True:
        query = input("\nEnter your question (or 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            break
        context = retrieve_milvus(query, embed_model, collection)
        print("🦙 Generating answer...")
        answer = generate_answer(context, query)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
