import os
import csv
import pinecone
import requests
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp-free")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "travel-index")

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=384, metric="cosine")
index = pinecone.Index(PINECONE_INDEX_NAME)

def generate_embedding(text: str) -> list:
    """Generate embeddings for the input text using Hugging Face's Inference API."""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL_ID}",
        headers=headers,
        json={"inputs": text}
    )
    response.raise_for_status()
    embedding = response.json()[0]
    return embedding

def load_csv_to_pinecone(csv_file):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            text_for_embedding = ' '.join(f"{key}: {value}" for key, value in row.items() if value)
            metadata = {key: row[key] for key in row.keys()}
            vector = generate_embedding(text_for_embedding)
            index.upsert([(row["Country/City"], vector, metadata)])

if __name__ == "__main__":
    csv_file_path = os.path.join(os.path.dirname(__file__), "travel_info.csv")
    if os.path.exists(csv_file_path):
        print("Loading data from:", csv_file_path)
        load_csv_to_pinecone(csv_file_path)
        print("Data loaded into Pinecone successfully.")
    else:
        print("CSV file not found at:", csv_file_path)
