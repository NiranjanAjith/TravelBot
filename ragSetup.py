import os
import csv
import requests
import time
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp-free")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot")

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")

# Create an instance of the Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if PINECONE_INDEX_NAME in pc.list_indexes().names():
    pc.delete_index(PINECONE_INDEX_NAME)
    logging.warning(f"index {PINECONE_INDEX_NAME} deleted!\n==================================================================")
# Create index if it does not exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    logging.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME} with dimension 384.")
    pc.create_index(PINECONE_INDEX_NAME, dimension=384, metric="cosine", spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pc.Index(PINECONE_INDEX_NAME)

def generate_embedding(text: str) -> list:
    """Generate embeddings for the input text using Hugging Face's Inference API."""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    try:
        logging.info(f"Generating embedding for text: {text[:50]}...")
        response = requests.post(
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL_ID}",
            headers=headers,
            json={"inputs": text}
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Error during embedding request: {e}")
        return None
    
    # Extract and check the embedding format
    embedding = response.json()
    
    # Handle unexpected float response directly
    if isinstance(embedding, float):
        logging.warning(
            f"Embedding for text '{text[:50]}...' returned a single float value: {embedding}. "
            "Skipping this entry due to unexpected format."
        )
        return None
    
    # Proceed if embedding is in list format
    if isinstance(embedding, list) and len(embedding) > 0:
        vector = embedding
        
        # Ensure vector is of correct dimensionality
        if isinstance(vector, list) and len(vector) == 384:
            logging.info(f"Successfully generated embedding for text: {text[:50]}.")
            return vector
        else:
            logging.warning(
                f"Embedding for text '{text[:50]}...' has invalid dimensions or format."
                " Expected a list of length 384. Skipping this entry."
            )
            return None
    else:
        logging.warning(
            f"Unexpected embedding response format for text '{text[:50]}...'. Skipping this entry."
        )
        return None

def load_csv_to_pinecone(csv_file):
    """Load data from a CSV file and upsert embeddings into Pinecone."""
    total_rows = 0
    successful_upserts = 0
    skipped_rows = 0
    
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            logging.info(f"Loading data from CSV file: {csv_file}")
            reader = csv.DictReader(file)
            total_rows = sum(1 for _ in reader)  # Count total rows
            file.seek(0)  # Reset file pointer to start reading again
            reader = csv.DictReader(file)
            
            for row in reader:
                text_for_embedding = ' '.join(f"{key}: {str(value)}" for key, value in row.items() if value)
                metadata = {key: row[key] for key in row.keys()}
                vector = generate_embedding(text_for_embedding)
                
                # Only upsert if the embedding vector is valid
                if vector is not None:
                    logging.info(f"Upserting data for '{row['Country/City']}' into Pinecone.")
                    index.upsert([(row["id"], vector, metadata)], namespace="destinations")
                    successful_upserts += 1
                else:
                    logging.info(f"Skipping row for '{row['Country/City']}' due to invalid embedding.")
                    skipped_rows += 1
                
                time.sleep(2)  # Throttle requests to avoid rate limiting
                
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_file}")
    except Exception as e:
        logging.error(f"An error occurred while loading the CSV file: {e}")

    # Log summary of the loading process
    logging.info(f"Total rows in CSV: {total_rows}")
    logging.info(f"Total successful upserts: {successful_upserts}")
    logging.info(f"Total skipped rows due to embedding issues: {skipped_rows}")

if __name__ == "__main__":
    # Example CSV file path
    csv_file_path = os.path.join(os.path.dirname(__file__), "travel_info.csv")
    
    if os.path.exists(csv_file_path):
        load_csv_to_pinecone(csv_file_path)
        logging.info("Data loading process completed.")
    else:
        logging.error(f"CSV file not found at: {csv_file_path}")

    # Output index statistics
    logging.info("Pinecone index stats:")
    logging.info(index.describe_index_stats())
