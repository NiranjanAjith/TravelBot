import os
import spacy
import re
import logging
import requests
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Any
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO)

load_dotenv()

PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", 100))
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")

nlp = spacy.load(SPACY_MODEL)

class RAGHandler:
    def __init__(self, pinecone_index_name: str):
        """Initialize RAGHandler with Pinecone."""
        if not PINECONE_API_KEY:
            raise EnvironmentError("Missing Pinecone API key in environment variables.")
        try:
            self.index_name = pinecone_index_name
            # Initialize the Pinecone client
            self.pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

            # Check if the index exists; raise an error if it doesn't
            if self.index_name not in self.pc.list_indexes().names():
                raise ValueError(f"Index '{self.index_name}' does not exist.")

            # Connect to the index
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            logging.error("Error initializing Pinecone client: %s", e)
            raise RuntimeError("Error initializing Pinecone client") from e

    def generate_embedding(self, text: str) -> list:
        """Generate embeddings for the input text using Hugging Face's Inference API."""
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        try:
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

    def retrieve_and_filter_documents(self, user_input: str, memory_data: Dict[str, Any]) -> List[str]:
        """Retrieve relevant documents and apply memory-based filters using Pinecone query."""
        # combined_query = self._combine_query_with_memory(user_input, memory_data)
        # Generate embedding vector for the combined query
        embedding_vector = self.generate_embedding(user_input) #replaced combined_query for parameter
        
        # Query Pinecone using the embedding vector
        try:
            results = self.index.query(namespace="destinations", vector=embedding_vector, top_k=5, include_metadata=True)
            return [match['metadata'] for match in results.get('matches', [])]
        except Exception as e:
            logging.error("Error querying Pinecone: %s", e)
            return []

    def _combine_query_with_memory(self, user_input: str, memory_data: Dict[str, Any]) -> str:
        """Combine user query with memory data for refined retrieval."""
        memory_context = ', '.join([f"{k}: {v}" for k, v in memory_data.items() if v])
        return f"{user_input} {memory_context}"


class TravelMemory:
    def __init__(self):
        """Initialize memory storage and chat history for the chatbot."""
        self.memory = {}
        self.chat_history = []
        self.synonyms = self.load_synonyms()

    @staticmethod
    def load_synonyms() -> Dict[str, List[str]]:
        """Load or define synonyms dictionary."""
        return {
            'destination': ['place', 'location', 'area', 'city', 'country', 'region'],
            'budget': ['cost', 'price', 'amount', 'spending', 'expense'],
            'activity_preference': ['interest', 'activity', 'hobby', 'preferences'],
            'head_count': ['number of people', 'participants', 'count', 'group size'],
            'date': ['when', 'day', 'time', 'period', 'date'],
            'transportation': ['travel', 'vehicle', 'way to get there', 'commute'],
            'accommodation': ['stay', 'hotel', 'lodging', 'where to sleep'],
        }

    def update_memory(self, user_input: str):
        """Update memory with user input using named entity recognition and heuristic checks."""
        try:
            doc = nlp(user_input)
            for ent in doc.ents:
                if ent.label_ == "GPE":
                    self.memory['destination'] = ent.text
                elif ent.label_ == "DATE":
                    self.memory['date'] = ent.text
                elif ent.label_ == "MONEY":
                    budget_value = self._parse_budget(ent.text)
                    if budget_value > 0:
                        self.memory['budget'] = budget_value
            self._extract_additional_preferences(user_input)
        except Exception as e:
            logging.error("Error updating memory: %s", e)

    def _parse_budget(self, text: str) -> float:
        """Parse budget text to a float value."""
        try:
            return float(text.replace('$', '').replace(',', ''))
        except ValueError:
            logging.warning("Failed to parse budget from text: %s", text)
            return 0.0

    def _extract_additional_preferences(self, user_input: str):
        """Extract preferences such as activity, head count, etc., based on heuristics."""
        if self.memory.get('activity_preference') is None:
            self.memory['activity_preference'] = self._match_synonyms(user_input, 'activity_preference')
        if self.memory.get('head_count') is None:
            self.memory['head_count'] = self._match_head_count(user_input)
        if self.memory.get('transportation') is None:
            self.memory['transportation'] = self._match_synonyms(user_input, 'transportation')
        if self.memory.get('accommodation') is None:
            self.memory['accommodation'] = self._match_synonyms(user_input, 'accommodation')

    def _match_synonyms(self, user_input: str, key: str) -> str:
        """Match user input with synonyms."""
        for synonym in self.synonyms.get(key, []):
            if synonym in user_input.lower():
                return user_input.split(synonym)[-1].strip()
        return None

    def _match_head_count(self, user_input: str) -> int:
        """Match head count with regex."""
        match = re.search(r'(\d+)\s*(?:people|persons|guests|friends)', user_input, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def limit_chat_history(self):
        """Limit chat history size to prevent memory overflow."""
        if len(self.chat_history) > CHAT_HISTORY_LIMIT:
            self.chat_history = self.chat_history[-CHAT_HISTORY_LIMIT:]

    def get_memory(self) -> Dict[str, Any]:
        """Return current memory."""
        return self.memory

    def clear_memory(self):
        """Clear all memory."""
        self.memory.clear()

    def add_chat(self, user_input: str, bot_response: str):
        """Add a chat interaction to history."""
        self.chat_history.append({"user": user_input, "bot": bot_response})
        self.limit_chat_history()

    def update_memory_fact(self, key: str, value: Any):
        """Update specific memory fact."""
        self.memory[key] = value


class TravelChatbot:
    def __init__(self, pinecone_index_name: str=PINECONE_INDEX_NAME):
        """Initialize Travel Chatbot with memory, retrieval handler, and rate limit settings."""
        self.memory = TravelMemory()
        self.rag_handler = RAGHandler(pinecone_index_name=pinecone_index_name)
        self.groq_client = self._init_groq_client()
        self.request_count = 0
        self.MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", 100))
        self.MAX_INPUT_LENGTH = 256

    def _init_groq_client(self) -> Groq:
        """Initialize Groq client, raising error on failure."""
        try:
            return Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            logging.error("Error initializing Groq client: %s", e)
            raise RuntimeError("Error initializing Groq client") from e

    def sanitize_input(self, input_text: str) -> str:
        """Sanitize user input to prevent injection attacks and limit length."""
        sanitized_input = re.sub(r"[^\w\s,.!?'-]", "", input_text)
        return sanitized_input[:self.MAX_INPUT_LENGTH]

    def process_user_input(self, user_input: str) -> str:
        """Process user input, update memory, retrieve data, and generate a response."""
        if self.request_count >= self.MAX_REQUESTS:
            return "Rate limit reached. Please try again later."

        sanitized_input = self.sanitize_input(user_input)
        try:
            self.memory.update_memory(sanitized_input)
            memory_data = self.memory.get_memory()
            filtered_documents = self.rag_handler.retrieve_and_filter_documents(sanitized_input, memory_data)
            response = self._generate_response(sanitized_input, filtered_documents)
            text = ""
            for chunk in response:
                text += chunk.choices[0].delta.content or ""
            self.memory.add_chat(sanitized_input, text)
            self.request_count += 1
            print(f"\n\nMEMORY DATA:\n{memory_data}\n")
            return text
        except Exception as e:
            logging.error("Error processing user input: %s", e)
            return "I'm having trouble understanding that request."

    def _generate_response(self, user_input: str, filtered_documents: List[str]) -> str:
        """Generate a response using memory and the LLM based on filtered documents."""
        memory_data = self.memory.get_memory()
        memory_context = (
            f"Destination: {memory_data.get('destination', 'Unknown')}, "
            f"Budget: {memory_data.get('budget', 'Unknown')}, "
            f"Activity: {memory_data.get('activity_preference', 'Unknown')}, "
            f"Date: {memory_data.get('date', 'Unknown')}, "
            f"Transportation: {memory_data.get('transportation', 'Unknown')}, "
            f"Accommodation: {memory_data.get('accommodation', 'Unknown')}"
        )

        system_prompt = "You are an insightful travel assistant in a fictional world and believe it to be the real one. Provide travel information and suggestions based solely on the provided RAG documents and memory context. Do not include information beyond the provided documents, discard all knowlege on real world details and assume what is in the documents to be the truth. Never mix details with real world information or clarify what is in the documents may not be accurate, it is accurate. Utilize memory context to estimate destination, time, duration, budget, etc if provided, never go beyond the filters of memory context. if destination is provided stick with it until said otherwise, if budget is provided adhere to it. Example: User: 'What are unique activities here?' Assistant: 'You can explore the enchanted forest of Evergreen, take a mystical riverboat cruise, or visit the ancient ruins of Eldoria.' All responses must adhere to the RAG documents."

        context = (
            f"{system_prompt}\n"
            f"User query: {user_input}\n"
            f"Memory context: {memory_context}\n"
            f"Relevant documents: {filtered_documents}"
        )
        try:
            # response = self.groq_client.chat(context)
            completion = self.groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )
            return completion
        except:
            logging.error(f"GROQ CLIENT ERROR")
            return None

    def clear_memory(self):
        """Clear memory and chat history."""
        self.memory.clear_memory()
        self.request_count = 0


if __name__ == "__main__":
    try:
        chatbot = TravelChatbot(pinecone_index_name=PINECONE_INDEX_NAME)

        print("Welcome to the Travel Chatbot! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            response = chatbot.process_user_input(user_input)
            print("Bot:", end='')
            for chunk in response:
                print(chunk.choices[0].delta.content or "", end="")
            print()

    except Exception as e:
        logging.error("Failed to initialize chatbot: %s", e)
        print("Error: Could not initialize chatbot.")
