import os
import pinecone
import spacy
import re
import logging
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)

load_dotenv()

PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", 100))
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

nlp = spacy.load(SPACY_MODEL)

class RAGHandler:
    def __init__(self, pinecone_index_name: str):
        """Initialize RAGHandler with Pinecone."""
        if not PINECONE_API_KEY:
            raise EnvironmentError("Missing Pinecone API key in environment variables.")
        try:
            self.index_name = pinecone_index_name
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
            self.index = pinecone.Index(index_name=self.index_name)
        except Exception as e:
            logging.error("Error initializing Pinecone client: %s", e)
            raise RuntimeError("Error initializing Pinecone client") from e

    def retrieve_and_filter_documents(self, user_input: str, memory_data: Dict[str, Any]) -> List[str]:
        """Retrieve relevant documents and apply memory-based filters using Pinecone query."""
        combined_query = self._combine_query_with_memory(user_input, memory_data)
        return self.retrieve_documents(combined_query)

    def retrieve_documents(self, combined_query: str) -> List[str]:
        """Query Pinecone to find relevant documents."""
        try:
            results = self.index.query(queries=[combined_query], top_k=5, include_metadata=True)
            return [match['metadata']['text'] for match in results.get('matches', [])]
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
    def __init__(self, pinecone_index_name: str):
        """Initialize Travel Chatbot with memory and retrieval handler."""
        self.memory = TravelMemory()
        self.rag_handler = RAGHandler(pinecone_index_name=pinecone_index_name)
        self.groq_client = self._init_groq_client()

    def _init_groq_client(self) -> Groq:
        """Initialize Groq client, raising error on failure."""
        try:
            return Groq()
        except Exception as e:
            logging.error("Error initializing Groq client: %s", e)
            raise RuntimeError("Error initializing Groq client") from e

    def process_user_input(self, user_input: str) -> str:
        """Process user input, update memory, retrieve data, and generate a response."""
        try:
            self.memory.update_memory(user_input)
            memory_data = self.memory.get_memory()
            filtered_documents = self.rag_handler.retrieve_and_filter_documents(user_input, memory_data)
            response = self._generate_response(user_input, filtered_documents)
            self.memory.add_chat(user_input, response)
            return response
        except Exception as e:
            logging.error("Error processing user input: %s", e)
            return "I'm having trouble understanding that request."

    def _generate_response(self, user_input: str, filtered_documents: List[str]) -> str:
        """Generate a response using memory and the LLM based on filtered documents."""
        memory_data = self.memory.get_memory()
        memory_context = (
            f"Destination: {memory_data.get('destination', 'unknown')}, "
            f"Date: {memory_data.get('date', 'unspecified')}, "
            f"Budget: {memory_data.get('budget', 'unspecified')}, "
            f"Activity Preference: {memory_data.get('activity_preference', 'none')}, "
            f"Head Count: {memory_data.get('head_count', '0')}"
        )
        combined_input = f"User Query: {user_input}\nContext: {memory_context}\nRelevant Information: {filtered_documents}"
        
        try:
            completion = self.groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": combined_input}],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            return ''.join(chunk.choices[0].delta.content or "" for chunk in completion)
        except Exception as e:
            logging.error("Error generating response: %s", e)
            return "I’m having trouble generating a response."

    def clear_chat(self):
        """Clears chat history and memory."""
        self.memory.clear_memory()

class TravelChatbotCoordinator:
    def __init__(self):
        """Initialize Travel Chatbot Coordinator with input validation and rate limiting."""
        self.chatbot = TravelChatbot(pinecone_index_name=PINECONE_INDEX_NAME)
        self.request_count = 0
        self.MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", 100))

    def sanitize_input(self, input_text: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        return re.sub(r"[^\w\s,.!?'-]", "", input_text)

    def start_chat(self):
        """Begin chatbot interaction with security checks and rate limiting."""
        print("Hello! I'm your travel assistant.")
        while self.request_count < self.MAX_REQUESTS:
            user_input = input("You: ")
            user_input = self.sanitize_input(user_input)
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            response = self.chatbot.process_user_input(user_input)
            print("Bot:", response)
            self.request_count += 1
        else:
            print("Rate limit reached. Please try again later.")
