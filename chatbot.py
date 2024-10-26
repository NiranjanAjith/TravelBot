import os
import pinecone
import spacy
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class RAGHandler:
    def __init__(self, pinecone_index_name):
        """Initialize RAGHandler with Pinecone."""
        self.index_name = pinecone_index_name
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
        self.index = pinecone.Index(index_name=self.index_name)

    def retrieve_and_filter_documents(self, user_input, memory_data):
        """Retrieve relevant documents and apply filters based on user input context."""
        retrieved_documents = self.retrieve_documents(user_input)

        relevant_memory = {}
        if "stay" in user_input.lower() or "accommodation" in user_input.lower():
            relevant_memory['accommodation'] = memory_data.get('accommodation')
        if "transport" in user_input.lower():
            relevant_memory['transportation'] = memory_data.get('transportation')
        if "destination" in user_input.lower():
            relevant_memory['destination'] = memory_data.get('destination')
        if "budget" in user_input.lower():
            relevant_memory['budget'] = memory_data.get('budget')
        if "date" in user_input.lower():
            relevant_memory['date'] = memory_data.get('date')
        if "activity" in user_input.lower() or "interest" in user_input.lower():
            relevant_memory['activity_preference'] = memory_data.get('activity_preference')
        if "people" in user_input.lower() or "head count" in user_input.lower():
            relevant_memory['head_count'] = memory_data.get('head_count')

        relevant_memory = {k: v for k, v in relevant_memory.items() if v is not None}

        return self.filter_documents_by_memory(retrieved_documents, relevant_memory)

    def retrieve_documents(self, combined_query):
        """Uses Pinecone to find relevant documents based on the user query and memory context."""
        results = self.index.query(queries=[combined_query], top_k=5, include_metadata=True)
        if 'matches' not in results:
            return []

        documents = [match['metadata']['text'] for match in results['matches']]
        return documents

    def filter_documents_by_memory(self, documents, memory_data):
        """Filter retrieved documents based on dynamically extracted memory entities."""
        filtered_docs = []

        for doc in documents:
            matches = True
            
            for key, value in memory_data.items():
                if isinstance(value, str) and value.lower() not in doc.lower():
                    matches = False
                    break
                elif isinstance(value, (int, float)) and str(value) not in doc:
                    matches = False
                    break

            if matches:
                filtered_docs.append(doc)

        return filtered_docs


class TravelMemory:
    def __init__(self):
        """Initialize memory storage for the chatbot."""
        self.memory = {}
        self.chat_history = []
        self.nlp = spacy.load("en_core_web_sm")
        self.synonyms = {
            'destination': ['place', 'location', 'area', 'city', 'country', 'site', 'spot', 'region'],
            'budget': ['cost', 'price', 'amount', 'budget', 'spending', 'expense'],
            'activity_preference': ['interest', 'activity', 'hobby', 'thing to do', 'preferences'],
            'head_count': ['number of people', 'participants', 'count', 'group size', 'friends'],
            'date': ['when', 'day', 'time', 'period', 'date'],
            'transportation': ['travel', 'transport', 'vehicle', 'way to get there', 'commute'],
            'accommodation': ['stay', 'place to stay', 'hotel', 'lodging', 'accommodation', 'where to sleep'],
        }

    def update_memory(self, user_input):
        """Update memory based on user input using NER and additional processing."""
        doc = self.nlp(user_input)

        for ent in doc.ents:
            if ent.label_ == "GPE":
                self.memory['destination'] = ent.text
            elif ent.label_ == "DATE":
                self.memory['date'] = ent.text
            elif ent.label_ == "MONEY":
                budget_value = float(ent.text.replace('$', '').replace(',', ''))
                if budget_value > 0:
                    self.memory['budget'] = budget_value

        self.extract_activity_preference(user_input)
        self.extract_head_count(user_input)
        self.extract_transportation(user_input)
        self.extract_accommodation(user_input)

        self.clarify_missing_info(user_input)

    def extract_activity_preference(self, user_input):
        """Extract activity preference from user input, considering synonyms."""
        for synonym in self.synonyms['activity_preference']:
            if synonym in user_input.lower():
                self.memory['activity_preference'] = user_input.split(synonym)[-1].strip()
                return

    def extract_head_count(self, user_input):
        """Extract head count from user input using regex."""
        match = re.search(r'(\d+)\s*(?:people|persons|guests|friends)', user_input, re.IGNORECASE)
        if match:
            self.memory['head_count'] = int(match.group(1))
            return
        
        for synonym in self.synonyms['head_count']:
            if synonym in user_input.lower():
                head_count_str = user_input.split(synonym)[-1].strip()
                self.memory['head_count'] = int(head_count_str) if head_count_str.isdigit() else 0
                return

    def extract_transportation(self, user_input):
        """Extract transportation preferences from user input."""
        for synonym in self.synonyms['transportation']:
            if synonym in user_input.lower():
                self.memory['transportation'] = user_input.split(synonym)[-1].strip()
                return

    def extract_accommodation(self, user_input):
        """Extract accommodation preferences from user input."""
        for synonym in self.synonyms['accommodation']:
            if synonym in user_input.lower():
                self.memory['accommodation'] = user_input.split(synonym)[-1].strip()
                return

    def clarify_missing_info(self):
        """Ask for clarification if certain entities are missing."""
        if 'destination' not in self.memory:
            print("Could you please specify where you'd like to go?")
        if 'budget' not in self.memory:
            print("What is your budget for this trip?")
        if 'head_count' not in self.memory:
            print("How many people are you traveling with?")
        if 'activity_preference' not in self.memory:
            print("What activities are you interested in?")
        if 'date' not in self.memory:
            print("When do you plan to travel?")
        if 'transportation' not in self.memory:
            print("What transportation do you prefer?")
        if 'accommodation' not in self.memory:
            print("What type of accommodation do you prefer?")

    def get_memory(self):
        """Return current memory."""
        return self.memory

    def clear_memory(self):
        """Clear all memory."""
        self.memory.clear()

    def add_chat(self, user_input, bot_response):
        """Store chat interactions."""
        self.chat_history.append({"user": user_input, "bot": bot_response})

    def get_chat_history(self):
        """Return chat history."""
        return self.chat_history

    def update_memory_fact(self, key, value):
        """Dynamically update specific memory facts."""
        self.memory[key] = value


class TravelChatbot:
    def __init__(self):
        """Initialize the Travel Chatbot with RAG handler."""
        self.memory = TravelMemory()
        self.rag_handler = RAGHandler(pinecone_index_name="your_pinecone_index_name")
        self.groq_client = Groq()

    def process_user_input(self, user_input):
        """Process user input, update memory, retrieve relevant data, and generate a response."""
        self.memory.update_memory(user_input)

        memory_data = self.memory.get_memory()
        
        filtered_documents = self.rag_handler.retrieve_and_filter_documents(user_input, memory_data)

        response = self.generate_response(user_input, filtered_documents)

        self.memory.add_chat(user_input, response)
        return response

    def generate_response(self, user_input, filtered_documents):
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

        completion = self.groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": combined_input}],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        response = ''.join(chunk.choices[0].delta.content or "" for chunk in completion)
        return response

    def clear_chat(self):
        """Clears chat history and memory."""
        self.memory.clear_memory()

class TravelChatbotCoordinator:
    def __init__(self):
        """Initialize TravelChatbotCoordinator to manage the chatbot components."""
        os.environ["PINECONE_API_KEY"] = "your_pinecone_api_key"

        self.chatbot = TravelChatbot()

    def process_query(self, user_input):
        """Process a user query through the chatbot."""
        return self.chatbot.process_user_input(user_input)


if __name__ == "__main__":
    coordinator = TravelChatbotCoordinator()

    user_query = "I want to plan a trip to Paris on the 15th of July with a budget of $1000 and my friend John, who is interested in museums. The head count is 2."
    response = coordinator.process_query(user_query)
    print("Chatbot Response:", response)

