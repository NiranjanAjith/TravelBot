---

# Travel Chatbot

A Retrieval-Augmented Generation (RAG)-based chatbot designed to assist users with travel queries by leveraging memory,
a vector database for contextual retrieval, and a large language model to generate contextually relevant responses.
This chatbot integrates multiple technologies, including Streamlit for UI, Pinecone for vector database storage and search,
Hugging Face for embeddings, and a Groq-powered language model.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Approach and Solution Design](#approach-and-solution-design)
3. [Architecture](#architecture)
4. [Setup and Testing](#setup-and-testing)
5. [Example Queries](#example-queries)

---

### 1. Project Overview

This project is a travel chatbot that guides users through travel planning, such as selecting destinations, suggesting activities, and budgeting for trips.
Using a Retrieval-Augmented Generation (RAG) architecture, the chatbot combines memory, contextual data retrieval, and generative responses.
Key components include:

- **Memory and Context**: Chatbot retains and uses context for coherent conversations.
- **Embedding-based Retrieval**: Uses Pinecone as a vector store to match contextually similar information.
- **Response Generation**: A Groq-powered LLM generates natural responses based on retrieved documents and contextual information.

---

### 2. Approach and Solution Design

This chatbot uses a hybrid approach, combining traditional rule-based memory with advanced vector-based retrieval and generative language models.

1. **Memory Management**: Captures user preferences and stores details like destinations, budget, and activities.
                             The `TravelMemory` class holds contextual memory across queries in a single session, improving coherence.
2. **Embedding Generation**: Uses a model hosted on Hugging Face to convert user input and memory data into vector representations.
                             These embeddings are used to query the Pinecone vector database.
3. **Contextual Retrieval**: By embedding the user input, the system queries relevant documents from the Pinecone index,
                             filtered to match the context and preferences stored in memory.
4. **Response Generation**: The Groq model processes the context-enriched input, creating responses based on the user's query, memory,
                             and relevant documents from the retrieval step.

---

### 3. Architecture

The architecture of this project can be described as a three-part Retrieval-Augmented Generation (RAG) solution:

- **Front-end/UI**:
  - Built with **Streamlit** to provide a simple, interactive UI for the user.
  - Handles session state to maintain chat history and memory.

- **Memory and Context Management**:
  - `TravelMemory` handles user memory, storing details like destination, budget, activities, and date preferences.
  - Updates with each interaction using Spacy's NER for recognizing relevant travel information in user input.
      - en_core_web_sm: This model is lightweight, efficient, and pre-trained to recognize general entities in English.
        It’s suitable for travel-related entities (e.g., locations, dates), enabling quick entity extraction without high computational overhead.
  - A better approach would be to use an LSTM model's long term memory for context retention and preference management. 

- **Retrieval and Generation Backend**:
  - **Embedding Generation**: Converts user input to embeddings via Hugging Face’s API, ensuring contextual similarity.
    -  sentence-transformers/all-MiniLM-L6-v2: Known for its efficiency and accuracy, this model provides 384-dimensional embeddings
        that balance performance and computational cost.It’s effective for encoding short text inputs, making it suitable for travel-related queries
        and memory data retrieval within Pinecone.
  - **Vector Database**: Embedding vectors are matched with stored documents in **Pinecone** to retrieve contextually relevant information.
    - Pinecone: speed, scalability, and ease of integration.Pinecone supports efficient storage and retrieval of high-dimensional vector embeddings,
      which is crucial for the performance of this RAG-based chatbot.
  - **Generative Model**: Groq-powered LLM receives context-enriched input and generates travel-related responses.
    - llama-3.1-70b-versatile: This model is chosen for its versatility, fine-tuned to handle various topics, including travel queries.
      The 70 billion parameters allow for nuanced, detailed responses while balancing response quality with manageable computational demands.

---

### 4. Setup and Testing

#### Prerequisites

Ensure the following packages are installed:
- Python 3.8+
- Streamlit for UI
- Pinecone for vector storage
- Hugging Face Transformers for embeddings
- Groq client SDK

#### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/NiranjanAjith/TravelBot.git
   cd travel-chatbot
   ```

2. **Set up Virtual Environment**
   ```bash
   python3 -m venv chatbot_env
   source chatbot_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   - Create a `.env` file with the following variables:
     ```
      CHAT_HISTORY_LIMIT=100
      PINECONE_API_KEY="pcsk_***8" 
      PINECONE_ENV="Server" 
      PINECONE_INDEX_NAME="chatbot"
      SPACY_MODEL="en_core_web_sm"
      GROQ_API_KEY="gsk_***q" 
      LLM_MODEL="llama-3.1-70b-versatile"
      HF_API_TOKEN="hf_***B"
      HF_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
     ```

#### Running the Application

1. **Setup Pinecone DB**
      The ragSetup.py script initializes your Pinecone database and sets up the necessary configurations.
       This needs to be done only once to ensure the Pinecone index is created and ready for document retrieval.
   ```bash
   python ragSetup.py
   ```
      
2. **Start Streamlit Application**
   ```bash
   streamlit run app.py
   ```

3. **Using the Chatbot**
   - Open your browser and go to `http://localhost:8501`.
   - Type in travel-related queries like *"Find a budget-friendly destination for a solo trip in November"*.
   - The chatbot will process and display a relevant response based on your input and stored context.

#### Troubleshooting
- **Errors** related to embedding generation may arise if the Hugging Face API token is invalid.
- **Pinecone Index Issues**: Ensure your Pinecone index is correctly set up with compatible dimensions.

---

### 5. Example Queries

1. **Budget-focused Query**:
   ```
   "What destinations are affordable for a weekend trip?"
   ```

2. **Activity-based Query**:
   ```
   "I love hiking and beaches. Any suggestions for an adventurous trip?"
   ```

3. **Location-specific Query**:
   ```
   "Suggest activities in Paris for a family of four."
   ```

4. **Multi-context Query**:
   ```
   "Find a romantic destination in Europe with a moderate budget for early December."
   ```

--- 
