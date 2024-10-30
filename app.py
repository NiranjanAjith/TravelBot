import streamlit as st
from chatbot import TravelChatbot  # Ensure this import matches your file structure

# Streamlit UI
def main():
    st.title("Travel Chatbot")

    # Initialize chatbot in session state to persist across requests
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = TravelChatbot()  # Create only once

    # Retrieve the chatbot instance
    chatbot = st.session_state.chatbot

    # Initialize chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize user input
    user_input = st.text_input("You: ", "")

    # Process user input when the Send button is clicked
    if st.button("Send") and user_input:
        response = chatbot.process_user_input(user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": response})

    # Display chat history
    for chat in st.session_state.chat_history:
        st.write(f"      You: {chat['user']}")
        st.write(f"Bot: {chat['bot']}")

if __name__ == "__main__":
    main()
