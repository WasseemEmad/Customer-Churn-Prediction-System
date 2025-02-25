import streamlit as st
import requests

st.title("💬 Churn Prediction Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture user input
user_input = st.chat_input("Ask about a customer's churn prediction...")

if user_input:

    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send request to FastAPI backend for churn prediction
    response = requests.post("http://127.0.0.1:8000/chat", json={"customer_description": user_input})
    
    # Handle API response
    if response.status_code == 200:
        bot_response = response.json().get("bot_response", "I didn't understand that.")
    else:
        bot_response = "I'm having trouble connecting. Try again later."

    # Add chatbot response to session state and display it
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
