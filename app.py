import streamlit as st
from ChatModel import *

st.title("Code Llama Python Assistant")

# Load the ChatModel once and cache it for better performance
@st.cache_resource
def load_model():
    return ChatModel()

# Load the model
model = load_model()

# Sidebar for configuring model parameters
with st.sidebar:
    temperature = st.slider("Temperature", 0.0, 2.0, 0.1, help="Controls randomness.")
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9, help="Limits sampling to the top probability mass.")
    max_new_tokens = st.number_input("Max Tokens", 128, 4096, 256, help="Limits the response length.")
    system_prompt = st.text_area(
        "System Prompt",
        value=model.DEFAULT_SYSTEM_PROMPT,
        height=500,
        help="Customize the assistant's behavior by modifying this prompt.",
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept and process user input
if prompt := st.chat_input("Ask me anything!"):
    # Add user input to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from the model
    with st.chat_message("assistant"):
        user_prompt = st.session_state.messages[-1]["content"]
        answer = model.generate(
            user_prompt,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
        )
        st.markdown(answer)
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
