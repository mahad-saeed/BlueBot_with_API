"""
Streamlit chat frontend for BlueBot. Calls the FastAPI backend's /chat endpoint.
Run: streamlit run streamlit_app.py
(Keep the FastAPI server running in a separate terminal: uvicorn src.api:app --reload)
"""

import requests
import streamlit as st

import os
API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

st.set_page_config(page_title="BlueBot — Airblue Assistant", page_icon="✈️")

st.markdown(
    """
    <style>
    .stApp { background-color: #f0f6ff; }
    h1 { color: #003580; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✈️ BlueBot")
st.caption("Ask about Airblue's baggage, fares, check-in, and travel policies.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

query = st.chat_input("Ask a question about Airblue policies...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json={"query": query}, timeout=30)
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])
            except requests.RequestException as exc:
                answer = f"Error reaching BlueBot backend: {exc}"
                sources = []

        st.write(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )