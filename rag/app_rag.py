import streamlit as st
import tempfile
import os
import sys

# allow project root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_chatbot import create_chatbot
from src.logger.logger import get_logger
from src.exception.custom_exception import CustomException

logger = get_logger(__name__)


# ---------------------------
# Streamlit Page Setup
# ---------------------------

st.set_page_config(
    page_title="Construction AI Assistant",
    page_icon="📄"
)

st.title("📄 Construction Document Chatbot")
st.write("Upload a construction document and ask questions about it.")

st.markdown("---")


# ---------------------------
# Session State
# ---------------------------

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------
# File Upload
# ---------------------------

uploaded_file = st.file_uploader(
    "Upload Construction PDF",
    type=["pdf"]
)


# ---------------------------
# Build RAG Pipeline
# ---------------------------

try:

    if uploaded_file and st.session_state.chatbot is None:

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Building knowledge base..."):

            logger.info("Creating RAG chatbot")
            st.session_state.chatbot = create_chatbot(pdf_path)

        st.success("Chatbot ready!")

except Exception as e:
    raise CustomException(e, sys)


# ---------------------------
# Ask Question
# ---------------------------

if st.session_state.chatbot:

    question = st.text_input("Ask a question")

    if question:

        try:

            result = st.session_state.chatbot.invoke({"question": question})

            raw_answer = result["answer"]

            # Handle OpenAI response object
            if hasattr(raw_answer, "content"):
                raw_answer = raw_answer.content

            if "Answer:" in raw_answer:
                answer = raw_answer.split("Answer:")[-1].strip()
            else:
                answer = raw_answer.strip()
                
            docs = result.get("documents", [])

            st.session_state.history.append((question, answer))

            st.markdown("### 🤖 Answer")
            st.success(answer)

            # Show sources
            if docs:
                st.markdown("### 📚 Sources")

                for d in docs[:3]:
                    page = d.metadata.get("page", "unknown")
                    st.write(f"Page {page}")

        except Exception as e:
            raise CustomException(e, sys)


# ---------------------------
# Chat History
# ---------------------------

if st.session_state.history:

    st.markdown("---")
    st.markdown("### 💬 Chat History")

    for q, a in reversed(st.session_state.history):

        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")