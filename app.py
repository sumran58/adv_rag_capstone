import logging

import streamlit as st

import config
from source.indexing import (
    build_index,
    clear_index,
    index_exists,
    index_summary,
    update_index,
)
from source.ingestion import SUPPORTED_EXTENSIONS, load_documents_from_folder
from source.memory import add_exchange, clear_memory, create_memory
from source.rag_pipeline import answer_question, load_pipeline
from source.reranking import reranker_status

# Show our own INFO messages, but keep third-party libraries (httpx, faiss,
# openai) at WARNING so the terminal stays readable during a demo.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("src").setLevel(logging.INFO)

st.set_page_config(page_title="Advanced RAG Assistant", page_icon="📚", layout="wide")