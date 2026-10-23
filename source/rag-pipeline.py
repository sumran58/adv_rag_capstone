import logging
from typing import Dict, List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm_client import get_chat_model
from source.chunking import build_parent_lookup
from source.indexing import load_doc_store, load_vector_store
from source.memory import format_history, get_recent_messages
from source.refinement import describe_source, refine_context
from source.reranking import rerank_documents
from source.retrieval import (
    build_keyword_retriever,
    generate_query_variants,
    hybrid_retrieve,
    recover_parents,
)

logger = logging.getLogger(__name__)