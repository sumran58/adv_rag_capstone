import re 
import logging 
from collections import Counter
from functools import lru_cache
from typing import List
from langchain_core.documents import Document
import config 

logger=logging.getLogger(__name__)
_cross_encoder_works = True
_fallback_reason=""

# Words that appear in almost every sentence and so say nothing about
# relevance. Removing them stops "the" and "is" from inflating the score.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
}

@lru_cache(maxsize=1)
def get_cross_encoder():
    """
    Load the cross-encoder model, once per application run.

    The model is about 90 MB and is downloaded from Hugging Face the first
    time you use it, then cached on your machine. Loading it once and reusing
    it keeps every later question fast.

    Returns:
        A sentence-transformers CrossEncoder model.

    Raises:
        ImportError: If `sentence-transformers` is not installed.
        Exception: Anything PyTorch raises while loading, such as a missing
            DLL on Windows. The caller turns this into the simple reranker.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError("senetnece transformer not installed "
                          "(pip install sentence transformer)") from e
    logger.info("loading cross enoder %s",config.CROSS_ENCODER_MODEL)
    return CrossEncoder(config.CROSS_ENCODER_MODEL)

def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int | None = None,
) -> List[Document]:
    """
    Score every candidate document against the query and keep the best ones.

    Each document gets a `rerank_score` in its metadata so the next step
    (context refinement) and the debug panel can both see how confident the
    reranker was.

    Args:
        query: The user's original question.
        documents: Candidate documents from hybrid retrieval.
        top_k: How many documents to keep. Defaults to config.RERANK_TOP_K.

    Returns:
        The documents sorted from most to least relevant, trimmed to `top_k`.

    Example:
        Query: "How long is the refund window?"
        A chunk saying "Refunds are accepted within 30 days" scores about 7,
        while a chunk about office parking scores about -11.
    """
    global _cross_encoder_works, _fallback_reason

    if not documents:
        return []

    keep = top_k or config.RERANK_TOP_K

    # RERANKER=simple in .env skips PyTorch completely. Use it on machines
    # where you cannot (or do not want to) install sentence-transformers.
    if config.RERANKER == "simple" or not _cross_encoder_works:
        return keyword_rerank(query, documents, keep)

    try:
        model = get_cross_encoder()
        # The model expects pairs of (question, chunk text). It reads both at
        # the same time, which is what makes it more accurate than embeddings.
        pairs = [(query, document.page_content) for document in documents]
        scores = model.predict(pairs)
    except Exception as error:
        # Catching everything is deliberate: the failure can be a missing
        # package, a Windows DLL error, or a failed model download, and the
        # right response to all of them is the same - keep the app running.
        _cross_encoder_works = False
        _fallback_reason = str(error).split("\n")[0]
        logger.warning(
            "Cross-encoder unavailable (%s). Falling back to the simple "
            "keyword reranker. Set RERANKER=simple in .env to silence this.",
            _fallback_reason,
        )
        return keyword_rerank(query, documents, keep)

    for document, score in zip(documents, scores):
        document.metadata["rerank_score"] = float(score)

    return sort_by_score(documents, keep)


def keyword_rerank(query: str, documents: List[Document], keep: int) -> List[Document]:
    """
    Rerank by counting how many of the question's words appear in the chunk.

    This is the no-dependency fallback. It is much cruder than a cross-encoder
    (it understands words, not meaning), but it keeps the pipeline shape
    intact so the rest of the project behaves exactly the same way.

    Args:
        query: The user's original question.
        documents: Candidate documents from hybrid retrieval.
        keep: How many documents to return.

    Returns:
        The documents sorted by keyword overlap, trimmed to `keep`.
    """
    for document in documents:
        document.metadata["rerank_score"] = keyword_score(query, document.page_content)

    return sort_by_score(documents, keep)

def keyword_score(query: str, text: str) -> float:
    """
    Score one chunk by the share of question words it contains.

    The result is deliberately mapped onto the same rough scale a cross-encoder
    produces (about -5 for junk, about +5 for a strong match), so the context
    refinement step and MIN_RERANK_SCORE keep working without any changes.

    Args:
        query: The user's question.
        text: The chunk text.

    Returns:
        A score between about -5.0 and +5.8.

    Example:
        Query "refund policy enterprise customers" -> {refund, policy,
        enterprise, customers}. A chunk containing three of those four words
        scores 10 * 0.75 - 5 = +2.5, plus a small bonus for repeats.
    """
    query_words = tokenize(query)
    if not query_words:
        return 0.0

    # Count how often each query word appears, not just whether it appears.
    # Coverage decides the score; repeats only break ties, because a chunk
    # that mentions "refund" five times is usually more on-topic than one
    # that mentions it once.
    found = Counter(
        word for word in re.findall(r"[a-z0-9]+", text.lower()) if word in query_words
    )
    coverage = len(found) / len(query_words)
    repeats = min(sum(found.values()) - len(found), 10) * 0.08

    return 10.0 * coverage - 5.0 + repeats


def tokenize(text: str) -> set:
    """
    Split text into a set of meaningful lowercase words.

    Args:
        text: Any piece of text.

    Returns:
        A set of words, without punctuation, stop words or single characters.

    Example:
        "What is the refund policy?" -> {"refund", "policy"}
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if len(word) > 1 and word not in STOP_WORDS}


def sort_by_score(documents: List[Document], keep: int) -> List[Document]:
    """
    Sort documents by their `rerank_score`, highest first.

    Args:
        documents: Documents that already have a `rerank_score` in metadata.
        keep: How many of the best documents to return.

    Returns:
        The top `keep` documents, best first.
    """
    ranked = sorted(
        documents,
        key=lambda document: document.metadata.get("rerank_score", 0.0),
        reverse=True,
    )
    return ranked[:keep]


def reranker_status() -> str:
    """
    Describe which reranker is actually being used right now.

    The Streamlit sidebar shows this so a learner can tell at a glance whether
    they are getting real cross-encoder reranking or the simple fallback.

    Returns:
        A short status string.
    """
    if config.RERANKER == "simple":
        return "simple keyword (set by RERANKER=simple)"
    if not _cross_encoder_works:
        return f"simple keyword (cross-encoder failed: {_fallback_reason[:60]})"
    return f"cross-encoder ({config.CROSS_ENCODER_MODEL})"

