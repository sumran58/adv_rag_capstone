import logging 
from typing import List,Tuple,Dict
import config 
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from llm_client import get_chat_model

logger =logging.getLogger(__name__)

MULTI_QUERY_PROMPT=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite search queries for a document search engine.\n"
            "Given a user question, write {num_variants} alternative versions "
            "of it that use different wording or synonyms but keep the same "
            "meaning.\n"
            "If conversation history is given, make each version standalone by "
            "replacing words like 'it' or 'that policy' with the real subject.\n"
            "Return one query per line. No numbering, no extra text.",
        ),
        (
            "human",
            "Conversation history:\n{history}\n\nUser question: {question}",
        ),
    ]    
)


def generate_query_variants(question:str,history:str=" ")->List[str]:
    """
     Ask the LLM to rewrite the question in a few different ways.

    Users and documents rarely use the same words. Someone asks about "time
    off" while the handbook says "annual leave", so a single search can miss
    the right chunk. Searching with several phrasings makes a miss far less
    likely.

    Passing the conversation history also fixes follow-up questions: "does it
    apply to enterprise customers?" is rewritten into something searchable
    like "does the refund policy apply to enterprise customers?".

    Args:
        question: The user's original question.
        history: Recent conversation text, or "" for the first question.

    Returns:
        A list of queries. The original question is always first, so the
        pipeline still works if the LLM call returns nothing useful.

    Example:
        "What is the refund policy?" ->
            ["What is the refund policy?",
             "How do refunds work?",
             "money back guarantee terms"]
    """
    queries=[question]
    try:
        chain=MULTI_QUERY_PROMPT | get_chat_model() | StrOutputParser()
        response=chain.invoke(
            {
                "question":question,
                "history":history or "(None)",
                "num_variants":config.NUM_QUERY_VARIANTS,
            }
        )
        for line in response.split("\n"):
            cleaned=line.strip(" -*0123456789.").strip()
            if cleaned and cleaned.lower() != question.lower():
                queries.append(cleaned)
    except Exception as e:
        # If the rewrite call fails we still have the original question, so
        # the user gets an answer instead of an error screen.
        logger.warning("Multi-query generation failed (%s). Using original query.", e)

    return queries[: config.NUM_QUERY_VARIANTS + 1]




def build_keyword_retriever(children: List[Document]) -> BM25Retriever:
    """
    Build a BM25 keyword retriever over the child chunks.

    Vector search understands meaning but can miss exact strings such as
    "SOC 2", "v2.4" or "TN-4410". BM25 is old-fashioned word matching and is
    excellent at exactly those cases, which is why we run both.

    Args:
        children: The small chunks stored in `doc_store.json`.

    Returns:
        A BM25Retriever ready to answer queries.
    """
    retriever=BM25Retriever.from_documents(children)
    retriever.k=config.BM25_TOP_K
    return retriever  


def search_vector_store(store: VectorStore, query: str, k: int) -> List[Document]:
    """
    Run one semantic (meaning-based) search against the vector database.

    Args:
        store: The FAISS or Pinecone vector store.
        query: One search query.
        k: How many chunks to return.

    Returns:
        Up to `k` child chunks, most similar first. Returns an empty list if
        the search fails, so one bad query cannot break the whole pipeline.
    """
    try:
        return store.similarity_search(query,k=k)
    except Exception as e:
        logger.warning("vector search failed for %s %s",query,e)  
        return []
     


def search_keywords(retriever: BM25Retriever, query: str) -> List[Document]:
    """
    Run one keyword (BM25) search over the child chunks.

    Args:
        retriever: The BM25 retriever built by `build_keyword_retriever`.
        query: One search query.

    Returns:
        The best keyword matches, most relevant first.
    """
    try:
        return retriever.invoke(query)
    except Exception as e :
        logger.logging("keyword search failed for %s %s",query,e)
        return []   

#so we are using hybrid search using semantic and keyword serach so botht the rtrievers will bring the list of releveant documents and then we have to combine the ranking of the retrivers based on the ranking of the document in their respective retrieval list . so we  use algorithm and one such commonalgorithm is rrf reciprocal rank fusion whoich ranks the algorithm based upon their ranks in both the retrieval list ... higher the rank higher the imprtance 
def fuse_results(ranked_lists: List[List[Document]], top_k: int) -> List[Document]:
    """
    Merge several result lists into one, using Reciprocal Rank Fusion (RRF).

    We now have many result lists (one per query, per search method) and need
    a single ranking. RRF is the simple, standard way to do this: a document
    scores 1 / (60 + its position) in every list it appears in, and the scores
    are added up. A chunk that several searches agree on rises to the top.

    Args:
        ranked_lists: Each inner list is one search result, best match first.
        top_k: How many documents to keep.

    Returns:
        A single de-duplicated list of documents, best first.

    Example:
        Chunk A is 1st in one list and 3rd in another:
            1/(60+0) + 1/(60+2) = 0.0328
        Chunk B is 1st in one list only:
            1/(60+0) = 0.0167
        So chunk A wins.
    """
    rrf_constant=60
    scores:Dict[str,float]={} 
    documents: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for position,document in enumerate(ranked_list):
            chunk_id=document.metadata.get("chunk_id",document.page_content[:80])
            scores[chunk_id]=scores.get(chunk_id,0.0)+1.0/(rrf_constant+position)
            documents[chunk_id]=document
        best_id=sorted(scores,key=lambda key:scores[key],reverse=True)[:top_k]
        return [documents[chunk_id] for chunk_id in best_id]




def hybrid_retrieve(
        queries: List[str],
        store: VectorStore,
        keyword_retriever: BM25Retriever,
) -> List[Document]:
    """
    Run every query through both search methods and fuse the results.

    This is the "Hybrid Retrieval" technique. With 4 queries we perform 8
    searches (4 semantic + 4 keyword) and merge them into one candidate list.

    Args:
        queries: The original question plus its rewritten variants.
        store: The vector database.
        keyword_retriever: The BM25 retriever.

    Returns:
        The fused list of candidate child chunks (at most HYBRID_TOP_K).
    """
    ranked_lists:List[List[Document]]=[]
    for query in queries:
        ranked_lists.append(search_vector_store(store, query, config.VECTOR_TOP_K))
        ranked_lists.append(search_keywords(keyword_retriever, query))

    return fuse_results(ranked_lists, config.HYBRID_TOP_K)



def recover_parents(
        children: List[Document],
        parent_lookup: Dict[str, Document],
) -> List[Document]:
    """
    Replace each retrieved child chunk with the larger parent chunk it came from.

    The small child chunk was perfect for *finding* the right place in the
    document, but it is often too short to *answer* from. Its parent holds the
    surrounding sentences the LLM needs, so we hand the parent over instead.

    Args:
        children: Child chunks returned by hybrid retrieval, best first.
        parent_lookup: Mapping of parent_id to parent Document.

    Returns:
        Parent documents in the same order, with duplicates removed. Two
        children from the same parent collapse into a single parent, which
        also stops us from sending the same text to the LLM twice.
    """    
    parents: List[Document] = []
    seen_parent_ids = set()

    for child in children:
        parent_id = child.metadata.get("parent_id")
        parent = parent_lookup.get(parent_id)

        if parent is None:
            # No parent found (for example an index built by an older
            # version), so we keep the child rather than losing the result.
            parents.append(child)
            continue

        if parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            parents.append(parent)

    return parents