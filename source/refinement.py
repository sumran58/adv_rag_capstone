import logging 
from typing import List,Dict,Tuple
from langchain_core.documents import Document
import config 

logger=logging.getLogger(__name__)

def describe_source(document:Document)->str:
    """
    Build a short, human-readable label describing where a chunk came from.

    This label is shown under every answer, and is also written into the
    prompt so the model knows which document it is reading.

    Args:
        document: A chunk with metadata from the ingestion step.

    Returns:
        A string such as "technova_employee_handbook.pdf - Page 4" or
        "technova_security_policy.docx - Section: Password Policy".
        When neither a page nor a section is known, just the filename.
    """
    filename=document.metadata.get("filename","unknown_filename")
    page_number=document.metadata.get("page_number")

    if page_number:
        return f"{filename} - Page {page_number}"

    section = document.metadata.get("section")
    if section:
        return f"{filename} - Section: {section}"

    return filename



def drop_irrelevant(documents:List[Document])->List[Document]:
    """
    Remove chunks the cross-encoder scored below the relevance threshold.

    Retrieval always returns its top results, even when the documents contain
    nothing about the question. Feeding those to the LLM invites made-up
    answers, so we throw away anything that scored too low.

    Args:
        documents: Reranked documents carrying a `rerank_score`.

    Returns:
        Only the documents worth reading. If every document is below the
        threshold we keep the single best one, so the LLM can still say
        "the documents do not cover this" based on real text.
    """
    relevant=[
        document
        for document in documents 
        if document.metadata.get("rerank_score",0.0)>=config.MIN_RERANK_SCORE
    ]
    if not relevant and documents:
        logger.info("All chunks scored below the threshold; keeping the best one.")
        return documents[:1]

    return relevant
      




def apply_context_budget(documents:List[Document])->list[Document]:
    """
    Keep adding documents until the character budget runs out.

    Long prompts cost more, are slower, and actually make answers worse: a
    model asked to read ten pages often misses the one sentence that matters.
    Because the list is already sorted best-first, cutting from the end
    removes the least useful text.

    Args:
        documents: Relevant documents, best first.

    Returns:
        The documents that fit inside config.MAX_CONTEXT_CHARS.

    Example:
        With a 8000-character budget and 2500-character parents, the 4th
        document is dropped (3 x 2500 = 7500 fits, 10000 does not).
    """
    selected: List[Document] = []
    used_characters = 0

    for document in documents:
        length = len(document.page_content)
        if used_characters + length > config.MAX_CONTEXT_CHARS:
            continue
        selected.append(document)
        used_characters += length

    # Never return nothing: if even the first document is larger than the
    # whole budget, send a trimmed version of it rather than empty context.
    if not selected and documents:
        trimmed = Document(
            page_content=documents[0].page_content[: config.MAX_CONTEXT_CHARS],
            metadata=documents[0].metadata,
        )
        selected.append(trimmed)

    return selected

def format_context(documents:List[Document])->str:
     """
    Turn the selected documents into the single text block the LLM reads.

    Each chunk is numbered and labelled with its source. That numbering is
    what lets the prompt say "cite the source numbers you used".

    Args:
        documents: The final selected documents.

    Returns:
        A formatted string, for example:

            [1] Source: handbook.pdf - Page 2
            Employees receive 24 days of annual leave...

            [2] Source: support_policy.txt - Section: Refunds
            Refunds are issued within 30 days...
    """

     blocks = []
     for number, document in enumerate(documents, start=1):
            blocks.append(
            f"[{number}] Source: {describe_source(document)}\n{document.page_content}"
        )
     return "\n\n".join(blocks)

def refine_content(documents:List[Document])->Tuple[List[Document],str]:
    """
    Run the full refinement step: filter, budget, then format.

    This is the single function the pipeline calls. Keeping the three smaller
    steps separate makes each one easy to read, test and explain.

    Args:
        documents: Reranked documents.

    Returns:
        A tuple `(selected_documents, context_text)` where `context_text` is
        ready to drop straight into the prompt.
    """
    relevant = drop_irrelevant(documents)
    selected = apply_context_budget(relevant)
    return selected, format_context(selected)

