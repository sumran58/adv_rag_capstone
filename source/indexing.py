import json 
import logging 
import shutil
from typing import List ,Tuple,Dict
from langchain_core.documents import Document
import config
from langchain_core.vectorstores import VectorStore
from llm_client import get_embedding_model
from source.chunking import split_documents
logger=logging.getLogger(__name__)

def _documents_to_dicts(documents:List[Document])->List[dict]:
    """
    Convert LangChain Documents into plain dictionaries so they can be saved.

    `json.dump` cannot write a Document object, but it happily writes
    {"page_content": "...", "metadata": {...}}.

    Args:
        documents: Documents to convert.

    Returns:
        A list of JSON-friendly dictionaries.
    """
    return [
        {"page_content":doc.page_content,"metadata":doc.metadata}
        for doc in documents
    ]


def _dicts_to_documents(items:List[dict])->List[Document]:
    """
    Convert dictionaries loaded from JSON back into LangChain Documents.

    This is the exact opposite of `_documents_to_dicts` and lets the rest of
    the project keep working with Document objects everywhere.

    Args:
        items: Dictionaries read from `doc_store.json`.

    Returns:
        A list of Documents.
    """
    return [
        Document(page_content=item["page_content"],metadata=item["metadata"])
        for item in items
    ]

def save_doc_store(parents:List[Document],children:List[Document])->None:
    """
    Write all parent and child chunks to `vectorstore/doc_store.json`.

    Keeping the chunks on disk means BM25 keyword search and parent lookup
    still work after you close and reopen the app, without re-reading and
    re-splitting every PDF.

    Args:
        parents: Large chunks used as LLM context.
        children: Small chunks that were embedded into the vector index.
    """
    config.VECTORSTORE_DIR.mkdir(parents=True,exist_ok=True)
    payload={
        "parents":_documents_to_dicts(parents),
        "children":_documents_to_dicts(children)

    }
    config.DOC_STORE_PATH.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    logger.info(
        "Saved %d parents and %d children to %s",
        len(parents),
        len(children),
        config.DOC_STORE_PATH,
    ) #SAVING THIS SO THAT WE CAN PERFROM BM25 LATER ON THIS SAVED JSON 

def load_doc_store()->Tuple[List[Document],List[Document]]:
    """
    Read the parent and child chunks back from `doc_store.json`.

    Returns:
        A tuple `(parents, children)`. Both lists are empty if the index has
        never been built, which the caller treats as "nothing indexed yet".
    """ 
    if not config.DOC_STORE_PATH.exists():
        return [],[]

    payload=json.loads(config.DOC_STORE_PATH.read_text(encoding="utf-8"))
    parents=_dicts_to_documents(payload.get("parents",[]))
    children=_dicts_to_documents(payload.get("children",[]))   
    return parents ,children 

def create_vector_store(children: List[Document]) -> VectorStore:
    """
    Embed the child chunks and store them in the configured vector database.

    This is the only function that knows the difference between FAISS and
    Pinecone. Everywhere else in the project just calls `load_vector_store()`
    and gets back something with a `.similarity_search()` method.

    Args:
        children: The small chunks to embed.

    Returns:
        A ready-to-search LangChain vector store.

    Raises:
        ValueError: If there is nothing to index or VECTOR_DB is unknown.
    """
    if not children:
        raise ValueError("there are no chunks to index . Add the documents to data/documents or upload a file before building the index ")
    embeddings=get_embedding_model()

    if config.VECTOR_DB=="faiss":
        store = _faiss_class().from_documents(children, embeddings)
        config.FAISS_DIR.mkdir(parents=True,exist_ok=True)
        store.save_local(str(config.FAISS_DIR))
        return store
    if config.VECTOR_DB=="pinecone":
        config.check_pinecone_config()
        PineconeVectoreStore=_pinecone_class()
        return PineconeVectoreStore.from_documents(
            children,
            embeddings,
            index_name=config.PINECONE_INDEX_NAME,
            namesapce=config.PINECONE_NAMESPACE

        )

    raise ValueError("Unknown vectoreb please use faiss or pinecone")

def load_vector_store() -> VectorStore:
    """
    Open an index that was built earlier.

    Called every time the app answers a question, so it must not rebuild
    anything - it only reopens what is already stored.

    Returns:
        A searchable LangChain vector store.

    Raises:
        FileNotFoundError: If the FAISS index has not been built yet.
        ValueError: If VECTOR_DB is set to an unknown value.
    """
    embeddings = get_embedding_model()    

    if config.VECTOR_DB == "faiss":
        if not (config.FAISS_DIR / "index.faiss").exists():
            raise FileNotFoundError(
                "No FAISS index found. Click 'Build VectorDB' in the sidebar or run: python scripts/build_index.py."
            )
        return _faiss_class().load_local(
            str(config.FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization = True,
        )

    if config.VECTOR_DB == "pinecone":
        config.check_pinecone_config()
        PineconeVectorStore = _pinecone_class()
        return PineconeVectorStore.from_existing_index(
            index_name = config.PINECONE_INDEX_NAME,
            embedding = embeddings,
            namespace = config.PINECONE_NAMESPACE,
        )

    raise ValueError(f"Unknown VECTOR_DB '{config.VECTOR_DB}'. Use faiss or piencone please.")


def _faiss_class():
    """
    Import the FAISS vector store class only when it is actually needed.

    Importing FAISS takes a moment and the library is only required when
    VECTOR_DB=faiss, so we import it inside a function instead of at the top
    of the file.

    Returns:
        The LangChain FAISS class.
    """
    from langchain_community.vectorstores import FAISS

    return FAISS

def _pinecone_class():
    """
    Import the Pinecone vector store class only when Pinecone is selected.

    FAISS users never install or import Pinecone this way, and if the package
    is missing they get a clear instruction instead of an ImportError.

    Returns:
        The LangChain PineconeVectorStore class.

    Raises:
        ImportError: If `langchain-pinecone` is not installed.
    """
    try:
        from langchain_pinecone import PineconeVectorStore
    except ImportError as e:
        raise ImportError("VECTOR_DB=pinecone needs the Pinecone packed, do pip install it.")    

def build_index(documents: List[Document]) -> Dict[str, int]:
    """
    Build the index from scratch: split documents, embed children, save all.

    Use this the first time, or whenever you want a clean rebuild. Anything
    that was indexed before is replaced.

    Args:
        documents: Documents loaded by `src.ingestion`.

    Returns:
        A small summary dictionary, for example
        {"documents": 12, "parents": 20, "children": 83}.

    Raises:
        ValueError: If no documents were given.
    """
    if not documents:
        raise ValueError("No documents to index available. Add files to data/documents first.")    
    
    parents, children = split_documents(documents)

    clear_index()

    create_vector_store(children)
    save_doc_store(parents, children)

    return {
        "documents": len(documents),
        "parents": len(parents),
        "children": len(children),
    }

def update_index(documents: List[Document]) -> Dict[str, int]:
    """
    Add new documents to an index that already exists.

    This is what happens when a learner uploads one extra PDF: we only split
    and embed the new file instead of re-embedding the whole library, which
    saves both time and OpenAI credits.

    Args:
        documents: Newly loaded documents to add.

    Returns:
        A summary dictionary with how many chunks were added and how many were
        skipped as duplicates.
    """
    existing_parents, existing_children = load_vector_store()    
    if not existing_children:
        return build_index(documents)
    
    new_parents, new_children = split_documents(documents)

    known_chunk_ids = {doc.metadata["chunk_id"] for doc in existing_children}
    known_parent_ids = {doc.metadata["parent_id"] for doc in existing_parents}

    children_to_add = [
        doc for doc in new_children if doc.metadata["chunk_id"] not in known_chunk_ids
    ]

    parents_to_add = [
        doc for doc in new_parents if doc.metadata["parent_id"] not in known_parent_ids
    ]

    if not children_to_add:
        return {"added_children": 0, "skipped_children": len(new_children)}
    
    store = load_vector_store()

    if config.VECTOR_DB == "pinecone":
        store.add_documents(children_to_add, namespace=config.PINECONE_NAMESPACE)
    else:
        store.add_documents(children_to_add)
        store.save_local(str(config.FAISS_DIR))

    save_doc_store(
        existing_parents + parents_to_add,
        existing_children + children_to_add,
    )

    return {
        "added_children": len(children_to_add),
        "skipped_children": len(new_children) - len(children_to_add),
    }

def clear_index() -> None:
    """
    Delete everything we have indexed so the next build starts clean.

    Useful when you change the embedding model, change the chunk sizes, or
    simply want to start over with a different set of documents.
    """    
    if config.FAISS_DIR.exists():
        shutil.rmtree(config.FAISS_DIR)

    if config.DOC_STORE_PATH.exists():
        config.DOC_STORE_PATH.unlink()

    if config.VECTOR_DB == "pinecone":
        _clear_pinecone_namespace()

    logger.info("index cleared.")

def _clear_pinecone_namespace() -> None:
    """
    Remove all vectors from the configured Pinecone namespace.

    FAISS is cleared by deleting a folder, but Pinecone lives in the cloud, so
    we ask Pinecone to delete the vectors instead.
    """
    try:
        from pinecone import Pinecone

        pinecone = Pinecone(api_key = config.PINECONE_API_KEY)    
        index = pinecone.Index(config.PINECONE_INDEX_NAME)
        index.delete(delete_all = True, namespace = config.PINECONE_NAMESPACE)

    except Exception as e:
        logger.warning("Could not clear Pinecone namespace: %s", e)


def index_exists() -> bool:
    """
    Check whether there is an index ready to answer questions.

    The Streamlit app uses this to decide between showing the chat box and
    showing "please build the index first".

    Returns:
        True if chunks have been indexed, False otherwise.
    """
    if not config.DOC_STORE_PATH.exists():
        return False
    
    if config.VECTOR_DB == "faiss":
        return (config.FAISS_DIR / "index.faiss").exists()

    return True

def index_summary() -> Dict[str, object]:
    """
    Describe what is currently in the index, for display in the sidebar.

    Returns:
        A dictionary with chunk counts and the list of indexed filenames.
    """
    parents, children = load_doc_store()    
    filenames = sorted({doc.metadata.get("filename", "?") for doc in parents})

    return {
        "parents": len(parents),
        "children": len(children),
        "files": filenames,
    }




