import logging

import streamlit as st
import traceback

import config
from source.indexing import (
    build_index,
    clear_index,
    index_exists,
    index_summary,
    update_index,
)
from source.ingestion import (
    SUPPORTED_EXTENSIONS,
    load_documents_from_folder,
)
from source.memory import add_exchange, clear_memory, create_memory
from source.rag_pipeline import answer_question, load_pipeline
from source.reranking import reranker_status

# Show our own INFO messages, but keep third-party libraries (httpx, faiss,
# groq
# ) at WARNING so the terminal stays readable during a demo.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("src").setLevel(logging.INFO)

st.set_page_config(page_title="Advanced RAG Assistant", page_icon="📚", layout="wide")
#st.session_state is a dictionary-like object in Streamlit that maintains variables across script reruns within a user's session. It is useful for maintaining application state, such as chatbot history, counter values, and user selections, without resetting them on every interaction."

def init_session_state() -> None:
    """
    Create the session values the app needs, the first time it runs.

    Streamlit reruns the script constantly, so without this guard we would
    wipe the chat history on every single click.
    """
    if "memory" not in st.session_state:
        st.session_state.memory=create_memory()
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    if "pipeline" not in st.session_state:
        st.session_state.pipeline=None

def get_pipeline():
    """
    Load the retrieval pipeline once and reuse it for every question.

    Opening the vector store and building the BM25 index on every question
    would make the app feel slow, so the loaded objects are cached in the
    session.

    Returns:
        The pipeline dictionary from `load_pipeline()`, or None if the index
        is missing or could not be opened.
    """
    if st.session_state.pipeline is None:
        try:
            with st.spinner("index is loading..."):
                st.session_state.pipeline=load_pipeline()
        except Exception as e:
            st.error(f"could not load the index:{e}")
            return None
    return st.session_state.pipeline

def save_uploaded_files(uploaded_files)->list:
     """
    Write files uploaded in the browser into `data/documents`.

    The LangChain loaders read from disk, so an uploaded file has to be saved
    before it can be processed. Keeping them in the same folder as the sample
    documents also means a future "Rebuild" picks them up automatically.

    Args:
        uploaded_files: Files returned by `st.file_uploader`.

    Returns:
        A list of the saved file paths.
    """
     config.DOCUMENT_DIR.mkdir(parents=True,exist_ok=True)
     saved_paths=[]
     for uploaded_file in uploaded_files:
         destination=config.DOCUMENT_DIR/uploaded_file.name
         destination.write_bytes(uploaded_file.get_buffer())    #to get the binary content and write_bytes is t write to the destination file 
         saved_paths.append(destination)
     return saved_paths

def render_sidebar() -> bool:
    """
    Draw the sidebar: configuration, document management and chat controls.

    Returns:
        True if the user switched on "Show Retrieval Details".
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Changing this only affects the running app. To make it permanent
        # (and to use the command-line scripts) edit VECTOR_DB in your .env.
        options = ["faiss", "pinecone"]
        chosen_db = st.selectbox(
            "Vector database",
            options,
            index=options.index(config.VECTOR_DB) if config.VECTOR_DB in options else 0,
            help="FAISS runs locally. Pinecone needs an API key in your .env file.",
        )
        if chosen_db != config.VECTOR_DB:
            config.VECTOR_DB = chosen_db
            st.session_state.pipeline = None

        for name, value in config.describe_settings().items():
            st.caption(f"**{name}:** {value}")

        # Shows "cross-encoder (...)" normally, or tells you the simple
        # keyword fallback is active because PyTorch could not be loaded.
        st.caption(f"**Reranker:** {reranker_status()}")

        st.divider()
        render_document_management()

        st.divider()
        st.header("💬 Chat controls")

        if st.button("Clear conversation", width="stretch"):
            clear_memory(st.session_state.memory)
            st.session_state.messages = []
            st.rerun()

        show_debug = st.toggle(
            "Show Retrieval Details",
            value=False,
            help="See the queries, retrieved chunks, reranked chunks and final context.",
        )

    return show_debug


def render_document_management() -> None:
    """
    Draw the document upload and index-building part of the sidebar.

    These three buttons map directly onto the three functions in
    `src/indexing.py`: build (from scratch), update (add new files) and clear.
    """
    st.header("📄 Documents")

    if index_exists():
        summary = index_summary()
        st.success(
            f"Index ready: {summary['children']} child chunks "
            f"from {len(summary['files'])} file(s)."
        )
        with st.expander("Indexed files"):
            for filename in summary["files"]:
                st.write(f"- {filename}")
    else:
        st.info("No index yet. Build one to start asking questions.")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Save uploaded files", width="stretch"):
        saved = save_uploaded_files(uploaded_files)
        st.success(f"Saved {len(saved)} file(s) to data/documents.")

    if st.button("🔨 Build Vector DB", width="stretch"):
        run_build()

    if st.button("➕ Update Vector DB", width="stretch"):
        run_update()

    if st.button("🗑️ Clear / Rebuild", width="stretch"):
        clear_index()
        st.session_state.pipeline = None
        st.success("Index cleared. Click 'Build Vector DB' to start again.")


def run_build() -> None:
    """
    Rebuild the whole index from everything in `data/documents`.

    Shows a spinner while groq embeds the chunks, because this is the one
    step that can take a while (and costs a few cents).
    """
    try:
        with st.spinner("Loading documents, chunking and embedding..."):
            documents = load_documents_from_folder(config.DOCUMENT_DIR)
            stats = build_index(documents)

        st.session_state.pipeline = None


        st.success(
            f"Indexed {stats['documents']} pages -> {stats['parents']} parent chunks "
            f"-> {stats['children']} child chunks."
        )
    except Exception as error:
        st.error(f"Build failed: {error}")


def run_update() -> None:
    """
    Add any new documents to the existing index without rebuilding it.

    Chunks that were already indexed are detected by their id and skipped, so
    clicking this twice does not create duplicates.
    """
    try:
        with st.spinner("Adding new documents to the index..."):
            documents = load_documents_from_folder(config.DOCUMENT_DIR)
            stats = update_index(documents)

        st.session_state.pipeline = None
        if "added_children" in stats:
            st.success(
                f"Added {stats['added_children']} new chunks "
                f"(skipped {stats['skipped_children']} already indexed)."
            )
        else:
            st.success(f"Built a new index with {stats['children']} chunks.")
    except Exception as error:
        st.error(f"Update failed: {error}")


def render_debug_panel(debug: dict) -> None:
    """
    Show what happened inside the pipeline for this question.

    This is the main teaching feature of the app: a learner can watch the
    candidate list shrink and reorder as it moves through hybrid retrieval,
    reranking and refinement.

    Args:
        debug: The "debug" dictionary returned by `answer_question`.
    """
    with st.expander("🔍 Retrieval details", expanded=False):
        st.markdown("**1. Original query**")
        st.write(debug["original_query"])

        st.markdown("**2. Generated queries (multi-query retrieval)**")
        for query in debug["generated_queries"]:
            st.write(f"- {query}")

        st.markdown(
            f"**3. Retrieved candidates after hybrid search "
            f"({len(debug['retrieved'])} chunks)**"
        )
        show_chunks(debug["retrieved"])

        st.markdown(
            f"**4. After cross-encoder reranking ({len(debug['reranked'])} chunks)**"
        )
        show_chunks(debug["reranked"])

        st.markdown(
            f"**5. Final context sent to the LLM ({len(debug['final_context'])} chunks)**"
        )
        show_chunks(debug["final_context"])


def show_chunks(chunks: list, limit: int = 5) -> None:
    """
    Print a short preview of a list of chunks in the debug panel.

    Args:
        chunks: Summary dictionaries with "source", "score" and "preview".
        limit: Maximum number of chunks to print, to keep the panel readable.
    """
    if not chunks:
        st.caption("_(none)_")
        return

    for number, chunk in enumerate(chunks[:limit], start=1):
        score = f" · score {chunk['score']}" if chunk["score"] is not None else ""
        st.caption(f"{number}. **{chunk['source']}**{score}")
        st.text(chunk["preview"][:300])

    if len(chunks) > limit:
        st.caption(f"_...and {len(chunks) - limit} more_")


def render_sources(sources: list) -> None:
    """
    Print the citation list underneath an answer.

    Args:
        sources: Citation strings such as "handbook.pdf - Page 4".
    """
    if not sources:
        return

    lines = [f"{number}. {source}" for number, source in enumerate(sources, start=1)]
    st.markdown("**Sources:**\n\n" + "\n".join(lines))


def main() -> None:
    """
    Draw the page and handle one chat turn per rerun.

    This is the entry point Streamlit executes. It keeps the layout simple:
    a sidebar for controls, a chat transcript in the middle, and an input box
    at the bottom.
    """
    init_session_state()

    st.title("📚 Advanced RAG Assistant")
    st.caption(
        "Hybrid retrieval · Multi-query · Parent-child chunks · "
        "Cross-encoder reranking · Context refinement"
    )

    show_debug = render_sidebar()

    if not config.GROQ_API_KEY:
        st.warning(
            "No groq API key found. Copy `.env.example` to `.env`, add your "
            "key, then restart the app."
        )
        return

    if not index_exists():
        st.info(
            "Nothing is indexed yet. Use **Build Vector DB** in the sidebar to "
            "index the sample TechNova documents (or upload your own first)."
        )
        return

    # Redraw the conversation so far. Streamlit forgets the screen on every
    # rerun, so we repaint it from st.session_state.messages each time.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))
                if show_debug and message.get("debug"):
                    render_debug_panel(message["debug"])

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    pipeline = get_pipeline()
    if pipeline is None:
        return

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and writing an answer..."):
            try:
                result = answer_question(question, pipeline, st.session_state.memory)
            except Exception as error:
                st.error(f"Something went wrong: {error}")
                st.code(traceback.format_exc())
                print(traceback.format_exc())
                return

        st.markdown(result["answer"])
        render_sources(result["sources"])
        if show_debug:
            render_debug_panel(result["debug"])

    # Save the turn in both places: the LLM's memory and the on-screen log.
    add_exchange(st.session_state.memory, question, result["answer"])
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "debug": result["debug"],
        }
    )


if __name__ == "__main__":
    main()
