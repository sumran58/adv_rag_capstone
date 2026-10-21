from functools import lru_cache
from langchain_groq import ChatGroq
import config
from langchain_huggingface import HuggingFaceEmbeddings
@lru_cache(max_size=1)
#lru cahcahe= when we call the ssame function with the same arguments we dont have to calculate the results again and again so here helps the lru cacche
def get_chat_model()->ChatGroq:
    """Create (or reuse) the OpenAI chat model used for every LLM call.

    The chat model does two jobs in this project: writing alternative versions
    of the user's question, and writing the final answer from the retrieved
    context.

    Returns:
        A configured LangChain ChatOpenAI model.

    Raises:
        ValueError: If OPENAI_API_KEY is missing."""

    config.check_api_key()
    return ChatGroq(
        model=config.GROQ_CHAT_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.GROQ_API_KEY
    )
@lru_cache(maxsize=1) #maxsize=1 stores at most 1 cached result.
def get_embedding_model()->HuggingFaceEmbeddings:
    """
    Create (or reuse) the Hugging Face embedding model used to build the vector index.

    An embedding converts text into a list of numbers (a vector).
    Texts with similar meanings get similar vectors, enabling semantic search.

    Returns:
        A configured LangChain Hugging Face embedding model.
    """
   
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )