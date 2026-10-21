from pathlib import Path 
from typing import List
import os
import logging
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader
)

from langchain_core.documents import Document

logger=logging.getLogger(__name__)  #whenever logs wll get fcreated for this file it will be created with this file name


SUPPORTED_EXTENSION=['.pdf','.txt','.docx']
def load_document(file_path:str | Path)->List[Document]:
    """
    Load a single PDF, TXT or DOCX file into LangChain Documents.

    Different file formats need different readers, so this function picks the
    right LangChain loader based on the file extension. Keeping that choice in
    one place means the rest of the project never worries about file formats.

    Args:
        file_path: Path to the file on disk.

    Returns:
        A list of Documents. PDFs return one Document per page; TXT and DOCX
        files return a single Document for the whole file.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """

    path=Path(file_path) #converting the file path in the path so that we can perform some operation swhich path supports 
    extension=path.suffix.lower() #security_lower.pdf ->.pdf
    if extension not in SUPPORTED_EXTENSION:
        raise ValueError(f" unsupported file extension {extension}."
                         f" and suuported ones are :{','.join(sorted(SUPPORTED_EXTENSION))} ")

    if not path.exists():
        raise FileNotFoundError(f" file is not found here :{path}")

    if extension=='.pdf':
        loader=PyPDFLoader(str(path))
    elif extension=='.docx':
        loader=Docx2txtLoader(str(path))
    else:
        loader=TextLoader(str(path),encoding="utf-8")

    documents=loader.load()

    cleaned_documents=[]

    for doc in documents:
        doc.page_content=clean_text(doc.page_content)
        if doc.page_content.strip():
            cleaned_documents.append(doc)
    if not cleaned_documents:
        logger.warning("no readable text found in %s. Skipping it,",path.name)
    return add_metadata(cleaned_documents,path)

def load_documents_from_folder(folder_path:str |Path)->List[Document]:
    """
    Load every supported document inside a folder.

    This is what the "Build Vector DB" button and the build script use to read
    the whole `data/documents` folder in one go.

    Args:
        folder_path: Folder to scan (not recursive, one level only).

    Returns:
        A combined list of Documents from all readable files. Files that fail
        to load are skipped with a warning instead of stopping the whole run.
    """

    folder=Path(folder_path)
    if not folder.exists():
        logger.warning(f"folder path does not exists :%s",folder)
        return []
    all_documents:List[Document]=[]

    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSION:
            continue
        try:
            all_documents.extend(load_document(file_path))
        except Exception as e:
            logger.warning(f"could not load %s : %s",file_path.name,e)
    logger.info("loaded %d document pages from %s",len(all_documents),folder)
    return all_documents

def clean_text():
    pass

def add_metadata():
    pass