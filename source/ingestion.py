from pathlib import Path 
from typing import List
import os
import re
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

def clean_text(text:str)->str:
    """
    Remove the noise that PDF and DOCX extraction usually leaves behind.

    Raw extracted text is full of stray line breaks, repeated spaces and
    hyphenated words split across lines. Cleaning it first means the chunks we
    embed are closer to real sentences, which improves retrieval quality.

    Args:
        text: Raw text straight out of a document loader.

    Returns:
        The same text, tidied up.
    """
    if not text:
        return " "
    # Ex: "secu-\nrity" -> "security"
    #this all is done base on the business standards what the company requires and what not 
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() #white sopace removed 

    
    

def add_metadata(documents:List[Document],path:Path)->List[Document]:
    """
     Attach source information to each Document so we can cite it later.

    Retrieval returns a piece of text with no memory of where it came from.
    By storing the filename, file type and page number now, the final answer
    can say "technova_employee_handbook.pdf - Page 4".

    Args:
        documents: Documents produced by a loader.
        path: Path of the file they were loaded from.

    Returns:
        The same documents, with extra metadata keys filled in.
    """
    for document in documents:
        document.metadata["source"]=str(path)
        document.metadata["filename"]=str(path.name)
        document.metadata["doc_type"]=path.suffix.lower().lstrip(".")

        if "page" in document.metadata:
            document.metadata["page_number"]=int(document.metadata["page"])+1
    return documents

def guess_section(text:str)->str:
    """
    Guess the section heading a piece of text belongs to.

    Policy documents are written as "Password Policy", "Refund Policy" and so
    on. Capturing that heading lets a citation say "Section: Password Policy",
    which is far more useful to a reader than a filename alone.

    Args:
        text: A chunk of document text.

    Returns:
        The heading we found, or an empty string if the text has no obvious
        heading. We deliberately keep this simple: a wrong guess only affects
        the citation label, never the answer itself.


    """   

    for raw_line in text.split("\n")[:15]:
        line =raw_line.split()
        # A heading is short, has a handful of words and does not end like a
        # sentence. "4. PASSWORD POLICY" passes; "We reset passwords." fails.
        if not (3 < len(line) <= 70) or line.endswith((".", ",", ";", ":")):
            continue
        if len(line.split()) > 8:
            continue

        # Does this line start like a numbered list item?
        # Examples: "1. Hello", "2) World", "10.3.4 item"
        # It matches: digits, optional ".digits" chunks, optional "." or ")" after the number,
        # then at least one space, then a real word.
        looks_numbered = re.match(r"^\d+(\.\d+)*[.)]?\s+\S", line) is not None
        if line.isupper() or line.istitle() or looks_numbered:
            # Strip a leading number so "4. Password Policy" becomes
            # "Password Policy".
            return re.sub(r"^\d+(\.\d+)*[.)]?\s+", "", line).strip()
        
    return ""