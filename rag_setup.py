"""
RAG Vectorstore Setup
Run this once to create the vectorstore from your documents
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Configuration
DOCUMENTS_FOLDER = "./documents"
FAISS_INDEX_PATH = "./faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_documents():
    """Load all supported documents from the documents folder."""
    if not Path(DOCUMENTS_FOLDER).exists():
        print(f"Creating {DOCUMENTS_FOLDER} folder. Please add your documents there.")
        Path(DOCUMENTS_FOLDER).mkdir(exist_ok=True)
        return []
    
    loader_mapping = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.csv': CSVLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
    }
    
    all_documents = []
    supported_extensions = list(loader_mapping.keys())
    
    document_files = []
    for ext in supported_extensions:
        document_files.extend(Path(DOCUMENTS_FOLDER).glob(f"*{ext}"))
        document_files.extend(Path(DOCUMENTS_FOLDER).glob(f"**/*{ext}"))
    
    document_files = list(set(document_files))
    
    if not document_files:
        print(f"No supported documents found in {DOCUMENTS_FOLDER}")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return []
    
    print(f"Found {len(document_files)} document(s)")
    
    for doc_file in document_files:
        try:
            print(f"Loading {doc_file.name}...")
            loader_class = loader_mapping[doc_file.suffix.lower()]
            
            if doc_file.suffix.lower() == '.txt':
                loader = loader_class(str(doc_file), encoding='utf-8')
            else:
                loader = loader_class(str(doc_file))
            
            documents = loader.load()
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} page(s) from {doc_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error loading {doc_file.name}: {e}")
            continue
    
    return all_documents

def chunk_documents(documents):
    """Split documents into chunks."""
    print(f"\nChunking {len(documents)} documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    return chunks

def create_vectorstore(force_rebuild=False):
    """Create or load the vectorstore"""
    print("\n" + "="*60)
    print("RAG Vectorstore Setup")
    print("="*60 + "\n")
    
    embeddings = OpenAIEmbeddings()
    
    if Path(FAISS_INDEX_PATH).exists() and not force_rebuild:
        print(f"Vectorstore already exists at {FAISS_INDEX_PATH}")
        response = input("Rebuild from scratch? (y/n): ").strip().lower()
        if response != 'y':
            print("✓ Using existing vectorstore.")
            return
    
    documents = load_documents()
    if not documents:
        print("\n✗ No documents found. Please add documents to ./documents/ folder")
        return
    
    chunks = chunk_documents(documents)
    
    if not chunks:
        print("\n✗ No chunks created. Check your documents.")
        return
    
    print("\nCreating vector store (this may take a moment)...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"✓ Vector store saved to {FAISS_INDEX_PATH}")
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nYou can now run: python patient_chatbot.py")

if __name__ == "__main__":
    create_vectorstore()