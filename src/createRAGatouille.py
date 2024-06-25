import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownTextSplitter
from ragatouille import RAGPretrainedModel

# Constants
DATA_PATH = "./data"
RAGATOUILLE_PATH = "./.ragatouille"
LOCK_DIR = "./.locks"
COLLECTION_NAME = "hci-rag"

# Load environment variables
load_dotenv()

# Load Markdown documents
def load_docs():
    loader = DirectoryLoader(DATA_PATH, glob="*.md", recursive=True, show_progress=True)
    return loader.load()

# Split documents into chunks
def split_docs(docs, chunk_size=1000, chunk_overlap=100):
    # Using chunk_overlap higher than chunk_size will give an error.
    # Failed to create ChromaDB! Got a larger chunk overlap (500) than chunk size (100), should be smaller.
    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def update_progress(status_str):
    print(status_str)

# Create ChromaDB
def create_trainer(ragatouille_path = RAGATOUILLE_PATH, data_path = DATA_PATH, index_name = COLLECTION_NAME, doc_chunk_size=160, model_name="colbert-ir/colbertv2.0"):
    update_progress(f"Loading RAG Pretrained model {model_name}...")
    RAG = RAGPretrainedModel.from_pretrained(model_name)

    update_progress(f"Loading documents from {data_path}...")
    docs = load_docs()
    if len(docs) == 0:
        update_progress("No documents found in the data directory.")
        return None
    
    update_progress(f"Splitting {len(docs)} documents into chunks...")
    split_documents = split_docs(docs)
    if split_documents is None or len(split_documents) == 0:
        update_progress("No chunks created.")
        return None
    
    update_progress(f"Split {len(docs)} documents into {len(split_documents)} chunks")
    
    update_progress(f"Creating ragatouille...")
    index_path = RAG.index(index_name=index_name, collection=split_documents)
    update_progress(f"Ragatouille created and saved to {ragatouille_path}")
    return index_path

def setup_ragatouille():
    ragatouille_index = None
    
    try:
        ragatouille_index = create_trainer()
        if ragatouille_index is not None:
            update_progress(f"Successfully created Ragatouille index!")
        else:
            update_progress(f"Failed to create Ragatouille index!")
    except Exception as e:
        update_progress(f"Failed to create Ragatouille index! {e}")
        chroma_db = None
    finally:
        pass # Nothing to do here
    
    return chroma_db

def main():
    setup_ragatouille()

if __name__ == "__main__":
    main()
