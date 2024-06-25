import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownTextSplitter

# Constants
DATA_PATH = "./data"
CHROMA_PATH = "./.chromadb"
COLLECTION_NAME = "hci-rag"
BATCH_SIZE = 160
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Load environment variables
load_dotenv()

# Load Markdown documents
def load_docs(data_path):
    loader = DirectoryLoader(data_path, glob="*.md", recursive=True, show_progress=True)
    return loader.load()

# Update progress
def update_progress(status_str):
    print(status_str)

# Get embedding
def get_embedding(chunk_size=50):
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_deployment = os.getenv('AZURE_OPENAI_API_DEPLOYMENT_EMBEDDING')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if azure_api_key and azure_endpoint:
        update_progress("Using Azure OpenAI Embedding.")
        return AzureOpenAIEmbeddings(
            azure_deployment=azure_deployment,
            api_key=azure_api_key, 
            azure_endpoint=azure_endpoint,
            model="text-embedding-3-large", 
            show_progress_bar=True, 
            chunk_size=chunk_size)
    elif openai_api_key:
        update_progress("Using OpenAI Embedding.")
        return OpenAIEmbeddings(api_key=openai_api_key,model="text-embedding-3-large", show_progress_bar=True, chunk_size=50)
    else:
        raise EnvironmentError(
            "No valid OpenAI API credentials found. Please set the environment variables "
            "'AZURE_OPENAI_API_KEY' and 'AZURE_OPENAI_ENDPOINT' for Azure OpenAI or "
            "'OPENAI_API_KEY' for OpenAI."
        )

# Load existing ChromaDB from path
def get_chroma_db_from_path(embedding, chroma_path = CHROMA_PATH):
    update_progress(f"Loading Chroma DB from {chroma_path}")
    chroma_db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding
    )
    update_progress(f"ChromaDB created from {chroma_path}")
    return chroma_db

# Split documents into chunks
def get_chunked_docs(docs, batch_size=160, chunk_size=1000, chunk_overlap=100):
        # Split documents into chunks
        # Using chunk_overlap higher than chunk_size will give an error.
        # Failed to create ChromaDB! Got a larger chunk overlap (500) than chunk size (100), should be smaller.
        update_progress(f"Start spliting {len(docs)} documents using splitter")
        text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        split_documents = text_splitter.split_documents(docs)
        if split_documents is None or len(split_documents) == 0:
            update_progress("No chunks created.")
            return None
                
        update_progress(f"Split {len(docs)} documents into {len(split_documents)} chunks")
      
        def split_list(input_list, chunk_size):
            for i in range(0, len(input_list), chunk_size):
                yield input_list[i:i + chunk_size]

        split_docs_chunked = split_list(split_documents, batch_size)
        
        update_progress(f"Completed split chunked!")

        return split_docs_chunked

# Create ChromaDB
def create_chroma_db(chunks, embedding, chroma_path = CHROMA_PATH, collection_name = COLLECTION_NAME):
    chroma_vectordb = None
    update_progress(f"Creating ChromaDB...")
    # The below gave an error: Failed to create ChromaDB! Batch size 2000 exceeds maximum batch size 166
    # chroma_vectordb = Chroma.from_documents(
    #     documents=split_documents, 
    #     embedding=embedding,
    #     collection_name=collection_name, 
    #     persist_directory=chroma_path)

    # Using the below as per https://github.com/chroma-core/chroma/issues/1049
    for chunk in chunks:
        chroma_vectordb = Chroma.from_documents( 
            documents=chunk,
            embedding=embedding,
            collection_name=collection_name,
            persist_directory=chroma_path,
            collection_metadata={"hnsw:space": "cosine"})          
    
    # the below also works!!!
    #   chroma_vectordb = Chroma(persist_directory=chroma_path, embedding_function=embedding, collection_name=collection_name)
    #   for i in range(0, len(split_docs), doc_chunk_size):
    #     chunk_docs = split_docs[i:i + doc_chunk_size]
    #     chroma_vectordb.add_documents(documents=chunk_docs)
    #   chroma_vectordb.persist() # Persist method in Chroma no longer exists in Chroma 0.4.x https://github.com/langchain-ai/langchain/issues/20851

    update_progress(f"ChromaDB created and saved to {chroma_path}")
    return chroma_vectordb

# Main function
def main(data_path=DATA_PATH, batch_size=BATCH_SIZE, chunk_size=BATCH_SIZE, chunk_overlap=CHUNK_OVERLAP, chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME):    
    # Get embedding
    embedding = get_embedding(batch_size)

    # Generate & store embedding in ChromaDB
    chroma_db = None
    if os.path.exists(chroma_path):
        #Load embedding from existing ChromaDB!
        chroma_db = get_chroma_db_from_path(embedding, chroma_path)
    else:
        # Load documents
        update_progress(f"Loading documents from {data_path}...")
        docs = load_docs(data_path)
        if len(docs) == 0:
            update_progress("No documents found in the data directory.")
            return
        
        # Chunk documents
        split_docs_chunked = get_chunked_docs(docs, batch_size, chunk_size, chunk_overlap)
        if split_docs_chunked is None:
            update_progress("No chunks created.")
            return
        
        # Create ChromaDB
        chroma_db = create_chroma_db(split_docs_chunked, embedding, chroma_path, collection_name)
    
    if chroma_db is None:
        update_progress("Failed to create ChromaDB!")
        return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        update_progress(f"Failed to create ChromaDB! {e}")
    finally:
        pass # Nothing to do here