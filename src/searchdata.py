import os
import argparse

from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
import chromadb.utils.embedding_functions as embedding_functions
import chromadb

# Load environment variables
load_dotenv()

CHROMA_PATH = "./.chromadb"
COLLECTION_NAME = "hci-rag"
MODEL= os.getenv('AZURE_OPENAI_API_DEPLOYMENT')

def get_llm_client():
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if azure_api_key and azure_endpoint:
        update_progress("Using Azure OpenAI.")
        return AzureOpenAI(
            api_key=azure_api_key, 
            azure_endpoint=azure_endpoint,
        )
    elif openai_api_key:
        update_progress("Using OpenAI.")
        return OpenAI(
            api_key=openai_api_key
        )
    else:
        raise EnvironmentError(
            "No valid OpenAI API credentials found. Please set the environment variables "
            "'AZURE_OPENAI_API_KEY' and 'AZURE_OPENAI_ENDPOINT' for Azure OpenAI or "
            "'OPENAI_API_KEY' for OpenAI."
        )    

def update_progress(status_str):
    print(status_str)

def get_embedding():
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_deployment = os.getenv('AZURE_OPENAI_API_DEPLOYMENT_EMBEDDING')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if azure_api_key and azure_endpoint:
        update_progress("Using Azure OpenAI Embedding.")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=azure_api_key,            
            api_base=azure_endpoint,
            api_type="azure",
            model_name="text-embedding-3-large",
            deployment_id=azure_deployment
        )
    elif openai_api_key:
        update_progress("Using OpenAI Embedding.")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-large"
        )
    else:
        raise EnvironmentError(
            "No valid OpenAI API credentials found. Please set the environment variables "
            "'AZURE_OPENAI_API_KEY' and 'AZURE_OPENAI_ENDPOINT' for Azure OpenAI or "
            "'OPENAI_API_KEY' for OpenAI."
        )

def get_llm():
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_deployment = os.getenv('AZURE_OPENAI_API_DEPLOYMENT')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_api_version = os.getenv('OPENAI_API_VERSION')

    if azure_api_key and azure_endpoint and azure_deployment and openai_api_version:
        update_progress("Using Azure OpenAI.")
        return AzureOpenAI(
            azure_deployment=azure_deployment,
            api_key=azure_api_key, 
            api_version=openai_api_version,
            azure_endpoint=azure_endpoint)
    elif openai_api_key:
        update_progress("Using OpenAI.")
        return OpenAI(
            api_key=openai_api_key
        )
    else:
        raise EnvironmentError(
            "No valid OpenAI API credentials found. Please set the environment variables "
            "'AZURE_OPENAI_API_KEY' and 'AZURE_OPENAI_ENDPOINT' for Azure OpenAI or "
            "'OPENAI_API_KEY' for OpenAI."
        )

# Get response
def get_response(llm, query, context, model):
    formatted_context = f"Context:\n\n{context}"
    formatted_question = f"User question: \n\n{query}"
    
    response = llm.chat.completions.create(
        model=model,
        messages=[
            { "role": "system", "content": "You are a helpful assistant. Answer the following question considering only the following context." },
            { "role": "user", "content": [
                {
                    "type": "text", 
                    "text": formatted_context 
                },
                { 
                    "type": "text",
                    "text": formatted_question
                }
            ] },
        ],
        max_tokens=2000,
    )

    return response

# Get ChromaDB
def get_chromadb(persist_directory):
    return chromadb.PersistentClient(path=persist_directory)

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-query", type=str, help="The query text.")
    parser.add_argument("-chromapath", type=str, help="The query text.", default=CHROMA_PATH)
    parser.add_argument("-collection", type=str, help="Chroma DB collection name.", default=COLLECTION_NAME)
    args = parser.parse_args()
    query_text = args.query
    chroma_path = args.chromapath
    collection_name = args.collection

    # Get LLM client.
    update_progress("Get LLM client.")
    client = get_llm_client()

    # Get embedding.
    update_progress("Get embedding function.")
    embedding_function = get_embedding()

    update_progress(f"Get chromadb from {chroma_path}.")
    chromadb = get_chromadb(persist_directory=chroma_path)

    update_progress(f"Get chromadb collection {collection_name}.")
    collection = chromadb.get_collection(name=collection_name, embedding_function=embedding_function)
    update_progress(f"Get collection {collection_name} with count {collection.count()}.")        

    # Search the DB.
    update_progress(f"Search chromadb collection for {query_text}.")
    results = collection.query(
        query_texts=query_text,
        n_results=3,
        include=['metadatas', 'documents', 'distances']
    )
    if len(results) == 0 or results['distances'][0][0] < 0.2:
        print(f"Unable to find matching results. {results}")
        return
    
    # create the context from the top 3 results
    rag_context = "\n\n---\n\n".join(doc for sublist in results['documents'] for doc in sublist)
    print(f"Got Chroma DB search results")

    # get the response from the AI
    ai_response = get_response(llm=client, query=query_text, context=rag_context, model=MODEL)
    print(ai_response.choices[0].message.content)
    
    # print the sources
    sources = [doc for sublist in results['metadatas'] for doc in sublist]
    formatted_response = f"\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")