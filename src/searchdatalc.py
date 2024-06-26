import os
from dotenv import load_dotenv
import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, AzureOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

CHROMA_PATH = "./.chromadb"
COLLECTION_NAME = "hci-rag"
BATCH_SIZE = 160
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MODEL="gpt-4o"
TEMPREATURE = 0.2

# Update progress
def update_progress(status_str):
    print(status_str)

# Get embedding
def get_embedding(chunk_size=CHUNK_SIZE):
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

# Get LLM
def get_llm(model, temperature):
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
            azure_endpoint=azure_endpoint,
            model=model,
            temperature=temperature,)
    elif openai_api_key:
        update_progress("Using OpenAI.")
        return ChatOpenAI(model=model, temperature=temperature)
    else:
        raise EnvironmentError(
            "No valid OpenAI API credentials found. Please set the environment variables "
            "'AZURE_OPENAI_API_KEY' and 'AZURE_OPENAI_ENDPOINT' for Azure OpenAI or "
            "'OPENAI_API_KEY' for OpenAI."
        )

# Get response
def get_ai_response(llm, query, context):
    template = """
      You are a helfpul assistant. Answer the following question considering only the following context.

      Context:
      {rag_context}

      User question:
      {user_question}
      """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
      "rag_context": context,
      "user_question": query,
    })

# Main function
def main(chunk_size=CHUNK_SIZE, model="gpt-4o", temperature=0.2):
    # Parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-query", type=str, help="The user query text.")
    parser.add_argument("-chromapath", type=str, help="Chroma DB persistent folder path.", default=CHROMA_PATH)
    parser.add_argument("-collection", type=str, help="Chroma DB collection name.", default=COLLECTION_NAME)
    args = parser.parse_args()
    query_text = args.query
    chroma_path = args.chromapath
    collection_name = args.collection

    # Prepare the DB.
    embedding_function = get_embedding(chunk_size)
    chroma_db = Chroma(
        persist_directory=chroma_path, 
        embedding_function=embedding_function,
        collection_name=collection_name)

    # Search the DB.
    results = chroma_db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.2:
        print(f"Unable to find matching results. {results}")
        return
    
    print(f"Got Chroma DB search results with scores {results}!")
    
    # create the context string from the top 3 results
    rag_context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Get the LLM
    llm = get_llm(model=model, temperature=temperature)
    
    # Get AI response
    ai_response = get_ai_response(llm=llm, query=query_text, context=rag_context)
    print(ai_response)

    # print the sources
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"\nSources: {sources}"

    print(formatted_response)

if __name__ == "__main__":
        # get the response from the AI
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
