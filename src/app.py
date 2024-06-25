import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# Constants
CHROMA_PATH = "./.chromadb"
LOCK_DIR = "./.locks"
COLLECTION_NAME = "hci-rag"
MODEL="gpt-4o"
TEMPREATURE = 0.2

# Load environment variables
load_dotenv()

def get_embedding():
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_deployment = os.getenv('AZURE_OPENAI_API_DEPLOYMENT_EMBEDDING')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if azure_api_key and azure_endpoint:
        return AzureOpenAIEmbeddings(
            azure_deployment=azure_deployment,
            api_key=azure_api_key, 
            azure_endpoint=azure_endpoint,
            model="text-embedding-3-large", 
            show_progress_bar=True, 
            chunk_size=50)
    elif openai_api_key:
        return OpenAIEmbeddings(
           api_key=openai_api_key,
           model="text-embedding-3-large",
           show_progress_bar=True,
           chunk_size=50)
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
        return AzureOpenAI(
            azure_deployment=azure_deployment,
            api_key=azure_api_key, 
            api_version=openai_api_version,
            azure_endpoint=azure_endpoint,
            temperature=0.2
            )
    elif openai_api_key:
        return ChatOpenAI(model="gpt-4o", temperature=0.2)
    else:
        raise EnvironmentError(
            "No valid OpenAI API credentials found. Please set the environment variables "
            "'AZURE_OPENAI_API_KEY' and 'AZURE_OPENAI_ENDPOINT' for Azure OpenAI or "
            "'OPENAI_API_KEY' for OpenAI."
        )

if not os.path.exists(LOCK_DIR):
    os.makedirs(LOCK_DIR)

def get_lock_file_path(function_name):
    """Generate a unique lock file path for a function."""
    return os.path.join(LOCK_DIR, f"{function_name}.lock")

def create_lock(lock_file_path):
    """Create a lock file atomically."""
    try:
        # Create a temporary file and move it to the lock file path
        with tempfile.NamedTemporaryFile(dir = LOCK_DIR, delete=False) as temp_file:
            temp_path = temp_file.name
        os.rename(temp_path, lock_file_path)
    except FileExistsError:
        # Handle the case where another thread/process created the lock
        pass

def remove_lock(lock_file_path):
    """Remove the lock file if it exists."""
    if os.path.exists(lock_file_path):
        os.remove(lock_file_path)

def is_locked(lock_file_path):
    """Check if the lock file exists."""
    return os.path.exists(lock_file_path)

def update_progress(progress_bar, status_text, progress, status_str):
    progress_bar.progress(progress)
    status_text.text(status_str)
    print(status_str)

# Create ChromaDB
def load_chromadb(progress_bar, status_text, chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME):
    if os.path.exists(chroma_path):
        update_progress(progress_bar, status_text, 25, f"Loading Chroma DB from {chroma_path}")
        chroma_db = Chroma(
            persist_directory=chroma_path,
            embedding_function=get_embedding(),
            collection_name=collection_name
        )
        update_progress(progress_bar, status_text, 100, f"ChromaDB created from {chroma_path}")
        return chroma_db

def setup_chromadb_lock(progress_bar, status_text, chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME):
    chroma_db = None
    lock_file_path = get_lock_file_path("create_chroma_db")
    if is_locked(lock_file_path):
        st.warning("Another user is currently creating the ChromaDB. Please wait for the process to finish.")
        update_progress(progress_bar, status_text, 100, f"Lock taken by another user. Please wait.")
        return chroma_db    
    try:
        create_lock(lock_file_path)
        if os.path.exists(chroma_path):
            chroma_db = load_chromadb(progress_bar,status_text, chroma_path, collection_name)
            # print("Total documents in DB: ", len(st.session_state.chromadb.get_all_documents()))
            update_progress(progress_bar, status_text, 100, f"Successfully created ChromaDB!")
        else:
           update_progress(progress_bar, status_text, 100, f"""
                           Failed to create ChromaDB! Folder {chroma_path} not found! Please check the path.
                           Run createchromadb.py to create the ChromaDB.
                           """)
    except Exception as e:
        st.error(f"Error: {e}")
        update_progress(progress_bar, status_text, 100, f"Failed to create ChromaDB! {e}")
        chroma_db = None
    finally:
        remove_lock(lock_file_path)
    
    return chroma_db
  
# Get response
def get_response(llm, query, context):
    template = """
      You are a helfpul assistant. Answer the following questions considering only the following context.

      Context:
      {rag_context}

      User question:
      {user_question}
      """    
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
      "rag_context": context,
      "user_question": query,
    })

st.set_page_config(page_title="AI Playground", page_icon="🧠", layout="wide")

st.title("AI Playground")

# Initialize session state if it doesn't exist
if "chromadb" not in st.session_state:
  with st.spinner('Loading data...'):
    progress_bar = st.progress(0)
    status_text = st.empty()
    st.session_state.chromadb = setup_chromadb_lock(progress_bar, status_text)
    st.rerun()

msgs = StreamlitChatMessageHistory()

# Get user input
user_query = st.chat_input(placeholder="Type something...")

if st.session_state['chromadb'] is not None and len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state["last_run"] = None
elif st.session_state['chromadb'] is None:
    msgs.clear()
    msgs.add_ai_message("ChromaDB is not loaded. Please wait for the process to finish.")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
  st.chat_message(avatars[msg.type]).write(msg.content)

if user_query is not None and user_query != "":
    
    # Add user message
    with st.chat_message("human"):
        st.write(user_query)
        msgs.add_user_message(user_query)
    
    # Search the DB.
    rag_results = st.session_state.chromadb.similarity_search_with_relevance_scores(user_query, k=3)

    print("RAG Results: ", len(rag_results))
    
    # Add response from AI
    with st.chat_message("ai"):
      ai_response = None
      try:
        if rag_results is None:
          str = "No documents found in the database."
          st.write(str)
          msgs.add_ai_message(str)
        elif len(rag_results) == 0:
          str = f"Unable to find matching results. Len={len(rag_results)}{rag_results[0][1]}"
          st.write(str)
          msgs.add_ai_message(str)
        else:
          # create the context from the top 3 results
          rag_context = "\n\n---\n\n".join([doc.page_content for doc, _score in rag_results])

          # get the LLM
          llm = get_llm()
          
          # get the response from the AI
          ai_response = st.write_stream(get_response(llm, user_query, rag_context))

          msgs.add_ai_message(AIMessage(ai_response))
      except Exception as e:
        str = f"Error: {e}"
        st.write(str)
        msgs.add_ai_message(str)

      
