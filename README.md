# Overview
- [src/app.py](src/app.py): Streamlit app to search local Chroma DB and provide results using LLM. This uses LangChain.
- [src/createchromadblc.py](src/createchromadblc.py): Creates a Chroma DB from local folder of Markdown files. This uses LangChain.
- [src/searchdatalc.py](src/searchdatalc.py): Search local Chroma DB based results and provide results using LLM. This use LangChain.
- [src/searchdata.py](src/searchdata.py): Search local Chroma DB based results and provide results using LLM. This doesn't use LangChain.

## Screenshot
![Terminal](images/image-01.png)
![Streamlit](images/image-02.png)

# Set up the project

## Set up conda env
1. Create virtual environment in Python using Conda. To install Miniconda, follow the instructions [here](https://docs.anaconda.com/miniconda/miniconda-install/).

1. Create a new environment
    ```shell
    conda create -n rag-gs
    ```
1. Activate a Virtual Environment on Windows
    ```shell
    conda activate rag-gs
    ```
## Install OpenAI Python client library
```shell
pip install numpy
pip install openai
pip install python-dotenv
pip install streamlit
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install unstructured
pip install nltk
pip install chromadb
```

## Download the Azure Stack HCI docs
To download the Azure Stack HCI docs, you can follow these steps:

1. Open your web browser and go to the [Azure Stack HCI docs repository](https://github.com/MicrosoftDocs/azure-stack-docs/).

1. Click on the green "Code" button located on the right side of the repository.

1. In the dropdown menu, select "Download ZIP".

1. Save the ZIP file to a location on your computer.

1. Once the download is complete, extract the contents of the ZIP file to a desired folder.

Now you have successfully downloaded the Azure Stack HCI docs. You can proceed with the next steps in your project.

## Set up your environment variable
1. Create a `.env` file
1. Add the following environment variables to the .env file
    ```env
    # Azure OpenAI API Key
    AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here

    # Azure OpenAI Endpoint
    AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here

    # Azure OpenAI Deployment for Embedding
    AZURE_OPENAI_API_DEPLOYMENT_EMBEDDING=your_azure_openai_deployment_embedding_here

    # OpenAI API Key
    OPENAI_API_KEY=your_openai_api_key_here
    ```
1. Save the `.env` file


## Create ChromaDB using Azure Stack HCI docs
1. Unzip the project files
1. Run the generate Chroma DB command
    ```shell
    python src/createchromadblc.py
    ```

## Check if the RAG is working
1. Run search data to see if RAG results are working properly.
    ```shell
    python src/searchdata.py
    ```

    If you want to validate using LangChain, use below:
    ```shell
    python src/searchdata.py
    ```

## Run streamlit
```shell
streamlit run ./src/app.py
```

# Disclaimer

> **Note:** Please note that there are currently a few outstanding issues with the quality of the local RAG (Retrieval-Augmented Generation) search results. I will be improving the accuracy and relevance of these results. Your understanding and patience are appreciated. This is a personal repository and provided "AS-IS" with no warranties or guarantees. 
