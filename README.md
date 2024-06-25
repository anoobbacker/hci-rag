# Set up the project

## Set up conda env
1. Create virtual environment in Python using Conda

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
pip install sentence-tranformers
pip install wikipedia-api
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