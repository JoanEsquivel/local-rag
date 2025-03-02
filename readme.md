# Project Title

Local RAG Setup for Testing Purposes

## Description

This project explores how to set up a local Retrieval-Augmented Generation (RAG) system for testing purposes. RAG is a powerful technique that combines retrieval-based and generation-based models to improve the quality and relevance of generated text.

## Table of Contents

- [OS requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)


## OS requirements

- Python 3.10+
- pip
- virtualenv
- git

- If you want to, create a local environment for this project:
    ```bash
    python -m venv local-ragas-env
    ```

- Activate the virtual environment:
    ```bash
    source local-ragas-env/bin/activate
    ```

- To deactivate the virtual environment:
    ```bash
    deactivate
    ```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/local-rag-setup.git
    ```
2. Navigate to the project directory:
    ```bash
    cd local-rag-setup
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. To execute the tests you need to set the PYTHONPATH environment variable, for instance:
    ```bash
    export PYTHONPATH="${PYTHONPATH}:/Users/joanesquivel/Desktop/personalRag"
    ```

5. Run the tests(-s is to print the print statements in the code - Optional):
    ```bash
    pytest tests/test_response_relevancy.py -s
    ```

## Environment Variables

You need to set the following environment variables under the .env file:

```
OPENAI_API_KEY="{your_openai_api_key}"
```


## RAG System Usage

I attached under the data folder a PDF document with some information about cats. My intention is to show how to use the RAG system to answer questions about the document and using a simple data set for testing purposes.

1- Run the database_setup.py file to feed the database with the documents under the data folder:
    ```bash
    python scripts/database_setup.py
    ```
You can have as many PDF documents as you want in the data folder. However, consider that you will need to conect to an LLM to generate the embeddings for the documents, and it may raise the cost of the API.

2- Run the query.py file to test the RAG system:
    ```bash
    python scripts/query.py "your query here"
    ```

3- Examples of queries:

```bash
python scripts/query.py "What is the most popular cat breed?"
```




