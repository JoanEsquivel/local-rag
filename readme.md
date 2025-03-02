# Evaluate of RAG responses using RAGAS metrics & Pytest

Local RAG Setup for Testing Purposes

## Description

This project explores how to set up a local Retrieval-Augmented Generation (RAG) system for testing purposes. RAG is a powerful technique that combines retrieval-based and generation-based models to improve the quality and relevance of generated text.

## Table of Contents

- [OS requirements](#os-requirements)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [RAG System Usage](#rag-system-usage)
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
