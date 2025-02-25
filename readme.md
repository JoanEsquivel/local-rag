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

## Usage

1- Run the database_setup.py file to feed the database with the documents under the data folder:
    ```bash
    python database_setup.py
    ```

2- Run the query.py file to test the RAG system:
    ```bash
    python query.py "your query here"
    ```

3- Examples of queries:

```bash
python query.py "what is the typical debugging process?"
```

```bash
python query.py "Based on the ISO/IEC 25010 standard, what is the classification of the non-functional quality characteristics?"
```


