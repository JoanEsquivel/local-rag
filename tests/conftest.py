import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
import json


# Langchain LLM RAGAS Wrapper
# This is a wrapper that allows us to use the Langchain LLM with the RAGAS library.
@pytest.fixture
def langchain_llm_ragas_wrapper():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    return langchain_llm

# Get Embeddings
# This is neccesary for the response relevancy metric. It needs to be part of the metric analysis since it uses the embeddings to compare the response with the retrieved context.
@pytest.fixture
def get_embeddings():
    return OpenAIEmbeddings()

# Get Question
# This is a fixture that returns the question from the JSON file.
@pytest.fixture
def get_question():

    def _get_question(file_name, key_name):
        with open(f"./tests/data/{file_name}.json", "r") as file:
            data = json.load(file)
            return str(data["questions"].get(key_name, ""))

    return _get_question

# Get Reference
# This is a fixture that returns the reference from the JSON file.
@pytest.fixture
def get_reference():
    def _get_reference(file_name, key_name):
        with open(f"./tests/data/{file_name}.json", "r") as file:
            data = json.load(file)
            return str(data["questions"].get(key_name, ""))
    return _get_reference

import pytest

@pytest.fixture
def print_log():
    def _log(question, response, retrieved_contexts, reference=None, score=None):
        formatted_contexts = "\n".join([
            f"File: {context['file_name']}\nContent: {context['page_content']}\n"
            for context in retrieved_contexts
        ])
        log = "\n".join([
            "--------------------------------",
            "===== LOG START =====",
            f"Question: {question}",
            "---------------------",
            f"Response: {response}",
            "---------------------",
            "Retrieved Contexts:",
            formatted_contexts,
            "---------------------",
            f"Reference: {reference}" if reference else "Reference: None",
            "---------------------",
            f"Score: {score}" if score is not None else "Score: None",
            "====== LOG END ======"
            "--------------------------------"
        ])
        print(log)
    
    return _log

