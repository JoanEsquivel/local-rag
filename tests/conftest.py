import pytest
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
import json


@pytest.fixture
def langchain_llm__ragas_wrapper():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    return langchain_llm

@pytest.fixture
def get_embeddings():
    return OpenAIEmbeddings()

@pytest.fixture
def get_question():

    def _get_question(file_name, key_name):
        with open(f"./tests/data/{file_name}.json", "r") as file:
            data = json.load(file)
            return str(data["questions"].get(key_name, ""))

    return _get_question

@pytest.fixture
def get_reference():
    def _get_reference(file_name, key_name):
        with open(f"./tests/data/{file_name}.json", "r") as file:
            data = json.load(file)
            return str(data["questions"].get(key_name, ""))
    return _get_reference

