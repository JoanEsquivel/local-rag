import pytest
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv


@pytest.fixture
def langchain_llm__ragas_wrapper():
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))
    langchain_llm = LangchainLLMWrapper(llm)
    return langchain_llm

@pytest.fixture
def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
