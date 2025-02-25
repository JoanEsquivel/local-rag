import os
from langchain_openai import OpenAIEmbeddings  # Updated import
from dotenv import load_dotenv


load_dotenv()

def get_embedding_function():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return embeddings