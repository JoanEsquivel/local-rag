
import os
#Used to get the embedding function.
from langchain_openai import OpenAIEmbeddings
#Used to load environment variables from a .env file.
from dotenv import load_dotenv


load_dotenv()

# What is an embedding? 
# An embedding is a way to turn text into numbers so that a computer can understand and compare it.
# - Words and sentences are just letters to a computer—it doesn't "understand" them like we do.
# - An embedding converts text into a set of numbers (a vector) that captures the meaning of the text.
# - Similar words or sentences will have similar numbers (vectors), making it easy to compare and find related content.

#  "cat" and "dog" will have closer embeddings than "cat" and "airplane" because they are more similar in meaning.

# Embeddings are used in many applications, including natural language processing, computer vision, and recommendation systems.


def get_embedding_function():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=openai_api_key)
    return embeddings