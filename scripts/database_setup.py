#Used to handle command-line arguments in Python scripts.
import argparse
#Used to load environment variables from a .env file.
import os
from dotenv import load_dotenv
#Used to delete directories and their contents.
import shutil
#Used to load documents from a directory.
from langchain_community.document_loaders import PyPDFDirectoryLoader
#Used to split documents into chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter
#Used to represent documents in a structured format.
from langchain.schema.document import Document
#Used to get the embedding function.
from get_embedding_function import get_embedding_function
#Used to create a Chroma database.
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    # Load the environment variables from the .env file.
    load_dotenv()
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


#Used to load documents from a directory.
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


#Used to split documents into chunks.
# - Chunk size: The maximum number of characters in a chunk.
# - Chunk overlap: The number of characters to overlap between chunks.
# - Length function: A function that determines the length of a document.
# - Is separator regex: A boolean that determines if the separator is a regular expression.
# - Separator: The character or string to use as a separator.
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

#What is Chroma, and why is it used for this project?
#ChromaDB is an open-source vector database designed specifically for storing and retrieving embeddings efficiently. It allows you to perform semantic search and similarity-based retrieval using vector embeddings generated from text.


#Why?
# Optimized for Embeddings
# - Unlike traditional databases (PostgreSQL, MongoDB, etc.), ChromaDB is built specifically for vector search.
# - It uses techniques like FAISS (Facebook AI Similarity Search) to store and quickly search embeddings.

# Fast Similarity Search
# When you ask a question, ChromaDB finds semantically similar text chunks in milliseconds.
# Example: If you search for "How to make coffee?", ChromaDB retrieves "Steps to brew coffee" even if the wording is different.

# Runs Locally
# - Unlike cloud-based solutions (Pinecone, Weaviate), ChromaDB works entirely on your machine, making it:
# - Faster (no API calls)
# - Cheaper (no cloud storage costs)
# - Private (no data leaks)

# Simple to Use
# - It integrates easily with LangChain, making it perfect for your use case.


#Used to add documents to the Chroma database.
def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

# Used to calculate the chunk IDs.
def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    # - Page Source: The source of the page.
    
    
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

#Used to clear the database.
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()