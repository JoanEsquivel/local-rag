#Used to handle command-line arguments in Python scripts.
import argparse
#Used to handle JSON data in Python scripts.
import json
#Used to create a Chroma database.
from langchain_chroma import Chroma  
#Used to create a chat prompt template.
from langchain.prompts import ChatPromptTemplate
#Used to create a chat openAI model.
from langchain_openai import ChatOpenAI
#Used to load environment variables from a .env file.
from dotenv import load_dotenv
import os
#Used to get the embedding function.
from scripts.get_embedding_function import get_embedding_function

# Load the environment variables from the .env file.
load_dotenv()

# The path to the Chroma database.
CHROMA_PATH = "chroma"

# The prompt template for the chat openAI model.
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

#Used to get the query text from the command line.
def main():
    # Create CLI to get the query text. For example: python query.py "What is the capital of France?"
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB.
    # - Get the embedding function.
    # - Get the Chroma database from the database setup script.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    # - It returns a list of documents most similar to the query text and distance in float for each. Lower score represents more similarity.
    # - k is the number of results to return.
    # When to use more results?

    # ✅ If you need more context for a complex query
    # Example: A longer, detailed RAG response benefits from more documents.
    # ✅ If your dataset is small
    # If you have a small database, fetching more results can help avoid missing relevant context.
    # ✅ If retrieval accuracy is uncertain
    # A higher k gives your system more choices, allowing post-processing (e.g., ranking/filtering).

    # 🚨 Downsides of a High k
    # - Slower retrieval.
    # - Might return less relevant results.
    # - Can overload the LLM with unnecessary information.


    results = db.similarity_search_with_score(query_text, k=8)

    # - Join the page content of the results into a single string.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # - Create a chat prompt template.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # - Send the context and question to the model.
    messages = prompt_template.format_messages(context=context_text, question=query_text)
    
    # Instantiate the ChatOpenAI model. 
    model = ChatOpenAI(
        # - Temperature is the randomness of the model. 0 is deterministic, 1 is random.
        temperature=0,
        # - Model name. 
        model_name="gpt-4o-mini",
        # If not set in your environment, you can hardcode your API key here:
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Use invoke() instead of directly calling the model.
    # - invoke() ensures that the model correctly processes messages in the right format.
    # - In LangChain, invoke() is the recommended way to interact with AI models.
    # - It abstracts away complexities like token handling and formatting.  
    #  Better for Streaming & Batch Processing

    response = model.invoke(messages)
    # - Get the content of the response.
    # - Strip() removes any leading and trailing whitespace characters (spaces, tabs, etc.) from the string.
    response_text = response.content.strip()

    # Build a list of retrieved documents in the desired JSON structure
    retrieved_docs = []
    for doc, _ in results:
    # Here we assume the doc.metadata contains a key named "id" 
    # that uniquely identifies the file.
        file_name = doc.metadata.get("id", "Unknown File Name")
        retrieved_docs.append({
            "file_name": file_name,
            "page_content": doc.page_content
        })


    # Build the final JSON response
    output = {
        "answer": response_text,
        "retrieved_docs": retrieved_docs
    }

    # Convert to a JSON string for printing
    json_output = json.dumps(output, indent=2, ensure_ascii=False)
    #print(json_output)

    return json_output

if __name__ == "__main__":
    main()
