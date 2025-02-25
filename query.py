import argparse
import json
from langchain_chroma import Chroma  # Updated import for Chroma from langchain-chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # Updated import for ChatOpenAI from langchain-openai
from dotenv import load_dotenv
import os

from get_embedding_function import get_embedding_function

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=8)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    messages = prompt_template.format_messages(context=context_text, question=query_text)
    
    # Instantiate the ChatOpenAI model. 
    model = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
        # If not set in your environment, you can hardcode your API key here:
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Use invoke() instead of directly calling the model.
    response = model.invoke(messages)
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
    print(json_output)

    return json_output

if __name__ == "__main__":
    main()
