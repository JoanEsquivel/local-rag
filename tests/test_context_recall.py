from scripts.query import query_rag
from ragas.metrics import LLMContextRecall
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import pytest
import json

# Definition: Measures how much of the total relevant information was retrieved.
# High Recall = Most of the relevant information available in the document was retrieved.
# Low Recall = The retrieval missed important relevant information.   




#User Input -> Query
# Response -> Response
# Reference -> Ground Truth
# Retrieved Context -> Top K retrieved documents



question = "What are the different types of cat coats and fur patterns"
reference = """Cats have various coat types and fur patterns. The main types include: 
        Tabby (Mackerel, Classic/Blotched, Ticked, Spotted), 
        Tortoiseshell and Calico (Tortie, Calico, Torbies), 
        Bicolor (Tuxedo, Cow/Moo, Black Mask, Turkish Van), 
        Solid Colors (Black, White, Blue, Chocolate, Cinnamon, Lilac), 
        Smoke and Shaded (Smoke Cats, Shaded Cats), 
        and Colorpoint (Siamese-Type Patterns with dark points on the ears, face, paws, and tail)."""   

response = query_rag(question)
parsed_response = json.loads(response)

print(response)

@pytest.mark.asyncio
async def test_context_recall(langchain_llm__ragas_wrapper):

    # Initialize the LLM and Ragas Setup for Context Precision 
    context_recall = LLMContextRecall(llm=langchain_llm__ragas_wrapper)


    # Feed Data
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts= [doc["page_content"] for doc in parsed_response["retrieved_docs"]],
        reference=reference
    )

    # Score 
    score = await context_recall.single_turn_ascore(sample)
    log = f"Question: {question}\nRetrieved Contexts: {parsed_response['retrieved_docs']}\nReference: {reference}\nScore: {score}"
    print(log)
    assert score >= 0.5
  

