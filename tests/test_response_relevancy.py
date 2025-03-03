from scripts.query import query_rag
from ragas.metrics import ResponseRelevancy
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
import pytest
import json

#  Response Relevancy measures how well the generated response answers the user’s question based on the retrieved context.
# ✅ High Response Relevancy
# The response fully answers the user’s question with clear, direct, and specific information.
# It uses the retrieved context correctly without adding unnecessary or unrelated details.




#User Input -> Query
# Response -> Response
# Reference -> Ground Truth
# Retrieved Context -> Top K retrieved documents



question = "How do cats communicate with humans and other cats?"
  

response = query_rag(question)
parsed_response = json.loads(response)

print(response)

@pytest.mark.asyncio
async def test_response_relevancy(langchain_llm__ragas_wrapper, get_embeddings):

    # Initialize the langchain wrapper and embeddings to be used for the response relevancy metric
    response_relevancy = ResponseRelevancy(llm=langchain_llm__ragas_wrapper, embeddings=get_embeddings)


    # Feed Data
    sample = SingleTurnSample(
        user_input=question,
        response=parsed_response["answer"],
        retrieved_contexts= [doc["page_content"] for doc in parsed_response["retrieved_docs"]],
    )

    # Score 
    score = await response_relevancy.single_turn_ascore(sample)
    log = f"Question: {question}\nResponse: {parsed_response['answer']}\nRetrieved Contexts: {parsed_response['retrieved_docs']}\nScore: {score}"
    print(log)
    assert score >= 0.5
  

# 2️⃣ Why Did the Response Get a Perfect Score (1.0)?

# ✅ All aspects of the response are explicitly supported by retrieved documents.
# ✅ No hallucinations (The response doesn’t introduce any extra, unsupported claims).
# ✅ No missing information (All major communication methods from the document are included).
# ✅ The response directly and fully answers the user’s question.

#If the response had been incomplete, vague, or contained incorrect details, the score would have dropped. Here’s how:

#| Issue | Example of Low-Scoring Response | Why It Lowers the Score? |
#|-------|--------------------------------|------------------------|
#| Too vague | "Cats use sounds and body movements to communicate." | Doesn't provide details on what sounds, what movements. |
#| Hallucinated information | "Cats use echolocation like bats to navigate at night." | Not in retrieved context and factually incorrect. |
#| Missing key information | "Cats communicate through meowing but not body language." | Leaves out body language and scent marking, making the answer incomplete. |
