from scripts.query import query_rag
from ragas.metrics import ResponseRelevancy
from ragas import SingleTurnSample
import pytest
import json

# Response Relevancy
# Definition: Measures how well the generated response answers the user’s question based on the retrieved context.
# High Response Relevancy = The response fully answers the user’s question with clear, direct, and specific information.
# It uses the retrieved context correctly without adding unnecessary or unrelated details.


@pytest.mark.asyncio
async def test_response_relevancy(langchain_llm_ragas_wrapper, get_embeddings, get_question):

    question = get_question("response_relevancy", "simple")
  

    response = query_rag(question)
    parsed_response = json.loads(response)

    # Initialize the langchain wrapper and embeddings to be used for the response relevancy metric
    response_relevancy = ResponseRelevancy(llm=langchain_llm_ragas_wrapper, embeddings=get_embeddings)


    # Feed Data
    sample = SingleTurnSample(
        user_input=question,
        response=parsed_response["answer"],
        retrieved_contexts= [doc["page_content"] for doc in parsed_response["retrieved_docs"]],
    )

    # Score 
    score = await response_relevancy.single_turn_ascore(sample)
    log = f"Question: {question}\n -- \nResponse: {parsed_response['answer']}\n -- \nRetrieved Contexts: {parsed_response['retrieved_docs']}\n -- \nScore: {score}"
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
