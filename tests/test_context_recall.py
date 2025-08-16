import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.query import query_rag
from ragas.metrics import LLMContextRecall
from ragas import SingleTurnSample
import pytest
import json

# Context Recall
# Definition: Measures how much of the total relevant information was retrieved.
# High Recall = Most of the relevant information available in the document was retrieved.
# Low Recall = The retrieval missed important relevant information.   


@pytest.mark.asyncio
async def test_context_recall(langchain_llm_ragas_wrapper, get_question, get_reference, print_log):

    # Get Question 
    question = get_question("questions", "context_recall")

    # Get Reference
    reference = get_reference("questions", "context_recall_ground_truth")

    # Get Response
    response = query_rag(question)
    parsed_response = json.loads(response)

    # Initialize the LLM and Ragas Setup for Context Precision 
    context_recall = LLMContextRecall(llm=langchain_llm_ragas_wrapper)

    # Feed Data
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts= [doc["page_content"] for doc in parsed_response["retrieved_docs"]],
        reference=reference
    )

    # Score 
    score = await context_recall.single_turn_ascore(sample)
    print_log(question, parsed_response["answer"], parsed_response["retrieved_docs"], reference, score)
    assert score >= 0.5
  