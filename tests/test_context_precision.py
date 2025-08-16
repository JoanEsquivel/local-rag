import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.query import query_rag
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas import SingleTurnSample
import pytest
import json

# Context Precision
# Definition: Measures how much of the retrieved context is actually relevant to answering the question.
# High Precision = Most of the retrieved contexts are useful for answering the query.
# Low Precision = Many of the retrieved contexts are irrelevant or only partially relevant.

@pytest.mark.asyncio
async def test_context_precision(langchain_llm_ragas_wrapper, get_question, print_log):

    # Get Question 
    question = get_question("questions", "simple")

    # Get Response
    response = query_rag(question)
    parsed_response = json.loads(response)

    # Initialize the LLM and Ragas Setup for Context Precision 
    context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm_ragas_wrapper)

    # Feed Data
    sample = SingleTurnSample(
        user_input=question,
        response=parsed_response["answer"],
        retrieved_contexts= [doc["page_content"] for doc in parsed_response["retrieved_docs"]],
    )

    # Score 
    score = await context_precision.single_turn_ascore(sample)
    print_log(question, parsed_response["answer"], parsed_response["retrieved_docs"], score=score)
    assert score >= 0.5

    