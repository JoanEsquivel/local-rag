import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.query import query_rag
from ragas.metrics import Faithfulness
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import pytest
import json

# Faithfulness
# Definition: The response does not introduce external or incorrect information. - It only rephrases and summarizes what is found in the retrieved context.
# Low Faithfulness = The response adds information that is not supported by the retrieved context. It may alter key facts, leading to a misleading or incorrect answer.
# High Faithfulness = The response is accurate and directly supported by the retrieved context.


@pytest.mark.asyncio
async def test_faithfulness(langchain_llm_ragas_wrapper, get_question, print_log):

    # Get Question 
    question = get_question("questions", "faithfulness")
  
    # Get Response
    response = query_rag(question)
    parsed_response = json.loads(response)

    # Initialize the LLM and Ragas Setup for Context Precision 
    faithfulness = Faithfulness(llm=langchain_llm_ragas_wrapper)

    # Feed Data
    sample = SingleTurnSample(
        user_input=question,
        response=parsed_response["answer"],
        retrieved_contexts= [doc["page_content"] for doc in parsed_response["retrieved_docs"]],
    )

    # Score 
    score = await faithfulness.single_turn_ascore(sample)
    print_log(question, parsed_response["answer"], parsed_response["retrieved_docs"], score=score)
    assert score >= 0.5
  