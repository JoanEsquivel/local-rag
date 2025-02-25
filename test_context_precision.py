from query import query_rag
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import pytest
import json


#User Input -> Query
# Response -> Response
# Reference -> Ground Truth
# Retrieved Context -> Top K retrieved documents



# question 1 = "What is the scientific classification of the domestic cat" - Page 37
# question 2 = "What are the common vocalizations and their meanings in cat communication?" - Page 29 -30 
# question 3 = "What are the key physical characteristics of domestic cats" - Page 26

question = "TBD"


#Feedback:
# Score and retrived context analysis is not the best when there are repeated words in the document
# The score is not the best when the question is not well defined

response = query_rag(question)
parsed_response = json.loads(response)

print(response)

@pytest.mark.asyncio
async def test_context_precision():
    # Precision = Number of relevant documents retrieved / Total number of documents retrieved
    # For instance: "Causes of deforestation in the Amazon rainforest"
    # Retrieved docuemtns: 5 total of which 3 are relevant and 2 are irrelevant
    # Precision = 3 / 5 = 0.6 (60%)

    # Initialize the LLM and Ragas Setup for Context Precision 
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))
    langchain_llm = LangchainLLMWrapper(llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm)


    # Feed Data
    sample = SingleTurnSample(
        user_input=question,
        response=parsed_response["answer"],
        retrieved_contexts= [doc["page_content"] for doc in parsed_response["retrieved_docs"]],
    )

    # Score 
    score = await context_precision.single_turn_ascore(sample)
    await print(parsed_response["answer"], [doc["page_content"] for doc in parsed_response["retrieved_docs"]] ,score)
  

