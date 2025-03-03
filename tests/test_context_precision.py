from scripts.query import query_rag
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import pytest
import json

# Definition: Measures how much of the retrieved context is actually relevant to answering the question.
# High Precision = Most of the retrieved contexts are useful for answering the query.
# Low Precision = Many of the retrieved contexts are irrelevant or only partially relevant.




#User Input -> Query
# Response -> Response
# Reference -> Ground Truth
# Retrieved Context -> Top K retrieved documents



# question 1 = "What is the scientific classification of the domestic cat" - Page 37
# question 2 = "What are the common vocalizations and their meanings in cat communication?" - Page 29 -30 
# question 3 = "What are the key physical characteristics of domestic cats" - Page 26

# Aspects to consider when testing context precision:
# 1- Retrieval query: It has to be optimized for the question - Think about the final user of the RAG system


# !!Factually Grounded Questions
#question = "What are the general physical characteristics of all cats, not specific breeds?"
# Score 0.49999
#Score (0.50) means the response was only 50% aligned with the retrieved context.
#Issue: The system retrieved both general traits and breed-specific traits, which diluted precision.
#Fix: Improve retrieval to exclude breed-specific chunks, and adjust response to only use retrieved data.

# Ambiguous Questions with Multiple Possible Answers
#question = "What is the most popular cat breed?"

# Niche Domain-Specific Queries
#question = "What are the common health issues in cats?"

# Temporal Context Questions
question = "How long do domestic cats typically live?"

# Confusable Entities
#question = "What is the function of a cat’s whiskers?"

# Comparative Questions
#question = "How do domestic cats and wild cats differ in behavior?"

# Complex Multi-Hop Queries
#question = "How does a cat’s sense of smell compare to its sense of sight?"

# Context Filtering in Multi-Document Retrieval
#question = "What is the role of the tail in a cat’s balance?"

# !!Counterfactual or Misleading Questions
#question = "Do all white cats have blue eyes? Retrieve facts specifically about eye color variation in white cats."
# Score 0.6388888888675925
#The score of 0.6389 indicates moderate precision, meaning that while the retrieved contexts were somewhat relevant, they weren’t fully optimized for the question.



# !!Step-by-Step Instruction Retrieval
#question = "How can I train my cat to use a litter box?"

#Score 0.0 
# Irrelevant or Partially Relevant Chunks
#   The retrieved chunks mostly discussed litter box types, materials, and maintenance rather than step-by-step training techniques.
#   There was some training information (about moving the litter box closer to the toilet), but it wasn’t comprehensive.
# Lack of Explicit Training Steps
#   The retrieved text mentioned that cats can be trained to use the toilet, but not explicitly how to train them to use a litter box from scratch.
#   The response inferred missing details, which did not directly match the retrieved context.
# Too Much External Knowledge Added
#   The answer contained general litter box training steps that may not have been fully grounded in the retrieved text.
#   ragas penalizes answers that use too much external knowledge if the provided context doesn’t fully support it.



#Feedback:
# Score and retrived context analysis is not the best when there are repeated words in the document
# The score is not the best when the question is not well defined

response = query_rag(question)
parsed_response = json.loads(response)

print(response)

@pytest.mark.asyncio
async def test_context_precision(langchain_llm__ragas_wrapper):
    # Precision = Number of relevant documents retrieved / Total number of documents retrieved
    # For instance: "Causes of deforestation in the Amazon rainforest"
    # Retrieved docuemtns: 5 total of which 3 are relevant and 2 are irrelevant
    # Precision = 3 / 5 = 0.6 (60%)

    # Initialize the LLM and Ragas Setup for Context Precision 
    context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm__ragas_wrapper)


    # Feed Data
    sample = SingleTurnSample(
        user_input=question,
        response=parsed_response["answer"],
        retrieved_contexts= [doc["page_content"] for doc in parsed_response["retrieved_docs"]],
    )

    # Score 
    score = await context_precision.single_turn_ascore(sample)
    log = f"Question: {question}\nResponse: {parsed_response['answer']}\nRetrieved Contexts: {parsed_response['retrieved_docs']}\nScore: {score}"
    print(log)
    assert score >= 0.5
  

