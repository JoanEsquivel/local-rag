from query import query_rag
from ragas.metrics import Faithfulness
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import pytest
import json

# Definition: The response does not introduce external or incorrect information. - It only rephrases and summarizes what is found in the retrieved context.
# Low Faithfulness = The response adds information that is not supported by the retrieved context. It may alter key facts, leading to a misleading or incorrect answer.
# High Faithfulness = The response is accurate and directly supported by the retrieved context.




#User Input -> Query
# Response -> Response
# Reference -> Ground Truth
# Retrieved Context -> Top K retrieved documents



question = "What are some unique physical traits and abilities of cats"
  

response = query_rag(question)
parsed_response = json.loads(response)

print(response)

@pytest.mark.asyncio
async def test_faithfulness():

    # Initialize the LLM and Ragas Setup for Context Precision 
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))
    langchain_llm = LangchainLLMWrapper(llm)
    faithfulness = Faithfulness(llm=langchain_llm)


    # Feed Data
    sample = SingleTurnSample(
        user_input=question,
        response=parsed_response["answer"],
        retrieved_contexts= [doc["page_content"] for doc in parsed_response["retrieved_docs"]],
    )

    # Score 
    score = await faithfulness.single_turn_ascore(sample)
    log = f"Question: {question}\nResponse: {parsed_response['answer']}\nRetrieved Contexts: {parsed_response['retrieved_docs']}\nScore: {score}"
    print(log)
    assert score >= 0.5
  

#2. Unfaithful (Potentially Hallucinated) Parts

# ❌ Sense of Smell & Vomeronasal Organ

# Response: "Their sense of smell is about 14 times stronger than that of humans, and they possess a vomeronasal organ that helps them detect scents."
# 📄 Retrieved Context: "A domestic cat's sense of smell is about 14 times stronger than a human's. Cats also have a scent organ in the roof of their mouths called the vomeronasal, or Jacobson's, organ." 【29:2】
# 🔸 Partially correct but slightly misleading:
# The 14x stronger smell is supported.
# However, the retrieved text does not explicitly say that the vomeronasal organ "helps them detect scents". It only describes its existence.
# Minor inference, reducing faithfulness slightly.

# ❌ Grooming Needs

# Response: "Cats have a grooming behavior that is essential for their health, and while they groom themselves, they may also benefit from regular brushing due to their fur length and quantity."
# 📄 Retrieved Context: "Grooming, but due to the length and quantity of hair, most will also benefit from a simple brushing once a week." 【136:1】
# 🔸 Mostly correct, but minor issue:
# The retrieved text mentions brushing for long-haired cats, but not all cats need it.
# The phrase "essential for their health" is not explicitly mentioned in the retrieved text, though it's implied.

# 3. Why the Faithfulness Score is 0.89 (Not 1.0)?

# 🔹 Most of the response is strongly supported by the retrieved data.
# 🔹 Two minor issues (vomeronasal organ function & generalization of grooming behavior) slightly lowered the score.
# 🔹 No major hallucinations, but small extrapolations exist.
