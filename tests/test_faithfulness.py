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
    question = get_question("faithfulness", "simple")
  
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
  

#2. Unfaithful (Potentially Hallucinated) Parts

# ------------------------------------------------------------------------------------------------
# Score: 0.8571428571428571

# Question: What are some unique physical traits and abilities of cats?

# Answer: Some unique physical traits and abilities of cats include:
# 1. **Retractable Claws**: Cats have claws that are sheathed in skin and fur when relaxed, which helps keep them sharp.
# They can extend their claws involuntarily by stretching or swatting at prey.
# 2. **Superior Night Vision**: Cats possess a tapetum lucidum, which enhances their ability to see in low light conditions,
# although it may reduce visual acuity in bright light.
# 3. **Acute Sense of Balance**: Cats can reflexively twist their bodies during a fall to right themselves,
# thanks to their acute sense of balance and flexibility.
# 4. **Digitigrade Locomotion**: Cats walk on their toes, with the bones of their feet making up the lower part of their visible legs.
# This allows them to walk quietly and precisely, minimizing noise and visible tracks.
# 5. **Whiskers**: Cats have about a dozen whiskers in four rows on each upper lip, which are sensitive and help them navigate their environment.
# 6. **Strong Sense of Smell**: A cat's sense of smell is about 14 times stronger than that of humans,
# aided by having twice as many smell-sensitive cells in their noses.
# 7. **Vomeronasal Organ**: Cats have a specialized scent organ in the roof of their mouths that enhances their ability to detect scents,
# which they access through a behavior called gaping.
# These traits contribute to their effectiveness as predators and their adaptability in various environments.

# Retrieved Context:

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:28:3
# Content: they are superior in many ways to those of humans. These along with the cat's highly 
# advanced eyesight, taste, and touch receptors make the cat extremely sensitive among 
# mammals. 
# Sight 
# Testing indicates that a cat's vision is superior at night in comparison to humans, and 
# inferior in daylight. Cats, like dogs, have a tapetum lucidum that reflects extra light to the 
# retina. While this enhances the ability to see in low light, it appears to reduce net visual 
# acuity, thus detracting when light is abundant. In very bright light, the slit-like iris closes very 
# narrowly over the eye, reducing the amount of light on the sensitive retina, and improving 
# depth of field. The tapetum and other mechanisms give the cat a minimum light detection

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:28:2
# Content: and visible tracks. 
# Like many predators, cats have retractable claws. This is actually a misnomer because in 
# their normal, relaxed position the claws are sheathed with the skin and fur around the toe 
# pads. This is done to keep the claws sharp by preventing wear from contact with the ground. 
# It is only by stretching, such as swatting at prey, that the connecting tendons are pulled taut, 
# forcing the claws to extend. Thus extending the claws is an involuntary action. 
# Senses 
# Measuring the senses of any animal can be difficult, because there is usually no explicit 
# communication (e.g., reading aloud the letters of a Snellen chart) between the subject and the 
# tester. 
# While a cat's senses of smell and hearing may not be as keen as, say, those of a mouse,

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:29:2
# Content: inches (7.5 cm) the location of a sound being made one yard (approximately one meter) 
# away. 
# Smell 
# A domestic cat's sense of smell is about 14 times stronger than a human's. Cats have twice 
# as many smell-sensitive cells in their noses as people do, which means they can smell things 
# we are not even aware of. Cats also have a scent organ in the roof of their mouths called the 
# vomeronasal, or Jacobson's, organ. When a cat wrinkles its muzzle, lowers its chin, and lets 
# its tongue hang a bit, it is opening the passage to the vomeronasal. This is called gaping. 
# Gaping is the equivalent of the Flehmen response in other animals, such as dogs and horses. 
# Touch 
# Cats generally have about a dozen whiskers in four rows on each upper lip, a few on each

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:27:1
# Content: gesturing. Because the domestication of the cat is relatively recent, cats may also still live 
# effectively in the wild, often forming small colonies. The cat's association with humans leads 
# it to figure prominently in the mythology and legends of several cultures, including the 
# ancient Egyptians, Vikings, and Chinese. 
# Characteristics 
# Physical 
# Cats typically weigh between 2.5 and 7 kg (5.5–16 lb); however, some breeds, such as the 
# Maine Coon can exceed 11.3 kg (25 pounds). Some have been known to reach up to 23 kg 
# (50 lb), due to overfeeding. This is very unhealthy for the cat, and should be prevented 
# through diet and exercise (playing), especially for cats living exclusively indoors. 
# In captivity, indoor cats typically live 15 to 20 years, though the oldest-known cat lived

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:28:1
# Content: A popular belief holds that cats always land on their feet. They do usually, but not always. 
# During a fall, a cat can reflexively twist its body and right itself using its acute sense of balance 
# and flexibility. [8] It always rights itself in the same way, provided it has the time to do so 
# during a fall. Certain breeds that don't have a tail are a notable exception, since a cat moves 
# its tail and relies on conservation of angular momentum to set up for landing. 
# Cats, like dogs, are digitigrades: they walk directly on their toes, the bones of their feet 
# making up the lower part of the visible leg. They are capable of walking very precisely, 
# placing each hind paw directly in the print of the corresponding forepaw, minimising noise 
# and visible tracks.

# ------------------------------------------------------------------------------------------------
# Analisis
# ------------------------------------------------------------------------------------------------
# It means the system did a pretty good job of sticking to the facts from the retrieved content but still has a little room for improvement.``
# ------------------------------------------------------------------------------------------------
# What the System Did Well
# ✔️ Most of the response is accurate and well-supported by the retrieved text.
# - Retractable claws: Confirmed in the retrieved content (File: 28:2).
# - Superior night vision: Verified in (File: 28:3), which explains the tapetum lucidum.
# - Digitigrade locomotion (walking on toes): Supported by (File: 28:1).
# - Vomeronasal organ (Jacobson’s organ): Verified in (File: 29:2).
# - Whiskers & strong sense of smell: Both are discussed in (File: 29:2).
# ✔️ No major fabrications – The response does not add false information.
# ✔️ The system correctly structured the answer by listing key traits and explaining them concisely.

# What can be improved?

# 1️⃣ Some details were slightly reworded or extrapolated beyond the retrieved text.
# - The description of retractable claws says they can extend “involuntarily,” but the retrieved text only mentions that claws are sheathed when relaxed and extend when stretching or swatting. While mostly correct, the phrasing could be a bit more precise.
# 2️⃣ Some abilities could have been expanded further
# - Sense of balance: The retrieved content (File: 28:1) discusses the righting reflex, but it doesn’t fully explain how balance works or its relation to tail movement.
# - Strong sense of smell: While the system got the general idea right, it could have included more details from the retrieved context (e.g., the comparison to other animals like mice).