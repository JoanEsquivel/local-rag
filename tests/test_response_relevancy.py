import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
async def test_response_relevancy(langchain_llm_ragas_wrapper, get_embeddings, get_question, print_log):

    # Get Question  
    question = get_question("questions", "response_relevancy")

    # Get Response
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
    print_log(question, parsed_response["answer"], parsed_response["retrieved_docs"], score=score)
    assert score >= 0.5
  

# ------------------------------------------------------------------------------------------------
# Score: 0.9999999999999997

# Question: How do cats communicate with humans and other cats?

# Answer: 
# Cats communicate with humans and other cats through a variety of vocalizations, body language, and scent marking.

# 1. **Vocalizations**: Cats can produce about 100 different sounds, including meows, purrs, and other calls. 
# They use these sounds to indicate pain, request attention (such as being fed or played with), or as a greeting. 
# The pitch and tone of their calls can change as they age, starting from high-pitched squeaks in kittens to deeper sounds in adults.
# 2. **Body Language**: Cats use non-verbal cues to communicate their feelings and needs. 
# For example, whiskers pointing forward indicate curiosity and friendliness, while whiskers lying flat suggest defensiveness or aggression. 
# Other behaviors include rubbing against humans for affection, making eye contact to signal a need, and performing tricks similar to dogs.
# 3. **Scent Marking**: Cats have scent glands around their mouths and other areas, which they use to mark their humans as part of their territory. 
# This behavior is a way of claiming their human and establishing a bond.
# Overall, cats exhibit a combination of vocal and non-verbal communication methods to express their needs and emotions to both humans and other cats.

# Retrieved Context:

# Retrieved Contexts:
# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:36:2
# Content: a trip to the litter box before bedtime and snuggling up close to its companion in bed or on 
# the sofa. Oth er behaviors could include mimicking sounds of the owner or using certain 
# sounds the cat picks up from the human; sounds representing specific needs of the cat, which 
# the owner would recognize. The cat may also be capable of learning to communicate with 
# the human using non -spoken language or body language  such as rubbing for affection 
# (confirmation), facial expressions and making eye -contact with the owner if something 
# needs to be addressed (e.g . finding a bug crawling on the floor for the owner to get rid of). 
# Some owners like to train their cat to perform "tricks" commonly exhibited by dogs such as 
# jumping.

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:36:1
# Content: communications skills are required of the lone hunter. Thus, communicating with s uch an 
# animal is problematic, and cats in particular are labelled as opaque or inscrutable, if not 
# obtuse, as well as aloof and self -sufficient. However, cats can be very affectionate towards 
# their humans, especially if they imprint on them at a very young  age and are treated with 
# consistent affection. 
# Human attitudes toward cats vary widely. Some humans keep cats for companionship as 
# pets. Some people (known as cat lovers) go to great lengths to pamper their cats, sometimes 
# treating them almost as if they were children. When a cat bonds with its human owner, at 
# times, the cat may display behaviors similar to that of the human. Such behavior may include

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:30:2
# Content: depending on meaning. Usually cats call out to indicate pain, request human attention (to be 
# fed or played with, for example), or as a greeting. Some cats are very vocal, and others rarely 
# call out. Cats are capable of about 100 different vocalisations, compared to about 10 for dogs. 
# A kitten's call first starts out as a high -pitched squeak-like sound when very young, and 
# then deepens over time. Some cats, h owever, do not exercise their voices a lot, so their call 
# may remain similar to that of a kitten through adulthood. 
# Cats can also produce a purring noise that typically indicates that the cat is happy, but 
# also can mean that it feels distress. Cats purr among other cats—for example, when a mother

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:30:0
# Content: NICOLAE SFETCU: ABOUT CATS 
# 29 
# Whiskers are also an indication of the cat's attitude. Whiskers point forward when the 
# cat is inquisitive and friendly, and lie flat on th e face when the cat is being defensive or 
# aggressive. 
# Taste 
# According to National Geographic (December 8), cats cannot taste sugary foods due to a 
# faulty sweet receptor gene. Some scientists believe this is related to the cat's diet being 
# naturally high in protein, though it is unclear whether it is the cause or the result of it. 
# Communication 
# The unique sound a small cat makes is written onomatopoeically as "meow" in American 
# English; "meow" or "miaow" in British English; "miaou" or "miaw" in French; "miao " in 
# Mandarin Chinese and Italian; "miau" in German, Spanish, Finnish, Lithuanian, Polish,

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:93:2
# Content: food. It is also a way of 'marking' its human as its very own. Using scent glands located around 
# its mouth and elswhere, it subtly 'marks' its human as part of its cat territory. Most cats 
# prefer gentle rubs behind the ears. To inform their humans they need petting or attention, a 
# cat may push its entire body weight up against the human as the cat snuggles next to his/her 
# favorite person. 
# Some subtle Anthropomorphisms 
# • Disgust - Lifting and subsequent shaking of a paw or paws is sign of disgust. The 
# more paws the more disgusting. This can sometimes be a four paw affair wit h 
# each paw being lifted and shaken before the other. 
# • Agitation - The swishing or sweeping of the tail in one full 180 degree swoop mid-

# ------------------------------------------------------------------------------------------------
# Analisis  
# ------------------------------------------------------------------------------------------------
# The score is nearly perfect (1.0), meaning the system accurately used retrieved information to generate its response without adding false or misleading details.
# ------------------------------------------------------------------------------------------------
# What Went Well?
# ✔️ The response aligns perfectly with the retrieved content
# Vocalizations (100 different sounds, meowing for attention, pitch changes with age) → Verified in (File: 30:2).
# Body language (whisker positioning, rubbing for affection, eye contact) → Matches (File: 36:2 & 30:0).
# Scent marking (cats marking their humans as territory) → Directly mentioned in (File: 93:2).
# ✔️ No Fabrications or Overgeneralizations
# The system did not introduce any new or incorrect facts—everything it stated came directly from the retrieved text.
# ✔️ Good Structure & Readability
# The answer organizes information clearly, making it easy to understand without just copying raw text from the source.


#If the response had been incomplete, vague, or contained incorrect details, the score would have dropped. Here’s how:

#| Issue | Example of Low-Scoring Response | Why It Lowers the Score? |
#|-------|--------------------------------|------------------------|
#| Too vague | "Cats use sounds and body movements to communicate." | Doesn't provide details on what sounds, what movements. |
#| Hallucinated information | "Cats use echolocation like bats to navigate at night." | Not in retrieved context and factually incorrect. |
#| Missing key information | "Cats communicate through meowing but not body language." | Leaves out body language and scent marking, making the answer incomplete. |
