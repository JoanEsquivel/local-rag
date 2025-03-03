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
    question = get_question("context_recall", "simple")

    # Get Reference
    reference = get_reference("context_recall", "simple_reference")

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
  

# Terminal Log - If you want to get a similar response you need to run the command: pytest tests/test_context_recall.py -s
# ------------------------------------------------------------------------------------------------
# Score: 0.4

# Question: What are the different types of cat coats, fur textures, and patterns, and how do they affect a cat's appearance?

# Answer: The context describes several types of cat coats, fur textures, and patterns, which significantly affect a cat's appearance:
# 1. **Tabby Cat Patterns**:
#    - **Blotched Tabby**: Features a classic pattern with butterflies and bull's-eyes, common in NW Europe.
#    - **Mackerel Tabby**: Characterized by vertical stripes down the sides, resembling a fish.
#    - **Spotted Tabby**: A variation where the stripes are broken into spots.
# 2. **Tortoiseshell and Calico**:
#    - These coats feature three colors mottled throughout, also known as "tortie." A true tortoiseshell has specific color combinations that create a unique appearance.
# 3. **Fur Length and Texture**:
#    - Governed by the Long hair gene, where the dominant form (L) results in short hair and the recessive form (l) leads to long hair.
#    - There are also genes that produce curly-coated or "rex" cats, which have a unique texture that is not silky but has a drag like velvet. The coat is described as loose, bouncy, and airy, often standing away from the body.
# 4. **Coat Characteristics**:
#    - The coat can vary by season and maturity, typically being wavy or curly, with the longest curls found in the ruff and neck area. 
#    - Longhairs have a curly plumed tail, while shorthairs have tails resembling bottle brushes. Both types have long curled whiskers.
# These variations in coat types, patterns, and textures contribute to the overall aesthetic appeal of cats, influencing how they are perceived and appreciated by cat lovers.
# These traits reflect the adaptations that have occurred due to domestication and living in human environments.

# Reference(truth): Cats have various coat types, fur textures, and patterns that influence their appearance. Coat types include short-haired, long-haired, semi-longhaired, curly (like Rex breeds), and hairless (like the Sphynx). Fur textures vary from soft and silky (Persian) to coarse (Abyssinian) or wavy (Devon Rex). Common fur patterns include solid (one color), tabby (striped or swirled), bicolor (two colors), tortoiseshell (blended black and orange), calico (tortoiseshell with white), pointed (darker face, ears, paws, and tail, like Siamese), and spotted (like Bengals). These traits impact a cat’s appearance by influencing color distribution, texture feel, and overall distinctiveness, with genetics playing a key role in determining variations.



# Retrieved Contexts:

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:39:1
# Content: large ears and very short sleek fur). 
# Tabby cat  
# Striped, with a variety of patterns. The classic "blotched tabby" pattern is the most 
# common and consists of butterflies and bull's -eyes. The mackerel tabby is a series of 
# vertical stripes down the cat's side (resembling the fish). This pattern broken into spots 
# is referred to as spotted tabby. The worldwide evolution of the cat means that certain 
# types of tabby are associated with certain countries; for instance, blotched tabbies are 
# quite rare outside NW Europe, where they are the most common type. 
# Tortoiseshell and Calico  
# Featuring three colors mottled throughout the coat, this cat is also known as a Calimanco 
# cat or Clouded Tiger cat, and by the nickname "tortie". A true tortoiseshell must consist

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:55:1
# Content: as yet unidentified, believed to result in different degrees of shading, some more 
# desirable than others. 
# Genes involved in fur length and texture 
# Cat fur length is governed by the Long hair gene in which the dominant form, L codes for 
# short hair, and the recessive l codes for long hair. 
# There are many genes resulting in unusual fur. These genes were discovered in random-
# bred cats and selected for. Some of the genes are in danger of going extinct because the 
# breeders have not marketed their cats effectively, the cats are not sold beyond the region 
# where the mutation originated, or there is simply not enough demand for the mutation. 
# There are various genes producing curly coated or "rex" cats. New types of rex pop up

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:51:2
# Content: the Wikipedia. 
#  
# Cat Coat Genetics 
#  
#  Back | Home | Up | Next 
# Home | Up 
#  
# The genetics of cat coat coloration, pattern, length, and texture is a complex subject, and 
# many different genes are involved. 
# Genes involved in albinism, dominant white, and 
# white spotting 
# • The dominant C gene and its recessive alleles determine whether a cat is a 
# complete albino (either pink-eyed or blue-eyed), a temperature sensitive albino 
# (Burmese, Siamese, or a blend known as Tonkinese), or a non-albino. If a cat has

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:133:4
# Content: Tabby points are especially attractive. Newer varieties such as ticked tabbies, shadeds and 
# darker points are also being bred. The curl tends to open up the coat showing off shading, 
# ticking or silver undercoats. 
# The coat itself is described as having a unique textured feel. It is not silky, having a certain 
# drag on the hand like velvet and the texture comes as much from the shape of the curls as 
# from the mixture of different hair types. It should be soft and inviting, although the shorthairs

# File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:134:0
# Content: NICOLAE SFETCU: ABOUT CATS 
# 133 
# will have more texture to their coats. The coat is rather loose and bouncy often feeling 
# springy when patted, and stands away from the body with no thick undercoat. It is light and 
# airy and judges sometimes blow on the coat to see if it will part. The coat varies according to 
# the season and the maturity of the cat but is essentially wavy or curly all over with the longest 
# and most defined curls in the ruff and on the neck often falling in ringlets. There are also 
# curly ear furnishings including tufts at the ear tips and ear muffs. The longhairs have a curly 
# plumed tail while the shorthairs have tails rather like bottle brushes, and both have long 
# curled whiskers. Sometimes the coat falls into a natural parting along the bac k, jokingly


# ------------------------------------------------------------------------------------------------
# Analisis
# ------------------------------------------------------------------------------------------------
# It means the system didn’t pull in enough information to fully answer the question. It got some things right, but it missed a lot. Let’s go step by step.
# ------------------------------------------------------------------------------------------------
# What the System Did Well
#✔️ It found some good details about coat patterns, like:
# - Tabby patterns (striped, blotched, spotted).
# - Tortoiseshell and calico coats (the mix of black, orange, and white).
# - Some genetics behind fur length and texture (short hair vs. long hair, curly Rex coats).
#✔️ It mentioned how fur texture affects appearance, like how curly coats feel different.

# What Went Wrong? Why the Score is Low?
# 🚨 It left out a lot of important details.
# ❌ Didn’t mention all coat types – The system forgot about solid-colored cats, bicolor, pointed (like Siamese), and spotted coats (like Bengals).
# ❌ Missed some fur textures – It only talked about curly coats (Rex breeds), but what about silky Persians or coarse Abyssinians?
# ❌ Barely mentioned hairless cats – The Sphynx was completely left out.
# ❌ No real talk about genetics – Coat colors and patterns are largely genetic, but the system didn’t pull enough info on this.

# Basically, it got part of the answer but not the full picture—that’s why the recall score is low.


# ------------------------------------------------------------------------------------------------
