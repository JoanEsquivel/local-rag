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
    question = get_question("context_precision", "simple")

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

    # Terminal Log - If you want to get a similar response you need to run the command: pytest tests/test_context_precision.py -s
    # ------------------------------------------------------------------------------------------------
    # Score: 0.7499999999625

    # Question: What physical traits of domestic cats are unique compared to wild felines

    # Answer: Domestic cats exhibit several physical traits that are unique compared to wild felines, including:
    # 1. **Coloration**: Domestic cats have a wider range of colors and patterns due to less need for camouflage in captivity compared to their wild counterparts.
    # 2. **Size**: Domestic cats tend to be smaller in size than many wild felines, which is attributed to changes in diet and habitat.
    # 3. **Brain Size**: Domestic cats have a smaller brain size due to the gradual elimination of unnecessary survival instincts that are more critical in the wild.
    # 4. **Physical Characteristics**: There is a greater variation in body shape and size among domestic cats, influenced by differing gene pools across countries.

    # Retrieved Contexts:
   
    # File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:63:3
    # Content: protection from humans, in the sense that they would be safe from other predators as long 
    # as they remained near human habitats. These two species eventually fused to create a new 
    # breed of cat, related to the modern-day Egyptian Mau. 
    # The change in temperament is attributed to two principal factors: heredity and learned 
    # tolerance of humans. The changes due to domestication follow a pattern similar to other 
    # domesticated animals including wolves (dogs), and cattle. These changes include coloration 
    # as there is less need for camouflage in captivity than in the wild, smaller brain size due to the 
    # gradual elimination of unnecessary survival instincts, and an overall decrease in size due to 
    # the change in diet and habitat.

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

    # File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:34:2
    # Content: vinyl nail caps that are affixed to the claws with nontoxic glue, requiring periodic 
    # replacement when the cat sheds its claw sheaths (usually every four to six weeks). 
    # Environment 
    # The wild cat, ancestor of the domestic cat, is believed to have evolved in a desert climate, 
    # as evident in the behavior common to both the domestic and wild forms. Wild cats are native 
    # to all continents other than Australasia and Antarctica. Their feces are usually dry, and cats 
    # prefer to bury them in sandy places. They are able to remain motionless for long periods, 
    # especially when observing prey and preparing to pounce. In North Africa there are still small 
    # wildcats that are probably related closely to the ancestors of today's domesticated breeds.

    # File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:246:1
    # Content: can be any colour or combination of colours. They also exhibit a wide range of physical 
    # characteristics, and as a result, domestic shorthaired cats in different countries tend to look 
    # different in body shape and size, as they are working from differing gene pools. However, 
    # they are all recognizable as cats, and any male (tom) cat could successfully breed with any 
    # other female (queen), meaning they are the same species. 
    # See also 
    # • Cat coat genetics 
    # • Domestic longhaired cat 
    # Home | Up 
    
    # Farm Cat 
    # Farm cats are cats used for catching pests on farms. They are feral cats, meaning that 
    # they are wild and you should have caution around them. It depends on the farmer if they will 
    # be treated well, like with food and water, or just being there to do what they are supposed to

    # File: data/About_Cats-Nicolae_Sfetcu-CCNS.pdf:111:1
    # Content: crossbreeding, hybridizing domestic cats with desired coat and temperament features with 
    # Asian Leopard Cats (ALC) and ALC hybrids. The principle of hybrid vigor dictates that hybrid 
    # cats are often larger than either parent, but are typically infertile. F1 and F2 males are nearly 
    # always infertile, F3 males are normally infertile, but females are often fertile even in early 
    # hybrids. 
    # A cat with one wild ancestor is called an F1, short for first filial. An F1 bred with a 
    # domestic cat or other bengal filial cat yields an F2, or second filial. Any kittens from an F2 
    # female are termed F3. Any kittens from an F3 female are termed F4. F4 and higher 
    # generations are officially known as Stud Book Tradition (SBT) bengals and can be shown and

    # ------------------------------------------------------------------------------------------------
    # Analisis
    # These traits reflect the adaptations that have occurred due to domestication and living in human environments.
    # ------------------------------------------------------------------------------------------------
    # What Went Well?
    # ✔️ The system pulled useful details about why domestic cats are different from wild felines, like:
    # Smaller brain size (due to domestication).
    # Less need for camouflage, which explains the wide variety of coat colors.
    # Smaller overall size compared to wild felines.
    # Greater variation in body shape and size due to selective breeding.
    # The final answer was much more focused on the differences between domestic and wild cats, rather than just listing general cat traits.

    # What Could Be Better?
    # ⚠️ Some of the retrieved passages were not that relevant:
    # One passage talked about coat color diversity in domestic cats, which isn’t really about what makes them unique compared to wild cats.
    # Another passage focused on hybrid breeding with wild cats, which is interesting but not answering the question directly.
    # ⚠️ The response missed some key physical differences like:
    # Domestic cats have shorter snouts and smaller teeth compared to wild felines.
    # Their tails are more upright in communication, unlike wild cats.
    # Their fur is softer, since they don’t have to survive in harsh outdoor conditions.

    # How Can We Improve It?
    # If we ask a more specific question, the system will likely retrieve even better content.
    # 🔹 Better Question:
    # 👉 What skeletal and physiological changes distinguish domestic cats from wild felines?
  

