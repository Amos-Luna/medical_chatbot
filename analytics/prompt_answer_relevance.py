from typing import ClassVar
from langchain.prompts import PromptTemplate


class AnswerRelevance:
    """Metric related to: How much relevance does the Answer hold for the Question?"""

    system_prompt: ClassVar[PromptTemplate] = PromptTemplate.from_template(
        """You are a RELEVANCE grader; providing the relevance of the given RESPONSE to the given QUESTION.
        Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant.

        A few additional scoring guidelines:

        - Long RESPONSES should score equally well as short RESPONSES.

        - If both occur together RESPONSES and QUESTION tends to express greetings or farewells or sinonyms), should also be counted as the most RELEVANT. 
        
        - RESPONSES that intentionally do not RESPONSE the QUESTION, such as 'I don't know/understand' and model refusals, should also be counted as the most RELEVANT.
        
        - RESPONSES that tends to Apologizing, Frustrations, Madness, etc or other similars cases because of the QUESTION, should also be counted as the most RELEVANT.

        - RESPONSE must be relevant to the entire QUESTION to get a score of 10.

        - RELEVANCE score should increase as the RESPONSE provides RELEVANT context to more parts of the QUESTION.

        - RESPONSE that is RELEVANT to none of the QUESTION should get a score of 0.

        - RESPONSE that is RELEVANT to some of the QUESTION should get as score of 2, 3, or 4. Higher score indicates more RELEVANCE.

        - RESPONSE that is RELEVANT to most of the QUESTION should get a score between a 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

        - RESPONSE that is RELEVANT to the entire QUESTION should get a score of 9 or 10.

        - RESPONSE that is RELEVANT and answers the entire QUESTION completely should get a score of 10.

        - RESPONSE that confidently FALSE should get a score of 0.

        - RESPONSE that is only seemingly RELEVANT should get a score of 0.

        - Never elaborate.
        """
    )

    user_prompt: ClassVar[PromptTemplate] = PromptTemplate.from_template(
        """QUESTION: {question}

        RESPONSE: {response}

        RELEVANCE: """
    )