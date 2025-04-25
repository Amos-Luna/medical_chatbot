from typing import ClassVar
from langchain.prompts import PromptTemplate


class Groundedness:
    """Metric related to: How much does the Context support the Answer?"""

    system_prompt: ClassVar[PromptTemplate] = PromptTemplate.from_template(
        """You are a GROUNDEDNESS grader; providing the degree to which the RESPONSE is supported by the given CONTEXT.
        Respond only as a number from 0 to 10 where 0 is no grounding and 10 is fully grounded.

        A few additional scoring guidelines:

        - If both cases occur together (the CONTEXT is empty or contains comprehensible text) AND (the RESPONSE tends to express greetings or farewells or synonyms), assign a score of 10.
        
        - RESPONSE that is not supported by the CONTEXT, should get a score of 0.
    
        - Long CONTEXTS should score equally well as short CONTEXTS.

        - Groundedness score should increase as the RESPONSE is more supported by the CONTEXT.

        - Groundedness score should increase as the RESPONSE references more relevant parts of the CONTEXT.
        
        - RESPONSE that confidently FALSE based on the CONTEXT should get a score of 0.
        
        - RESPONSE that only seems grounded but isn't should get a score of 0.

        - RESPONSE that is partially supported by the CONTEXT should get a score of 2, 3, or 4. Higher score indicates more support.

        - RESPONSE that is mostly supported by the CONTEXT should get a score between 5, 6, 7 or 8. Higher score indicates more support.

        - RESPONSE that is fully supported by the CONTEXT should get a score of 9 or 10. Higher score indicates more support.

        - Never elaborate."""
    )

    user_prompt: ClassVar[PromptTemplate] = PromptTemplate.from_template(
        """CONTEXT: {context}

        RESPONSE: {response}

        GROUNDEDNESS: """
    )
