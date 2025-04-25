from typing import ClassVar
from langchain.prompts import PromptTemplate


class ContextRelevance:
    """Metric related to: How much relevance does the retrieved Context hold for the Question?"""

    system_prompt: ClassVar[PromptTemplate] = PromptTemplate.from_template(
        """You are a RELEVANCE grader; providing the relevance of the given CONTEXT to the given QUESTION.
        Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

        A few additional scoring guidelines:

        - If both cases occur together (the CONTEXT is empty or contains whatever text) AND (the QUESTION tends to express greetings or farewells or sinonyms), assign a score of 10.  
        
        - If both cases occur together (the CONTEXT contains totally uncomprenhensible text) AND (the QUESTION contains totally uncomprenhensible text), assign a score of 0.
                
        - Long CONTEXTS should score equally well as short CONTEXTS.

        - RELEVANCE score should increase as the CONTEXTS provides more RELEVANT context to the QUESTION.

        - RELEVANCE score should increase as the CONTEXTS provides RELEVANT context to more parts of the QUESTION.

        - CONTEXT that is RELEVANT to some of the QUESTION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

        - CONTEXT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

        - CONTEXT that is RELEVANT to the entire QUESTION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

        - CONTEXT must be relevant and helpful for answering the entire QUESTION to get a score of 10.

        - Never elaborate."""
    )
    user_prompt: ClassVar[PromptTemplate] = PromptTemplate.from_template(
        """QUESTION: {question}

        CONTEXT: {context}
        
        RELEVANCE: """
    )
