from langchain.prompts import ChatPromptTemplate
from analytics.prompt_answer_relevance import AnswerRelevance
from analytics.prompt_context_relevance import ContextRelevance
from analytics.prompt_groundedness import Groundedness
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def qualify_answer_relevance(
    question: str, 
    response: str
) -> int:
    """How much is question related to the ai answer?"""

    prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", AnswerRelevance.system_prompt.template),
                ("user", AnswerRelevance.user_prompt.template),
            ]
        )
    
    chain = prompt_template | llm
    result = chain.invoke({
            "question": question,
            "response": response
        }
    )
    
    try:
        return int(result.content)
    except ValueError as ve:
        print(f"Error - Answer Relevance Score: {ve}")


def qualify_context_relevance(
    question: str, 
    context: str
)-> int:
    """How much context is related to the user message?"""

    prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", ContextRelevance.system_prompt.template),
                ("user", ContextRelevance.user_prompt.template),
            ]
        )
    
    chain = prompt_template | llm
    result = chain.invoke({
            "question": question,
            "context": context
        }
    )
    
    try:
        return int(result.content)
    except ValueError as ve:
        print(f"Error - Answer Relevance Score: {ve}")
        

def qualify_groundedness(
    context: str, 
    response: str
) -> int:
    """How much context is related to ai response?"""

    prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", Groundedness.system_prompt.template),
                ("user", Groundedness.user_prompt.template),
            ]
        )
    
    chain = prompt_template | llm
    result = chain.invoke({
            "context": context,
            "response": response
        }
    )
    
    try:
        return int(result.content)
    except ValueError as ve:
        print(f"Error - Answer Relevance Score: {ve}")