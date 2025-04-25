import streamlit as st
import logging
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from typing_extensions import Literal
from resources.prompt_template import (
    SupervisorAgentPromptTemplate,
    AllergyAgentPromptTemplate,
    DigestiveAgentPromptTemplate,
    VisionLossAgentPromptTemplate,
)
from chatbot.core.states import CustomState
from resources.schemas import AgentResponseSchema
from chatbot.tools.vector_store_tool import (
    allergy_tools, 
    digestive_tools, 
    vision_loss_tools
)
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def supervisor_agent(
    state: CustomState,
) -> Command[Literal[
    "allergy_agent", 
    "digestive_agent", 
    "vision_loss_agent", 
    "__end__"]
]:
    """Supervisor agent that determine the next step to continue"""
    
    if (state["allergy_result"] or state["digestive_result"] or state["vision_loss_result"]):
        print(f"Going to __end__ the process....")
        return Command(
            update={
                "messages": state["messages"],
                "chunks_retrieved": state["chunks_retrieved"],
                "allergy_result": state["allergy_result"],
                "digestive_result": state["digestive_result"],
                "vision_loss_result": state["vision_loss_result"],
            },
            goto="__end__",
        )
        
    llm_with_structure_output = llm.with_structured_output(AgentResponseSchema)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", SupervisorAgentPromptTemplate.system_prompt.template),
            ("user",SupervisorAgentPromptTemplate.user_prompt.template)
        ]
    )
    chain = prompt_template | llm_with_structure_output
    result = chain.invoke({"user_message": state["messages"]})
    print(f"supervisor_agente go to.... {result.next_step.strip()}")

    if result.next_step.strip() not in ["allergy_agent", "digestive_agent", "vision_loss_agent"]:
        print(f"---> [Error]--> Agent decision incorrectly: {result.next_step.strip()}. This should be 'allergy_agent' or 'digestive_agent' or 'vision_loss_agent'")
        raise ValueError(f"---> [Error]--> Agent decision incorrectly: {result.next_step.strip()}. This should be 'allergy_agent' or 'digestive_agent' or 'vision_loss_agent'")

    if result.next_step.strip() == "allergy_agent":
        return Command(
            update={
                "messages": state["messages"],
                "chunks_retrieved": state["chunks_retrieved"],
                "allergy_result": state["allergy_result"],
                "digestive_result": state["digestive_result"],
                "vision_loss_result": state["vision_loss_result"],
            },
            goto=result.next_step,
        )

    if result.next_step.strip() == "digestive_agent":
        return Command(
            update={
                "messages": state["messages"],
                "chunks_retrieved": state["chunks_retrieved"],
                "allergy_result": state["allergy_result"],
                "digestive_result": state["digestive_result"],
                "vision_loss_result": state["vision_loss_result"],
            },
            goto=result.next_step,
        )
    
    if result.next_step.strip() == "vision_loss_agent":
        return Command(
            update={
                "messages": state["messages"],
                "chunks_retrieved": state["chunks_retrieved"],
                "allergy_result": state["allergy_result"],
                "digestive_result": state["digestive_result"],
                "vision_loss_result": state["vision_loss_result"],
            },
            goto=result.next_step,
        )


def allergy_agent(
    state: CustomState
) -> Command[Literal["analyst_agent_allergy","supervisor_agent"]]:
    """Agent to answer questions related to allergies"""
    print("Inside allergy_agent....")
    print()
    print(f"---> State --->: {state}")
    
    if isinstance(state["messages"][-1], ToolMessage):
        print(f"Going to supervisor agent because the last message is a ToolMessage")
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", AllergyAgentPromptTemplate.formulate_answer.template),
            ]
        )
        chain = prompt_template | llm
        result = chain.invoke({
                "user_message": state["messages"][0].content,
                "chunks": state["messages"][-1].content
            }
        )
        print("Result of allergy to supervisor agent: ", result)
        return Command(
            update={
                "messages": state["messages"],
                "chunks_retrieved": state["messages"][-1].content,
                "allergy_result": result.content,
                "digestive_result": state["digestive_result"],
                "vision_loss_result": state["vision_loss_result"],
            },
            goto="supervisor_agent",
        )
    
    model_with_tools = llm.bind_tools(allergy_tools)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", AllergyAgentPromptTemplate.system_prompt.template),
            ("user", AllergyAgentPromptTemplate.user_prompt.template),
        ]
    )

    chain = prompt_template | model_with_tools
    result = chain.invoke({
            "user_message": state["messages"][-1].content
        }
    )
    print(f"result: {result}")
    
    return Command(
        update={    
            "messages": [result],
            "chunks_retrieved": state["chunks_retrieved"],
            "allergy_result": state["allergy_result"],
            "digestive_result": state["digestive_result"],
            "vision_loss_result": state["vision_loss_result"],
        },
        goto="analyst_agent_allergy",
    )


def digestive_agent(
    state: CustomState
) -> Command[Literal["analyst_agent_digestive", "supervisor_agent"]]:
    """Agent to answer questions related to digestive issues"""
    print("Inside digestive_agent....")
    print(f"state: {state}")
    
    if isinstance(state["messages"][-1], ToolMessage):
        print(f"Going to supervisor agent because the last message is a ToolMessage")
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", DigestiveAgentPromptTemplate.formulate_answer.template),
            ]
        )
        chain = prompt_template | llm
        result = chain.invoke({
                "user_message": state["messages"][0].content,
                "chunks": state["messages"][-1].content
            }
        )
        print("Result of digestive to supervisor agent: ", result)        
        return Command(
            update={
                "messages": state["messages"],
                "chunks_retrieved": state["messages"][-1].content,
                "allergy_result": result.content,
                "digestive_result": state["digestive_result"],
                "vision_loss_result": state["vision_loss_result"],
            },
            goto="supervisor_agent",
        )
        
    model_with_tools = llm.bind_tools(digestive_tools)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", DigestiveAgentPromptTemplate.system_prompt.template),
            ("user", DigestiveAgentPromptTemplate.user_prompt.template),
        ]
    )

    chain = prompt_template | model_with_tools
    result = chain.invoke({
            "user_message": state["messages"][-1].content
        }
    )

    return Command(
        update={    
            "messages": [result],
            "chunks_retrieved": state["chunks_retrieved"],
            "allergy_result": state["allergy_result"],
            "digestive_result": state["digestive_result"],
            "vision_loss_result": state["vision_loss_result"],
        },
        goto="analyst_agent_digestive",
    )


def vision_loss_agent(
    state: CustomState
) -> Command[Literal["analyst_agent_vision_loss", "supervisor_agent"]]:
    """Agent to answer questions related to vision loss"""
    print("Inside vision_loss_agent....")
    print(f"state: {state}")
    
    if isinstance(state["messages"][-1], ToolMessage):
        print(f"Going to supervisor agent because the last message is a ToolMessage")
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", VisionLossAgentPromptTemplate.formulate_answer.template),
            ]
        )
        chain = prompt_template | llm
        result = chain.invoke({
                "user_message": state["messages"][0].content,
                "chunks": state["messages"][-1].content
            }
        )
        print("Result of vision loss to supervisor agent: ", result)
        return Command(
            update={
                "messages": state["messages"],
                "chunks_retrieved": state["messages"][-1].content,
                "allergy_result": result.content,
                "digestive_result": state["digestive_result"],
                "vision_loss_result": state["vision_loss_result"],
            },
            goto="supervisor_agent",
        )
        
    model_with_tools = llm.bind_tools(vision_loss_tools)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", VisionLossAgentPromptTemplate.system_prompt.template),
            ("user", VisionLossAgentPromptTemplate.user_prompt.template),
        ]
    )

    chain = prompt_template | model_with_tools
    result = chain.invoke({
            "user_message": state["messages"][-1].content
        }
    )

    return Command(
        update={    
            "messages": [result],
            "chunks_retrieved": state["chunks_retrieved"],
            "allergy_result": state["allergy_result"],
            "digestive_result": state["digestive_result"],
            "vision_loss_result": state["vision_loss_result"],
        },
        goto="analyst_agent_vision_loss",
    )

    
analyst_agent_allergy = ToolNode(allergy_tools)
analyst_agent_digestive = ToolNode(digestive_tools)
analyst_agent_vision_loss = ToolNode(vision_loss_tools)