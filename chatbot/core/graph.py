import logging
from langgraph.graph.graph import START
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from chatbot.core.states import CustomState
from typing import Dict, Any
from dotenv import load_dotenv
from chatbot.core.nodes import (
    supervisor_agent,
    allergy_agent,
    digestive_agent,
    vision_loss_agent,
    analyst_agent_allergy,
    analyst_agent_digestive,
    analyst_agent_vision_loss,
)
load_dotenv()

class Graph:
    workflow: StateGraph

    def __init__(self) -> None:
        self.workflow = StateGraph(CustomState)
        self.__set_nodes()
        self.__set_edges()
        self.graph = self.workflow.compile()


    def __set_nodes(self) -> None:
        """
        Defines the nodes for the workflow, including the AI agent and tool nodes.
        """
        workflow = self.workflow
        workflow.add_node("supervisor_agent", supervisor_agent)
        # agregar un agente para preguntas no relacionadas a medicamentos...
        # agregar metricas de trulens
        workflow.add_node("allergy_agent", allergy_agent)
        workflow.add_node("digestive_agent", digestive_agent)
        workflow.add_node("vision_loss_agent", vision_loss_agent)
        workflow.add_node("analyst_agent_allergy", analyst_agent_allergy)
        workflow.add_node("analyst_agent_digestive", analyst_agent_digestive)
        workflow.add_node("analyst_agent_vision_loss", analyst_agent_vision_loss)


    def __set_edges(self) -> None:
        """
        Defines the edges and conditional transitions between nodes in the workflow.
        """
        workflow = self.workflow
        workflow.add_edge(START, "supervisor_agent")
        workflow.add_edge("analyst_agent_allergy", "allergy_agent")
        workflow.add_edge("analyst_agent_digestive", "digestive_agent")
        workflow.add_edge("analyst_agent_vision_loss", "vision_loss_agent")


    def execute_agent(
        self, 
        message: str
    ) -> str:
        """
        Executes the AI agent workflow with the given user message and memory context.

        Args:
            message (str): The user's input message.
            memory (str): The stored conversation history.

        Returns:
            str: The AI-generated response or an error message.
        """
        try:
            print(f"User message: {message}")

            final_state: Dict[str, Any] = self.graph.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "chunks_retrieved": "",
                    "allergy_result": "",
                    "digestive_result": "",
                    "vision_loss_result": "",
                },
            )
            
            if final_state["allergy_result"]:
                result = final_state["allergy_result"]
                chunks_retrieved = final_state["chunks_retrieved"]
            elif final_state["digestive_result"]:
                result = final_state["digestive_result"]
                chunks_retrieved = final_state["chunks_retrieved"]
            elif final_state["vision_loss_result"]:
                result = final_state["vision_loss_result"]
                chunks_retrieved = final_state["chunks_retrieved"]
            else:
                result = final_state["messages"][-1].content
                chunks_retrieved = ""
            
            print()
            print(f"Final result: {result}")
            
            return result, chunks_retrieved

        except Exception as e:
            print(f"Error executing agent: {e}")
            return f"Error Agent Graph: {str(e)}"
