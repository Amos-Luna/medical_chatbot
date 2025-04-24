from pydantic import BaseModel, Field


class AgentResponseSchema(BaseModel):
    """Always use this schema to structure the response and decide whether the next step should be handled by an agent"""

    next_step: str = Field(
        description="Specifies the next step: 'allergy_agent' or 'digestive_agent' or 'vision_loss_agent'. Should be one of them."
    )
