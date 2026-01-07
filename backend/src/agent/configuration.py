import os
from typing import Any, Optional

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """Configuration for the Groq-based research agent."""


    query_generator_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model used to generate research queries",
    )

    reflection_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model used to reflect on gathered information",
    )

    answer_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model used to generate the final answer",
    )

    number_of_initial_queries: int = Field(
        default=3,
        description="Number of initial reasoning steps or sub-questions",
    )

    max_research_loops: int = Field(
        default=1,
        description="Maximum number of reasoning loops (kept for compatibility)",
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """
        Create Configuration from LangGraph RunnableConfig.

        Priority:
        1. RunnableConfig.configurable
        2. Environment variables
        3. Defaults
        """
        configurable = {}
        if config and isinstance(config, dict):
            configurable = config.get("configurable", {})

        values: dict[str, Any] = {}

        for field_name in cls.model_fields:
            env_key = field_name.upper()
            if field_name in configurable:
                values[field_name] = configurable[field_name]
            elif env_key in os.environ:
                values[field_name] = os.environ[env_key]

        return cls(**values)

