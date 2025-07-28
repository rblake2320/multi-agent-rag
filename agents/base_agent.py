from abc import ABC, abstractmethod
from typing import Any

from langchain_community.llms import LlamaCpp


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, model_path: str, **llm_kwargs: Any) -> None:
        """Initialize the agent with an LLM.
        
        Args:
            model_path: Path to the LLM model file
            **llm_kwargs: Additional arguments for LLM initialization
        """
        self.llm = LlamaCpp(model_path=model_path, **llm_kwargs)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent workflow.
        
        Args:
            *args: Positional arguments for the agent
            **kwargs: Keyword arguments for the agent
            
        Returns:
            Result of the agent execution
        """

    def _prompt(self, prompt: str) -> str:
        """Send a prompt to the LLM and get response.
        
        Args:
            prompt: Text prompt to send to the LLM
            
        Returns:
            LLM response
        """
        return self.llm(prompt)
