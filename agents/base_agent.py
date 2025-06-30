from abc import ABC, abstractmethod
from typing import Any

from langchain.llms import LlamaCpp


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, model_path: str, **llm_kwargs: Any):
        self.llm = LlamaCpp(model_path=model_path, **llm_kwargs)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any):
        """Execute the agent workflow."""

    def _prompt(self, prompt: str) -> str:
        return self.llm(prompt)
