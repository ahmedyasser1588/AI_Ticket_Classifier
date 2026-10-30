from abc import ABC,abstractmethod
from .Model import GenerationRequest
class LLM(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def generate(self,request:GenerationRequest):
        return self.llm.generate(request)

