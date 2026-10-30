from dataclasses import dataclass
from enum import Enum

class MessageRole(Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'
    
@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str 

@dataclass 
class GenerationRequest:
    messages: list[Message]
    prompt: str = ""
    top_p: float | None = 1.0
    top_k: int | None = 40
    temp: float | None = 0.7
    temperature: float | None = 0.7
    model: str | None = None
    max_new_tokens: int | None = 200
