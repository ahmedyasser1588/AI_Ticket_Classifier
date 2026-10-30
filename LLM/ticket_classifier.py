from structured.parser import parse_and_validate_ticket
from structured.schema import TicketSchema
from prompts.classifier_prompt import build_ticket_classifier
from .base import LLM
from .Model import GenerationRequest, Message, MessageRole


class TicketClassifier:
    """Application service for classifying and summarizing support tickets."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def classify(self, ticket: str) -> TicketSchema:
        if not isinstance(ticket, str) or not ticket.strip():
            raise ValueError("ticket must be a non-empty string")

        prompt = build_ticket_classifier(ticket)
        request = GenerationRequest(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You classify support tickets. Return only valid JSON "
                        "matching the requested schema."
                    ),
                ),
                Message(role=MessageRole.USER, content=prompt),
            ],
            prompt=prompt,
            temp=0.0,
            max_new_tokens=200,
        )
        raw_output = self.llm.generate(request)
        return parse_and_validate_ticket(raw_output)
