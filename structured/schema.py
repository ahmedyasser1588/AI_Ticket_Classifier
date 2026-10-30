from pydantic import BaseModel
from typing import Optional,Literal 
class TicketSchema(BaseModel):
    category:Literal["technical", "account", "delivery", "billing", "subscription"]
    sentiment:Literal["positive", "negative", "neutral"]
    urgency:Literal["low", "medium", "high"]
    summary:str
    