from pydantic import BaseModel
from structured.schema import TicketSchema
from typing import Optional
class EvaluationModel(BaseModel):
    """
    Stores the result of evaluting one support ticket 
    this model contains:
    - the ticket id
    - the ground truth labels
    - the model prediction
    - correctness information
    - generation status
    - error information
    """

    ticket_id:int
    # Did the model successfully produce a structured output
    success: bool
    # Graund-truth values from the evaluation dataset  
    expected_category:str
    expected_sentiment:str
    expected_urgency:str

    # Model Prediction 
    # None When generation fails
    result: Optional[TicketSchema]=None

    # Field-level correctness 
    category_correct:bool=False
    sentiment_correct:bool=False
    urgency_correct:bool=False

    # Error information when evaluation fails
    error: Optional[str]=None 
