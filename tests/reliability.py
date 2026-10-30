import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.inference.config import DataSet
except ImportError:
    from Src.inference.config import DataSet

import pandas as pd

try:
    from structured.generator import StructuredGeneration
except ImportError:
    from Structured.generator import StructuredGeneration

try:
    from src.inference.model import load_model
except ImportError:
    from Src.inference.model import load_model

try:
    from structured.schema import TicketSchema
except ImportError:
    from Structured.Schema import TicketSchema

def load_dataset(path=None):
    if path is None:
        return pd.read_csv(DataSet)
    else:
        return pd.read_csv(path)

def get_row(DataSet,index=0):
    row=DataSet.iloc[index]
    return{
        'ticket_id': row['ticket_id'],
        'message':row['message'],
        'expected':{
            'category':row["category"],
            'sentiment':row["sentiment"],
            'urgency':row["urgency"]
        }
    }
def build_baseline_prompt(message):
    return f"""
    You are a customer support ticket classifier.
    Classify the following customer ticket.
    return only the structured result.
    Ticket:
    <ticket>
    {message}
    </ticket>
    """.strip()


if __name__ =="__main__":
    tokenizer,model=load_model()
    generator=StructuredGeneration(model,tokenizer)

    dataset=load_dataset()
    test_case=get_row(dataset,1)
    prompt=build_baseline_prompt(test_case['message'])
    print(prompt)
    result=generator.generate(prompt,TicketSchema)
    print(result)
    print(test_case['expected'])
    

