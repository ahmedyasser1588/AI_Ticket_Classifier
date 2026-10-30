import outlines

try:
    from .schema import TicketSchema
except ImportError:
    try:
        from structured.schema import TicketSchema
    except ImportError:
        pass

def create_structured_model(model, tokenizer):
    if hasattr(outlines, "from_transformers"):
        return outlines.from_transformers(model, tokenizer)
    elif hasattr(outlines, "models") and hasattr(outlines.models, "transformers"):
        return outlines.models.transformers(model, tokenizer)
    else:
        raise AttributeError("Could not find transformer adapter in outlines package.")