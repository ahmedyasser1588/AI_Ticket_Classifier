try:
    from .outline_adapter import create_structured_model
except ImportError:
    try:
        from structured.outline_adapter import create_structured_model
    except ImportError:
        from Structured.outline_adapter import create_structured_model

class StructuredGeneration:
    def __init__(self, model, tokenizer):
        self.model = create_structured_model(model, tokenizer)

    def generate(self, prompt, schema):
        try:
            return self.model(prompt, schema, max_new_tokens=512)
        except TypeError:
            return self.model(prompt, schema)
     