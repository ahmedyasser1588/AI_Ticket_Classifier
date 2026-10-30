import os 
import sys
sys.path.append(os.path.abspath('..'))

try:
    from src.inference.model import load_model
except ImportError:
    from Src.inference.model import load_model

try:
    from structured.generator import StructuredGeneration
except ImportError:
    from Structured.generator import StructuredGeneration

try:
    from structured.schema import TicketSchema
except ImportError:
    from Structured.Schema import TicketSchema

try:
    from prompts.classifier_prompt import build_ticket_classifier
except ImportError:
    from Prompts.Classifier_Prompt import build_ticket_classifier



def main() : 
    tokenizer, model = load_model()

    generator = StructuredGeneration(model, tokenizer)

    user_query = """Your support team was very helpful, thank you!"""

    prompt = build_ticket_classifier(user_query,model,tokenizer)

    result = generator.generate(prompt, TicketSchema)

    print(result)


if __name__ == '__main__' :
    main()