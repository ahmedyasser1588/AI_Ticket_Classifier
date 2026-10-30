import json
import re
from pydantic import ValidationError
from .schema import TicketSchema


def parse_and_validate_ticket(raw_output: str) -> TicketSchema:
    cleaned = raw_output.strip()
    # Strip markdown code fences if present (e.g. ```json ... ```)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        return TicketSchema.model_validate(data)
    except json.JSONDecodeError:
        # Try extracting JSON object via regex if there's surrounding text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return TicketSchema.model_validate(data)
            except (json.JSONDecodeError, ValidationError):
                pass
        raise ValueError(f"Invalid JSON format: {raw_output}")
    except ValidationError as e:
        raise ValueError(
            f"Validation error: The output does not conform to the expected schema. Details: {e}"
        )


# Alias for backwards compatibility
parser_and_validate_ticket = parse_and_validate_ticket