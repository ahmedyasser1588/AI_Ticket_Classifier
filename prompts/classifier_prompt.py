def build_ticket_classifier(user_prompt: str, model=None, tokenizer=None) -> str:
    return f"""You are an expert customer support ticket classification system.
Your task is to analyze the customer ticket and classify it accurately into the required fields.

### Guidelines:
1. Category Definitions:
   - "technical": Hardware failures, app crashes, screen broken, bugs, system errors.
   - "account": Passwords, login issues, account access, credentials reset.
   - "delivery": Shipping status, package delays, tracking, physical shipment issues.
   - "billing": Double charges, payment failed, unauthorized deductions, refunds.
   - "subscription": Upgrading plan, cancelling subscription, membership changes.

2. Sentiment Definitions:
   - "positive": Customer expressing gratitude, praise, happiness.
   - "negative": Customer expressing frustration, problems, anger, payment issues.
   - "neutral": Informational questions, standard requests without strong emotion.

3. Urgency Definitions:
   - "high": Money deducted, broken screens, blocking bugs, urgent payment failures.
   - "medium": Delayed packages, password resets, standard cancellations.
   - "low": Thank you messages, simple questions about upgrades, early deliveries.

### Examples:
Ticket: "The payment failed but money was deducted from my bank."
Result: {{"category": "billing", "sentiment": "negative", "urgency": "high", "summary": "Payment failed but money was deducted."}}

Ticket: "Your support team was very helpful, thank you so much!"
Result: {{"category": "account", "sentiment": "positive", "urgency": "low", "summary": "Customer thanking support team."}}

### Now Classify This Ticket:
Ticket:
"{user_prompt}"

Provide valid JSON matching the schema:
"""
