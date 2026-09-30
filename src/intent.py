import sys
import os
import json
import re
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient

class IntentCategory(str, Enum):
    ORDER_TRACKING = "Order/Tracking Status"
    CANCELLATION_REFUND = "Cancellation/Refund Request"
    ACCOUNT_AUTH = "Account Access/Authentication"
    BILLING_PAYMENT = "Billing/Payment Issue"
    SERVICE_OUTAGE = "Service Outage/Technical Bug"
    GENERAL_INQUIRY = "General Inquiry/Feedback"

class IntentClassificationResult(BaseModel):
    predicted_intent: IntentCategory = Field(description="Primary intent category for customer query.")
    secondary_intent: Optional[IntentCategory] = Field(default=None, description="Optional secondary intent category if query contains compound intents.")
    confidence_score: float = Field(description="Confidence level between 0.0 and 1.0.")
    reasoning: str = Field(description="Brief justification for chosen intent(s).")

class IntentClassifier:
    def __init__(self, llm_client: LLMClient = None, generator_pipeline=None):
        if llm_client is not None:
            self.llm = llm_client
        else:
            self.llm = LLMClient.get_shared_client()

    def classify(self, customer_message: str) -> IntentClassificationResult:
        categories = [e.value for e in IntentCategory]
        schema_json = json.dumps(IntentClassificationResult.model_json_schema(), indent=2)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a customer intent classification system.\n"
                    f"Classify the input query into a primary intent (`predicted_intent`) and optionally a secondary intent (`secondary_intent`) from these categories: {categories}\n"
                    "If the query contains compound/multiple requests, emit the secondary intent; otherwise set `secondary_intent` to null.\n"
                    "Respond STRICTLY in JSON matching this JSON schema:\n"
                    f"{schema_json}"
                )
            },
            {"role": "user", "content": f"Customer Message: '{customer_message}'"}
        ]

        raw_text = self.llm.chat_completion(messages, max_tokens=250, temperature=0.0, json_mode=True)

        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        # Regex fallback to isolate JSON object if surrounding commentary exists
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)

        try:
            parsed_json = json.loads(raw_text)
            return IntentClassificationResult(**parsed_json)
        except Exception:
            # Resilient fallback: regex match predicted intent category from output
            detected_intent = IntentCategory.GENERAL_INQUIRY
            for cat in IntentCategory:
                if cat.value.lower() in raw_text.lower():
                    detected_intent = cat
                    break
            
            # Try to extract confidence score
            conf_match = re.search(r'confidence_score"?\s*:\s*([0-9]*\.?[0-9]+)', raw_text)
            conf = float(conf_match.group(1)) if conf_match else 0.75

            return IntentClassificationResult(
                predicted_intent=detected_intent,
                confidence_score=min(max(conf, 0.0), 1.0),
                reasoning="Extracted via resilient heuristic fallback from LLM response."
            )