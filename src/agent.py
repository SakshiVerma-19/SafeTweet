import sys
import os
import re
import json
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient
from src.intent import IntentClassifier
from src.rag import RAGRetriever

class ExecutionResult(BaseModel):
    customer_message: str
    predicted_intent: str
    secondary_intent: str | None = None
    confidence_score: float
    escalated: bool
    escalation_reason: str | None = None
    retrieved_context: list[dict] | None = None
    final_response: str
    is_clarification_asked: bool = False
    clarification_question: str | None = None

class SupportAgent:
    def __init__(self, llm_client: LLMClient = None):
        print("Initializing Support Agent Pipeline...")
        
        # 1. Initialize LLM Client (Groq or local fallback)
        self.llm = llm_client or LLMClient.get_shared_client()

        # Pass LLM Client to IntentClassifier
        self.intent_classifier = IntentClassifier(llm_client=self.llm)
        
        # 2. RAG Retriever (resolve paths relative to project root)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(base_dir, "chroma_db")
        sample_path = os.path.join(base_dir, "data", "raw_sample.csv")

        self.rag_retriever = RAGRetriever(db_path=db_path)
        self.rag_retriever.populate_index(sample_path)

        # 3. Guardrails
        self.pii_patterns = [
            r'\b(?:\d[ -]*?){13,16}\b',                                            # Credit Cards
            r'\b\d{3}-\d{2}-\d{4}\b',                                                # SSN
            r'\b(password|passcode|secret|api[-_ ]?key)\s*[:=]?\s*\S+\b'           # Credential Exposure
        ]
        self.legal_patterns = [
            r'\b(lawyer|lawsuit|sue|legal action|attorney|court|litigation|arbitration)\b'
        ]
        self.sentiment_patterns = [
            r'\b(scam|fraudulent|fraud|thieves|cheaters|disgusting|horrible|worst service)\b'
        ]
        self.sarcasm_hyperbole_patterns = [
            r'\b(jk|just kidding|lol|lmao|rofl|sarcasm|sarcastic|figuratively|hyperbole|not actually|not literally)\b',
            r'\b(sue if my \w+ is|gonna sue over a|suing for \$1)\b'
        ]

    def _is_figurative_legal_language(self, text: str) -> bool:
        """
        Lightweight sentiment/sarcasm gate to filter out figurative legal expressions.
        """
        for pattern in self.sarcasm_hyperbole_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_escalation(
        self, text: str, confidence_score: float, predicted_intent: str | None = None, secondary_intent: str | None = None, is_followup: bool = False
    ) -> tuple[bool, str | None]:
        for pattern in self.pii_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "PII_EXPOSURE_RISK"

        # Compound Ticket Routing Rule (Multi-label intent escalation)
        if secondary_intent is not None:
            return True, "COMPOUND_TICKET_DETECTED"

        for pattern in self.legal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Sentiment & Sarcasm Gate: Check if legal terminology is figurative/sarcastic
                if not self._is_figurative_legal_language(text):
                    return True, "LEGAL_RISK_DETECTED"

        # Sentiment Velocity Check: Extreme negative sentiment or hostility
        for pattern in self.sentiment_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "HIGH_NEGATIVE_SENTIMENT"

        if confidence_score < 0.65:
            reason = "LOW_INTENT_CONFIDENCE_PERSISTENT" if is_followup else "LOW_INTENT_CONFIDENCE"
            return True, reason

        # Context Trigger: Missing required order/tracking number for tracking status requests
        if predicted_intent == "Order/Tracking Status":
            has_order_id = bool(re.search(r'(#\s*[\w-]+|\bTRK[\w\d]+\b|\b\d{3}-\d{7}-\d{7}\b|\b\d{7,14}\b)', text, re.IGNORECASE))
            if not has_order_id and any(kw in text.lower() for kw in ["where", "package", "status", "delivery", "track", "lost"]):
                return True, "MISSING_ORDER_ID"

        return False, None

    def process_ticket(self, customer_message: str, previous_turn_message: str | None = None) -> ExecutionResult:
        is_followup = bool(previous_turn_message)
        effective_query = f"{previous_turn_message} (Clarification: {customer_message})" if is_followup else customer_message

        intent_res = self.intent_classifier.classify(effective_query)
        sec_intent_val = intent_res.secondary_intent.value if intent_res.secondary_intent else None

        should_escalate, reason_code = self._check_escalation(
            effective_query,
            intent_res.confidence_score,
            intent_res.predicted_intent.value,
            secondary_intent=sec_intent_val,
            is_followup=is_followup
        )

        # Task 5: Handle LOW_INTENT_CONFIDENCE with a 1-turn clarification attempt on initial turn
        if should_escalate and reason_code == "LOW_INTENT_CONFIDENCE" and not is_followup:
            clarifying_q = "Could you please clarify your request? E.g., provide your order ID, account details, or specific issue so we can assist you."
            return ExecutionResult(
                customer_message=customer_message,
                predicted_intent=intent_res.predicted_intent.value,
                secondary_intent=sec_intent_val,
                confidence_score=intent_res.confidence_score,
                escalated=False,
                escalation_reason=None,
                retrieved_context=None,
                final_response=clarifying_q,
                is_clarification_asked=True,
                clarification_question=clarifying_q
            )

        if should_escalate:
            return ExecutionResult(
                customer_message=customer_message,
                predicted_intent=intent_res.predicted_intent.value,
                secondary_intent=sec_intent_val,
                confidence_score=intent_res.confidence_score,
                escalated=True,
                escalation_reason=reason_code,
                retrieved_context=None,
                final_response=f"Ticket escalated to human queue. Reason: {reason_code}"
            )

        contexts = self.rag_retriever.retrieve_context(customer_message, intent=intent_res.predicted_intent.value, top_k=3)
        
        context_str = ""
        for i, ctx in enumerate(contexts, 1):
            context_str += f"\nExample {i}:\nCustomer: {ctx['historical_customer_query']}\nBrand: {ctx['historical_brand_response']}\n"

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are @AmazonHelp official Twitter support agent.\n"
                    "Draft a helpful, professional tweet response based ONLY on the verified policy history below.\n"
                    "Strict Constraints:\n"
                    "1. Response length MUST NOT exceed 240 characters.\n"
                    "2. Do not invent unverified claims or policies.\n\n"
                    f"Verified Policy History Context:\n{context_str}"
                )
            },
            {"role": "user", "content": customer_message}
        ]

        # Stage 1: Initial Generation Draft
        first_draft = self.llm.chat_completion(prompt_messages, max_tokens=120, temperature=0.0)

        # Stage 2: Character-budget-aware re-generation if draft exceeds 240 chars
        if len(first_draft) > 240:
            refine_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert tweet editor for @AmazonHelp.\n"
                        "Your task is to condense and rephrase the following draft response so that it is STRICTLY under 240 characters.\n"
                        "Preserve the key resolution steps and polite tone, but make it concise.\n"
                        "Output ONLY the final revised response text."
                    )
                },
                {"role": "user", "content": f"Draft response to condense:\n'{first_draft}'"}
            ]
            final_response = self.llm.chat_completion(refine_messages, max_tokens=100, temperature=0.0).strip()
            
            # Fallback character hard-cut if LLM still exceeds 240 chars
            if len(final_response) > 240:
                final_response = final_response[:237] + "..."
        else:
            final_response = first_draft

        return ExecutionResult(
            customer_message=customer_message,
            predicted_intent=intent_res.predicted_intent.value,
            secondary_intent=sec_intent_val,
            confidence_score=intent_res.confidence_score,
            escalated=False,
            escalation_reason=None,
            retrieved_context=contexts,
            final_response=final_response
        )

if __name__ == "__main__":
    agent = SupportAgent()
    sample_queries = [
        "My order #102-3948571 has been delayed for 3 days, can you please help check tracking?",
        "I was charged twice on my credit card 4111-2222-3333-4444! Fix this!",
        "Fix this now or I will contact my lawyer and sue you!"
    ]

    for q in sample_queries:
        print(f"\nQuery: {q}")
        res = agent.process_ticket(q)
        print(f"Intent: {res.predicted_intent} (Confidence: {res.confidence_score:.2f})")
        print(f"Escalated: {res.escalated} (Reason: {res.escalation_reason})")
        print(f"Response: {res.final_response}")