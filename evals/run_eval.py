import sys
import os
import json
import argparse
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient
from src.agent import SupportAgent
from src.judge import LLMJudge
from evals.baselines import TrivialBaseline, SimpleZeroShotBaseline
from evals.human_vs_judge import calculate_calibration_metrics

ALL_INTENTS = [
    "Order/Tracking Status",
    "Cancellation/Refund Request",
    "Account Access/Authentication",
    "Billing/Payment Issue",
    "Service Outage/Technical Bug",
    "General Inquiry/Feedback"
]

INTENT_ABBR = {
    "Order/Tracking Status": "OT",
    "Cancellation/Refund Request": "CR",
    "Account Access/Authentication": "AA",
    "Billing/Payment Issue": "BP",
    "Service Outage/Technical Bug": "SO",
    "General Inquiry/Feedback": "GI"
}

def print_confusion_matrices(true_intents, pred_intents, true_escalations, pred_escalations):
    print("\n" + "=" * 65)
    print("           INTENT CLASSIFICATION CONFUSION MATRIX (6x6)")
    print("=" * 65)
    print("Legend: OT=Order/Tracking | CR=Cancel/Refund | AA=Account Auth")
    print("        BP=Billing/Pay   | SO=Service Outage | GI=General Inq\n")

    cm_intent = confusion_matrix(true_intents, pred_intents, labels=ALL_INTENTS)
    header = "True \\ Pred  |" + "".join([f" {INTENT_ABBR[i]:>4} " for i in ALL_INTENTS])
    print(header)
    print("-" * len(header))
    for i, intent in enumerate(ALL_INTENTS):
        row_str = f"{INTENT_ABBR[intent]:<11} |" + "".join([f" {cm_intent[i][j]:>4} " for j in range(len(ALL_INTENTS))])
        print(row_str)

    print("\n" + "=" * 65)
    print("            ESCALATION GUARDRAIL CONFUSION MATRIX (2x2)")
    print("=" * 65)
    cm_esc = confusion_matrix(true_escalations, pred_escalations, labels=[0, 1])
    tn, fp, fn, tp = cm_esc.ravel() if cm_esc.size == 4 else (0, 0, 0, 0)
    print(f"                      Predicted: Safe (Auto)   Predicted: Escalate")
    print(f"Actual: Safe (Auto)        {tn:>6} (TN)              {fp:>6} (FP - False Alarm)")
    print(f"Actual: Risk (Escalate)    {fn:>6} (FN - Leak!)      {tp:>6} (TP)")
    print("-" * 65)

def evaluate_system(name: str, runner_fn, golden_set: list[dict], judge: LLMJudge = None, is_production: bool = False) -> dict:
    true_intents = []
    pred_intents = []

    true_escalations = []
    pred_escalations = []

    human_grounding_scores = []
    judge_grounding_scores = []
    judge_tone_scores = []
    human_tone_scores = []

    print(f"\nEvaluating: [{name}] over {len(golden_set)} samples...")

    for i, item in enumerate(golden_set, 1):
        tweet = item["customer_tweet"]
        true_intent = item["true_intent"]
        expected_esc = item["expected_escalate"]

        # Run candidate system
        res = runner_fn(tweet)

        true_intents.append(true_intent)
        pred_intents.append(res.predicted_intent)

        true_escalations.append(1 if expected_esc else 0)
        pred_escalations.append(1 if res.escalated else 0)

        # Run LLM Judge only if not escalated
        if not res.escalated and judge is not None:
            ctx_text = ""
            if hasattr(res, "retrieved_context") and res.retrieved_context:
                for c in res.retrieved_context:
                    ctx_text += f"Past Customer: {c['historical_customer_query']} -> Resolution: {c['historical_brand_response']}\n"
            else:
                ctx_text = "No retrieved context available."

            try:
                j_eval = judge.evaluate_response(
                    customer_msg=tweet,
                    response=res.final_response,
                    retrieved_context=ctx_text.strip()
                )
                judge_grounding_scores.append(j_eval.grounding_score)
                judge_tone_scores.append(j_eval.tone_score)
                human_grounding_scores.append(item.get("human_grounding_score", 5))
                human_tone_scores.append(item.get("human_tone_score", 5))
            except Exception:
                pass

        if i % 25 == 0 or i == len(golden_set):
            print(f"  Processed {i}/{len(golden_set)} items...")

    intent_f1 = f1_score(true_intents, pred_intents, average="weighted", zero_division=0)
    esc_precision = precision_score(true_escalations, pred_escalations, zero_division=0)
    esc_recall = recall_score(true_escalations, pred_escalations, zero_division=0)

    avg_grounding = float(np.mean(judge_grounding_scores)) if judge_grounding_scores else 0.0
    avg_tone = float(np.mean(judge_tone_scores)) if judge_tone_scores else 0.0

    if len(human_grounding_scores) > 1 and len(judge_grounding_scores) > 1:
        calib = calculate_calibration_metrics(human_grounding_scores, judge_grounding_scores)
        kappa = calib["cohen_kappa"]
    else:
        kappa = 0.0

    if is_production:
        print_confusion_matrices(true_intents, pred_intents, true_escalations, pred_escalations)

    return {
        "name": name,
        "intent_f1": round(intent_f1, 2),
        "grounding_score": round(avg_grounding, 1) if avg_grounding > 0 else "N/A",
        "tone_score": round(avg_tone, 1) if avg_tone > 0 else "N/A",
        "escalation_precision": round(esc_precision, 2),
        "escalation_recall": round(esc_recall, 2),
        "judge_kappa": round(kappa, 2) if kappa != 0.0 else "N/A"
    }

def run_benchmark(samples_limit: int = None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    golden_set_path = os.path.join(base_dir, "data", "golden_set.json")

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    if samples_limit and samples_limit < len(golden_set):
        golden_set = golden_set[:samples_limit]

    print("=" * 75)
    print(f"[AUTONOMOUS AI] SUPPORT AGENT - MULTI-BASELINE BENCHMARK (N={len(golden_set)})")
    print("=" * 75)

    # 1. Initialize Shared LLM Client
    llm = LLMClient.get_shared_client()
    agent = SupportAgent(llm_client=llm)
    judge = LLMJudge(llm_client=llm)

    # 2. Initialize Baselines
    trivial_base = TrivialBaseline()
    simple_base = SimpleZeroShotBaseline(llm_client=llm)

    # 3. Evaluate Systems
    res_trivial = evaluate_system("Baseline 1: Trivial (Majority Intent + Canned Reply)",
                                  trivial_base.process_ticket, golden_set, judge=None)
    res_trivial["tone_score"] = 3.0

    res_simple = evaluate_system("Baseline 2: Simple (Zero-Shot No-RAG Prompt)",
                                 simple_base.process_ticket, golden_set, judge=judge)

    res_prod = evaluate_system("Production Pipeline (RAG + Guardrails + Structured Intent)",
                               agent.process_ticket, golden_set, judge=judge, is_production=True)

    # 4. Print Summary Comparison Table
    print("\n" + "=" * 85)
    print("                      EVALUATION BENCHMARK & BASELINE COMPARISON")
    print("=" * 85)
    print(f"{'Metric':<35} | {'Baseline 1 (Trivial)':<20} | {'Baseline 2 (Simple)':<20} | {'Production Pipeline':<20}")
    print("-" * 85)
    print(f"{'Intent F1-Score (Weighted)':<35} | {str(res_trivial['intent_f1']):<20} | {str(res_simple['intent_f1']):<20} | {str(res_prod['intent_f1']):<20}")
    print(f"{'Grounding Score (1.0-5.0)':<35} | {str(res_trivial['grounding_score']):<20} | {str(res_simple['grounding_score']):<20} | {str(res_prod['grounding_score']):<20}")
    print(f"{'Tone Alignment (1.0-5.0)':<35} | {str(res_trivial['tone_score']):<20} | {str(res_simple['tone_score']):<20} | {str(res_prod['tone_score']):<20}")
    print(f"{'Escalation Precision':<35} | {str(res_trivial['escalation_precision']):<20} | {str(res_simple['escalation_precision']):<20} | {str(res_prod['escalation_precision']):<20}")
    print(f"{'Escalation Recall':<35} | {str(res_trivial['escalation_recall']):<20} | {str(res_simple['escalation_recall']):<20} | {str(res_prod['escalation_recall']):<20}")
    print(f"{'Human vs. Judge Alignment (kappa)':<35} | {str(res_trivial['judge_kappa']):<20} | {str(res_simple['judge_kappa']):<20} | {str(res_prod['judge_kappa']):<20}")
    print("=" * 85)

    print("\n[Markdown Table for README.md & report.md]:")
    print("| Metric | Baseline 1: Trivial (Majority Intent + Canned) | Baseline 2: Simple (Zero-Shot No-RAG) | Production Pipeline (RAG + Guardrails + Structured Intent) |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| **Intent F1-Score (Weighted)** | {res_trivial['intent_f1']} | {res_simple['intent_f1']} | {res_prod['intent_f1']} |")
    print(f"| **Grounding Score (1.0-5.0)** | {res_trivial['grounding_score']} | {res_simple['grounding_score']} | {res_prod['grounding_score']} |")
    print(f"| **Tone Alignment (1.0-5.0)** | {res_trivial['tone_score']} | {res_simple['tone_score']} | {res_prod['tone_score']} |")
    print(f"| **Escalation Precision** | {res_trivial['escalation_precision']} | {res_simple['escalation_precision']} | {res_prod['escalation_precision']} |")
    print(rf"| **Human vs. Judge Alignment (kappa)** | {res_trivial['judge_kappa']} | {res_simple['judge_kappa']} | {res_prod['judge_kappa']} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Support Agent Benchmarks")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples to evaluate (default: full 200 golden set)")
    args = parser.parse_args()

    run_benchmark(samples_limit=args.samples)
