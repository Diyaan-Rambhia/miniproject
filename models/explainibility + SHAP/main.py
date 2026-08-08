"""
Phase: Pipeline Orchestration / Entry Point
Expects: Imports from phase files in this directory (config, model, attention_explainer, shap_explainer, explainer)
Outputs: Runs verification check for explainability module and prints SHAP availability status
"""

import config
from model import ExplainableFlowTransformer
from attention_explainer import summarize_attention
from shap_explainer import top_shap_features
from explainer import explain_flagged_event


def main():
    print("Explainability Module (SHAP + Attention Weights)")
    print(f"Device: {config.DEVICE}")
    print(f"SHAP Available: {config.SHAP_AVAILABLE}")
    print("Use explain_flagged_event() to generate dual-channel explanations for flagged sequences.")


if __name__ == "__main__":
    main()
