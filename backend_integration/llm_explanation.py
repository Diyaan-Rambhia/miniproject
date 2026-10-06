"""Compatibility shim for the project's canonical LLM explanation entry point."""

from models.llm_layer.llm_explanation import explain


def generate_llm_explanation(explanation: dict, threat_score: float) -> str:
    """Backward-compatible wrapper around the real llm_layer implementation."""
    class_names = ["BENIGN", "ATTACK"]
    predicted = explanation.get("predicted_class", 0)
    if isinstance(predicted, str):
        class_names = [predicted]
    return explain({**explanation, "threat_score": threat_score}, class_names)
