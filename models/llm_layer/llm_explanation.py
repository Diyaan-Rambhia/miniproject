"""
LLM explanation entry point.

Provides a single explain() function for external modules to call. This
is the only file other project code should import from within the llm_layer.
"""

try:
    from .llm_prompting import build_prompt, call_llm_api
except ImportError:
    from llm_prompting import build_prompt, call_llm_api


def explain(explanation: dict, class_names: list) -> str:
    """
    Builds the prompt from the structured explanation dict and returns the LLM's
    plain-English explanation.
    """
    prompt = build_prompt(explanation, class_names)
    return call_llm_api(prompt)
