"""
LLM prompting utilities.

Builds a grounded security explanation prompt from explainability output and
calls the configured LLM provider using the shared project .env API key.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from .llm_config import API_PROVIDER, API_KEY_ENV_VAR, ATTACK_SIGNATURES

# Load the project root .env file regardless of current working directory.
DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)


def build_prompt(explanation: dict, class_names: list) -> str:
    """
    Builds a grounded prompt from the structured explanation dict produced by
    the explainability module's explain_flagged_event().
    """
    pred_class_name = class_names[explanation["predicted_class"]]
    confidence_pct = explanation["confidence"] * 100
    signature = ATTACK_SIGNATURES.get(pred_class_name, "an anomalous traffic pattern")

    attended = explanation.get("top_attended_timesteps", [])
    attended_str = ", ".join([f"timestep {pos} (weight {score:.2f})" for pos, score in attended])

    shap_features = explanation.get("top_shap_features", [])
    shap_str = ", ".join([
        f"{f['feature']} at timestep {f['timestep']} (impact {f['shap_value']:.3f})"
        for f in shap_features
    ]) if shap_features else "not computed for this event"

    prompt = f"""You are a security analyst assistant. Explain, in 2-3 plain-English sentences, why a network flow was flagged.

Detected class: {pred_class_name}
Model confidence: {confidence_pct:.1f}%
Known signature of this attack type: {signature}
Most attended timesteps in the sequence: {attended_str}
Most influential features (SHAP): {shap_str}

Write a clear, concise explanation a non-expert security team member could understand. Do not repeat raw numbers verbatim; describe what they mean. Do not speculate beyond what the data shows."""

    return prompt


def call_llm_api(prompt: str) -> str:
    """
    Calls the configured LLM API with the built prompt. Reads the API key from the
    environment (populated from the shared root .env file) and returns the model text.
    """
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise EnvironmentError(
            f"'{API_KEY_ENV_VAR}' not found in environment. Make sure a .env file exists "
            f"at the project root with a line like: {API_KEY_ENV_VAR}=your_key_here"
        )

    if API_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    elif API_PROVIDER == "openai":
        import openai

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    elif API_PROVIDER == "openrouter":
        import openai

        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        # Placeholder free/cheap model choice; verify the current OpenRouter free-tier model
        # list in the OpenRouter docs/dashboard before using this in production.
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unsupported API_PROVIDER: {API_PROVIDER}")
