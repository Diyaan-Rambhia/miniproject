"""
LLM Plain-English Explanation Synthesizer Layer
"""

def generate_llm_explanation(explanation: dict, threat_score: float) -> str:
    """
    Synthesizes raw SHAP attributions, attention timesteps, and threat score into plain-English text.
    """
    pred_class = explanation.get("predicted_class", "Unknown Attack")
    confidence = explanation.get("confidence", 0.0) * 100.0 if explanation.get("confidence") is not None else 0.0
    top_timesteps = explanation.get("top_attended_timesteps", [])
    top_shap = explanation.get("top_shap_features", [])

    lines = []
    lines.append(f"Security Analysis Summary:")
    lines.append(f"- Final Threat Score: {threat_score:.1f}/100")
    lines.append(f"- Classification: {pred_class} ({confidence:.1f}% confidence)")

    if top_timesteps:
        time_str = ", ".join([f"Timestep {t[0]} (weight: {t[1]:.3f})" if isinstance(t, (tuple, list)) else str(t) for t in top_timesteps])
        lines.append(f"- Temporal Attention: Peak focus on {time_str}.")

    if top_shap:
        feat_strs = [f"'{f['feature']}' (SHAP: {f['shap_value']:+.4f})" for f in top_shap[:3]]
        lines.append(f"- Key Driving Features: {', '.join(feat_strs)}.")

    lines.append("\nAssessment:")
    if threat_score >= 70.0:
        lines.append(
            f"HIGH RISK THREAT DETECTED (Score {threat_score:.1f}/100): The multi-signal pipeline flagged this sequence. "
            f"Anomalous traffic behaviors detected primarily around flow timestep {top_timesteps[0][0] if top_timesteps and isinstance(top_timesteps[0], (tuple, list)) else '0'}. "
            f"Immediate isolation and security audit recommended."
        )
    elif threat_score >= 40.0:
        lines.append(
            f"MODERATE RISK SUSPICION (Score {threat_score:.1f}/100): Mild anomalies observed across flow feature metrics. "
            f"Recommended for security monitoring."
        )
    else:
        lines.append(
            f"LOW RISK / BENIGN (Score {threat_score:.1f}/100): Traffic sequence conforms to baseline normal network flow behavior."
        )

    return "\n".join(lines)
