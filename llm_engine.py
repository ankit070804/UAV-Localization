import ollama
import pandas as pd

# =====================================================
# MODEL
# =====================================================

MODEL_NAME = "llama3.2"

# =====================================================
# SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = """
You are an Explainable AI assistant for UAV localization.

You receive VERIFIED FACTS from another reasoning engine.

Your job is ONLY to convert those verified facts into a professional UAV localization report.

Rules:

- Never invent reasons.
- Never invent recommendations.
- Never mention SHAP.
- Never mention Random Forest.
- Never mention machine learning.
- Never mention ORB-SLAM.
- Never recommend SIFT or SURF.
- Never contradict the verified facts.
- Explain everything in simple English.
- Be concise and professional.

Output format:

====================================================

Localization Summary

Reasons for Localization Error

Recommendations

Risk Level

====================================================
"""

# =====================================================
# BUILD PROMPT
# =====================================================

def build_prompt(prediction, facts, recommendations):

    prompt = f"""
Predicted Localization Error

{prediction:.3f} meters

The following information has already been verified by the Explainable AI reasoning engine.

IMPORTANT RULES

• Use ONLY the verified facts below.
• Do NOT invent new causes.
• Do NOT add algorithms.
• Do NOT contradict the verified facts.

====================================================

VERIFIED FACTS

"""

    for i, fact in enumerate(facts, start=1):

        prompt += f"""

Feature {i}

Feature Name :
{fact['feature']}

Measured Value :
{fact['value']:.3f}

Observation :
{fact['observation']}

Domain Meaning :
{fact['domain_meaning']}

Model Interpretation :
{fact['model_influence']}

Contribution :
{fact['direction']}

"""

    prompt += """

====================================================

Verified Recommendations

"""

    for r in recommendations:
        prompt += f"- {r}\n"

    prompt += """

====================================================

Generate a professional UAV localization report.

The report must contain these sections:

1. Localization Summary

- Mention the predicted localization error.
- Give a brief overall summary.

2. Reasons for Localization Error

For every verified feature:

- Explain the observation.
- Explain why it matters.
- Use only the supplied facts.

3. Recommendations

Use ONLY the verified recommendations.

4. Risk Level

Assign

Low    : error < 0.30 m
Medium : 0.30–0.80 m
High   : >0.80 m

Return ONLY the report.

"""

    return prompt

# =====================================================
# GENERATE REPORT
# =====================================================

def generate_report(prediction, facts, recommendations):

    prompt = build_prompt(
        prediction,
        facts,
        recommendations
    )

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    dummy = pd.DataFrame({

        "Feature":[
            "orb_features",
            "blur",
            "contrast",
            "brightness",
            "class_28_percent",
            "class_180_percent",
            "mean_depth",
            "edge_density",
            "dx"
        ],

        "Value":[
            4400,
            620,
            54,
            152,
            12.8,
            1.4,
            2.10,
            0.071,
            -0.25
        ],

        "SHAP":[
            -0.19,
             0.11,
             0.08,
            -0.03,
             0.05,
             0.02,
            -0.01,
             0.03,
             0.07
        ]

    })

    from reasoning_engine import explain

    facts, recommendations = explain(dummy)

    report = generate_report(
        prediction=0.445,
        facts=facts,
        recommendations=recommendations
    )

    print("\n")
    print("=" * 70)
    print("LLM GENERATED XAI REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70)
