import os
import ollama
import pandas as pd

# =====================================================
# MODEL
#
# Configurable via UAV_LLM_MODEL so this doesn't require
# editing source to try a different local Ollama model.
# =====================================================

MODEL_NAME = os.environ.get("UAV_LLM_MODEL", "llama3.2")
# =====================================================
# DIRECT SHAP -> LLM REPORT
#
# Skips reasoning_engine.py entirely. The LLM is given the raw
# top-N SHAP features (name, measured value, signed contribution)
# for this one prediction and does its own interpretation instead
# of rephrasing a fixed per-feature template. This is what makes
# it genuinely explainable-by-the-model rather than explainable-by-
# a-lookup-table: the LLM has to reconcile "high orb_features" with
# whatever sign SHAP actually gave it for THIS frame, instead of
# reciting a canned "more features = better" line regardless of
# direction.
#
# Guardrail: the LLM is still told to ground every claim in the
# supplied numbers only (no inventing features/values/signs that
# aren't in the list) — it's free to reason about *why*, not free
# to make up *what*.
# =====================================================

DIRECT_SYSTEM_PROMPT = """
You are an Explainable AI (XAI) assistant for a UAV visual-localization system.

You are given the model's predicted localization error for one video frame
and a short list of the features that most influenced that specific
prediction, each with its measured value, its rank by influence, and its
direction (whether it pushed the predicted error up or down).

Your job is to write TWO sections of a report: "Reasons for Localization
Error" and "Recommendations". Someone else has already written the summary
and risk level — do not write those, and do not repeat or restate the
overall ranking of features; just explain each one.

For each feature, explain in your own words what it means and why it
plausibly had the effect shown. If a feature's direction seems
counter-intuitive, say so explicitly rather than papering over it.

Hard rules:

- Only use the features, values, and directions given to you. Never invent
  a feature, value, or direction that wasn't provided, and never contradict
  the direction given for any feature.
- Where a feature has a "Magnitude (given, do not re-judge)" line, that
  label (SMALL/MODERATE/LARGE etc.) is a pre-computed fact. Never call a
  value "large" if its label says SMALL, or vice versa — use the given
  label instead of judging the raw number yourself.
- You may propose a plausible mechanism for WHY a feature had its effect
  (this is expected — it's the point of the reasoning section). But you do
  not actually know the true mechanism, only the measured value and its
  direction. Phrase every proposed mechanism as a hedge, not a fact:
  use "this may suggest...", "one possible explanation is...", "this
  could indicate...", etc. Never state a proposed mechanism in plain
  declarative form (e.g. NOT "this means the scene was cluttered" — INSTEAD
  "this could suggest the scene was more cluttered, though the data here
  doesn't confirm that"). The measured value, its direction, and its rank
  are facts and should be stated plainly; everything about WHY is a
  hypothesis and must read as one.
- The numbers given are NOT meters and are not directly comparable to the
  overall predicted error in meters. Never state an individual feature's
  contribution as if it were a distance in meters.
- Never invent a cause, mechanism, or fix that isn't implied by the
  features you were given. In particular, do not invent claims or
  recommendations about UAV speed, flight path, weather, obstacles,
  camera resolution, lighting, altitude, orientation, hardware, or
  alternative algorithms/techniques — unless one of the given features
  directly states it. This is unconditional: it applies to both sections
  you write, with no exceptions. A recommendation may only reference the
  specific features and values you were given (e.g. "reduce sideways
  motion" is fine because dx was given; "increase camera resolution" is
  not, because camera resolution was never given). If you don't have
  enough information to explain WHY a feature has the effect it does, say
  the effect is observed without asserting an unverified mechanism for it.
- Every recommendation must follow logically from the features you were
  given for this frame.
- Never mention "SHAP", "Random Forest", "machine learning", "SLAM",
  "ORB-SLAM", "SIFT", "SURF", or any specific algorithm name — write for a
  non-technical UAV operator. This project uses ORB feature matching
  (not SLAM); if you need to refer to it, call it "visual feature
  matching" only if a given feature name (e.g. orb features) requires it.
- Be concise, professional, and direct.

Output format — return ONLY these two sections, nothing else:

Reasons for Localization Error

Recommendations
"""


def compute_risk_level(prediction):
    if prediction < 0.30:
        return "Low (error < 0.30 m)"
    elif prediction <= 0.80:
        return "Medium (0.30-0.80 m)"
    else:
        return "High (> 0.80 m)"


def build_summary_section(prediction, top_features, risk_level):
    """
    Written entirely in Python, not by the LLM. In testing, the LLM
    reliably got this part wrong — misstating which feature was the
    biggest driver, or contradicting the direction it was given
    elsewhere in the same report — no matter how the prompt was worded.
    The ranking and direction are simple lookups, not something that
    needs "reasoning," so there's no reason to expose them to the
    model's failure mode. Only the free-text explanation of WHY (in
    build_direct_prompt / the LLM call) benefits from an LLM.
    """

    primary = top_features[0]
    direction = "decreases" if primary["shap"] < 0 else "increases"

    secondary_clause = ""
    if len(top_features) > 1:
        second = top_features[1]
        second_direction = "decreases" if second["shap"] < 0 else "increases"
        secondary_clause = (
            f" The next largest factor is {second['name']}, which "
            f"{second_direction} the predicted error."
        )

    return (
        f"Localization Summary\n\n"
        f"Predicted localization error: {prediction:.3f} meters "
        f"(Risk Level: {risk_level}).\n\n"
        f"The single largest driver of this prediction is {primary['name']}, "
        f"which {direction} the predicted error.{secondary_clause}"
    )


def describe_feature(raw_name):
    """
    What each raw feature column actually measures, in plain terms.
    Without this, the LLM has been observed guessing wrong (e.g. reading
    a segmentation percentage like "Carpet: 1.4" as a spatial
    "displacement" because it appeared near dx/dy/dz, which really are
    displacements). Giving it the unit/meaning up front removes the need
    to guess.
    """

    if raw_name.startswith("class_") and raw_name.endswith("_percent"):
        return "percentage of the visible camera image occupied by this object/surface"
    if raw_name in ("dx", "dy", "dz"):
        axis = raw_name[-1].upper()
        return (
            f"the UAV's movement along the {axis} axis between this frame "
            f"and the next, in meters (the physical direction this axis "
            f"points - e.g. forward, sideways, up - is not specified in "
            f"this data; do not assert a specific direction like "
            f"'forward' or 'sideways')"
        )
    if raw_name == "orb_features":
        return "the number of distinct visual tracking keypoints detected in the image"
    if raw_name == "blur":
        return "an image sharpness score (higher = sharper, less motion blur)"
    if raw_name == "contrast":
        return "an image contrast score"
    if raw_name == "brightness":
        return "average image brightness (0-255 grayscale scale)"
    if raw_name == "edge_density":
        return "fraction of image pixels detected as edges"
    if raw_name in ("mean_depth", "std_depth", "min_depth", "max_depth", "depth_range", "depth_entropy", "valid_depth_ratio"):
        return "a statistic computed from the depth image (not a spatial displacement)"
    return "a computed image/motion statistic (exact unit not specified — do not guess a unit for it)"


# Terms observed in testing where the LLM invents unsupported technical
# fixes despite being told not to (e.g. "adjust algorithm parameters",
# "increase processing power", "apply positional thresholds"). Prompt
# wording alone hasn't reliably stopped this across repeated runs, so
# recommendation lines containing these are filtered out in Python after
# generation rather than trusted to the model.
_UNGROUNDED_RECOMMENDATION_TERMS = [
    "algorithm", "processing power", "camera resolution", "lighting",
    "hardware", "threshold", "filter", "slam", "software", "firmware",
    "calibrat", "sensor fusion", "gps",
]


def sanitize_recommendations(llm_sections):
    """
    Strips any Recommendations bullet line that mentions an ungrounded
    technical fix (see _UNGROUNDED_RECOMMENDATION_TERMS), and notes how
    many were removed instead of silently deleting them.
    """

    lines = llm_sections.splitlines()
    out_lines = []
    removed = 0
    in_recommendations = False

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("recommendations"):
            in_recommendations = True
            out_lines.append(line)
            continue

        if in_recommendations and stripped.startswith(("-", "*", "•")):
            lower = stripped.lower()
            if any(term in lower for term in _UNGROUNDED_RECOMMENDATION_TERMS):
                removed += 1
                continue

        out_lines.append(line)

    result = "\n".join(out_lines)

    if removed:
        result += (
            f"\n\n[Note: {removed} recommendation(s) were removed because "
            f"they referenced equipment, algorithms, or settings not "
            f"present in the given features.]"
        )

    return result


def describe_magnitude(raw_name, value):
    """
    Qualitative label for the raw measured value, computed in Python.
    Added because the LLM has now misjudged magnitude twice in testing —
    once ranking SHAP contributions out of order, and separately calling
    a 1.4% scene-coverage value "large" (it was the smallest of the five
    values shown). Same fix as the ranking bug: state the judgment as a
    given fact instead of asking the model to eyeball raw numbers.
    """

    if raw_name.startswith("class_") and raw_name.endswith("_percent"):
        if value < 2:
            return "a SMALL fraction of the image"
        elif value < 10:
            return "a MODERATE fraction of the image"
        else:
            return "a LARGE fraction of the image"

    if raw_name in ("dx", "dy", "dz"):
        av = abs(value)
        if av < 0.05:
            return "a SMALL movement"
        elif av < 0.3:
            return "a MODERATE movement"
        else:
            return "a LARGE movement"

    if raw_name == "orb_features":
        if value < 1500:
            return "a LOW number of keypoints"
        elif value < 4000:
            return "a MODERATE number of keypoints"
        else:
            return "a HIGH number of keypoints"

    return None


def build_direct_prompt(prediction, top_features):
    """
    top_features: list of dicts, each with
        name     : human-readable feature name
        value    : measured value (float)
        shap     : signed SHAP contribution (float)
        raw_name : original column name, e.g. "class_180_percent" or "dx"
                   (used to look up a plain-English description of what
                   the value actually measures — see describe_feature)

    Only asks the LLM for the "Reasons" and "Recommendations" sections.
    The ranking/direction facts are stated here as given information
    (not asked of the model), and the summary + risk level are written
    separately in Python (see build_summary_section) and stitched in
    afterward — see generate_report_direct.
    """

    prompt = f"""
Predicted Localization Error

{prediction:.3f} meters

Features influencing this specific prediction, already ranked by
influence (item 1 = largest impact, do not reorder or re-rank them):

"""

    for i, f in enumerate(top_features, start=1):
        direction = "increases" if f["shap"] > 0 else "decreases"
        description = describe_feature(f.get("raw_name", ""))
        magnitude_label = describe_magnitude(f.get("raw_name", ""), f["value"])
        magnitude_line = (
            f"   Magnitude (given, do not re-judge)  : {magnitude_label}\n"
            if magnitude_label else ""
        )
        prompt += (
            f"\n{i}. {f['name']}\n"
            f"   What this measures : {description}\n"
            f"   Measured value      : {f['value']:.3f}\n"
            f"{magnitude_line}"
            f"   Direction           : {direction} the predicted error "
            f"(it does NOT {'decrease' if direction == 'increases' else 'increase'} it)\n"
        )

    prompt += """

====================================================

Write ONLY these two sections, using ONLY the features above:

Reasons for Localization Error
- For each feature above, explain in your own words what it means and
  why it plausibly had the effect shown. If the direction is
  counter-intuitive, say so explicitly.

Recommendations
- Practical, operational recommendations that follow from the features
  above. Every recommendation must name or clearly reference one of the
  features given above. Do not recommend anything involving equipment,
  settings, or techniques that weren't given as a feature.

Return ONLY these two sections. Do not write a summary, do not write a
risk level, do not restate which feature is largest/smallest — that has
already been handled elsewhere.
"""

    return prompt


def generate_report_direct(prediction, top_features, risk_level=None):
    """
    Assembles the final report from a deterministic Python-written
    Summary + Risk Level, and an LLM-written Reasons + Recommendations.
    See build_summary_section for why the summary isn't left to the LLM.
    """

    if risk_level is None:
        risk_level = compute_risk_level(prediction)

    summary_section = build_summary_section(prediction, top_features, risk_level)

    prompt = build_direct_prompt(prediction, top_features)
    llm_sections = _chat(DIRECT_SYSTEM_PROMPT, prompt).strip()
    llm_sections = sanitize_recommendations(llm_sections)

    return (
        "====================================================\n\n"
        f"{summary_section}\n\n"
        "====================================================\n\n"
        f"{llm_sections}\n\n"
        "====================================================\n\n"
        f"Risk Level\n\n{risk_level}\n\n"
        "===================================================="
    )


# =====================================================
# SHARED OLLAMA CALL
# =====================================================

def _chat(system_prompt, user_prompt):

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
    except Exception as exc:
        # Previously an unreachable/missing Ollama server (very likely,
        # since it must be installed and running locally with the
        # model already pulled) would crash the whole pipeline with a
        # raw connection traceback. Fail with a clear, actionable
        # message instead, and still return the verified facts so the
        # rest of the pipeline's output isn't lost.
        raise RuntimeError(
            f"Could not reach the local Ollama server to generate the "
            f"report (model='{MODEL_NAME}'). Make sure Ollama is "
            f"installed, running, and that you've pulled the model "
            f"(`ollama pull {MODEL_NAME}`).\n"
            f"Underlying error: {exc}"
        ) from exc

    return response["message"]["content"]

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    from class_names import get_class_name

    dummy_features = [
        ("orb_features", 4400, -0.19),
        ("dx", -0.25, 0.07),
        ("class_180_percent", 1.4, 0.02),
        ("class_28_percent", 12.8, 0.05),
        ("blur", 620, 0.11),
    ]
    dummy_features.sort(key=lambda t: abs(t[2]), reverse=True)

    top_features = [
        {"name": get_class_name(f), "value": v, "shap": s, "raw_name": f}
        for f, v, s in dummy_features
    ]

    report = generate_report_direct(
        prediction=0.445,
        top_features=top_features
    )

    print("\n")
    print("=" * 70)
    print("LLM GENERATED XAI REPORT (direct SHAP interpretation)")
    print("=" * 70)
    print(report)
    print("=" * 70)