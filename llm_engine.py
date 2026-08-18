import os
import re
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
  specific features and values you were given (e.g. "reduce motion along
  dx" is fine because dx was given; "increase camera resolution" is
  not, because camera resolution was never given). If you don't have
  enough information to explain WHY a feature has the effect it does, say
  the effect is observed without asserting an unverified mechanism for it.
- Never invent a system component, mechanism, or internal process that
  wasn't given to you as a feature — e.g. do not refer to a "compensation
  mechanism", "error model", "correction system", or any other internal
  process of the localization system. You were only given feature values
  and their effect on the prediction; you were not given, and must not
  assume, anything about how the system internally works.
- Never invent a physical property of an object that wasn't given to you.
  A segmentation percentage (e.g. "Carpet: 1.4") only tells you how much
  of the image an object occupies — it does not tell you the object is
  "solid", "reflective", "cluttered", or anything else about its physical
  nature. Do not assert such properties even as a hedge.
- Do not invent a rule about what a value "should" do. You are only told
  the actual direction (increases/decreases) for each feature — you were
  not given, and must not assume, any general rule like "a negative value
  should increase the error." If a direction feels surprising, say so
  plainly ("this direction may seem counter-intuitive") without inventing
  a reason it should have gone the other way.
- Every recommendation must follow logically from the features you were
  given for this frame.
- Never mention "SHAP", "Random Forest", "machine learning", "SLAM",
  "ORB-SLAM", "SIFT", "SURF", or any specific algorithm name — write for a
  non-technical UAV operator. This project uses ORB feature matching
  (not SLAM); if you need to refer to it, call it "visual feature
  matching" only if a given feature name (e.g. orb features) requires it.
- Be concise, professional, and direct.

Here are worked examples showing the difference between grounded and
ungrounded reasoning. Study the BAD example closely — every sentence in
it is a realistic mistake seen in previous runs of this exact prompt.
Note the GOOD example is a single hedged sentence only — it does not
restate the value, magnitude, or direction, because those are already
stated elsewhere in the final report and are not your job to write.

GOOD example (a percentage-of-image feature, "class_87_percent" /
"Window", value 1.932, SMALL, direction: increases error):

  "This may suggest the visual matching process finds this particular
  area harder to track consistently between frames, though the
  data here doesn't say why — only that its presence had this effect."

  Why this is good: it proposes a mechanism but clearly hedges it, never
  restates the value/magnitude/direction (already handled elsewhere),
  and does not claim to know anything about the window's physical
  properties (reflectivity, size, obstruction, etc.) beyond what it was
  given.

BAD example (do not write like this):

  "Window: A small fraction of 1.932 suggests the window is a small,
  clear opening, and this increased the predicted error. Since a
  negative value should typically increase error, and this instead
  decreases it, the system's error-compensation mechanism may be
  correcting for this automatically. Consider adjusting the system's
  handling of reflective, unobstructed surfaces."

  Why this is bad, sentence by sentence: it restates the value and
  direction ("this increased the predicted error") which is not your
  job and risks contradicting the direction actually given for this
  feature. "the window is a small, clear opening" invents a physical
  property never given. "a negative value should typically increase
  error" invents a rule that was never stated — you are only ever given
  the actual direction, never a rule about what a direction "should" be.
  "error-compensation mechanism" invents an internal system component
  that doesn't exist in what you were given. The recommendation about
  "reflective, unobstructed surfaces" invents object properties and
  follows from the invented mechanism, not from the actual feature.

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
    "compensation mechanism", "error model", "correction system",
    "compensate", "over-compensat", "overcompensat",
    # Same invented-physical-property terms blocked in Reasons hedges via
    # _UNGROUNDED_HEDGE_TERMS. A recommendation can invent a property
    # ("adjust handling of reflective... surfaces") just as easily as a
    # hedge sentence can — observed in testing ("reflective" survived in
    # Recommendations even though it's already blocked in Reasons).
    "reflective", "unobstructed", "clear opening", "solid object",
    "transparent", "translucent", "opaque",
]


_LIST_ITEM_START_PATTERN = re.compile(r"^([-*•]|\d+[.)])\s*")


def _is_list_item_start(stripped_line):
    """
    True if this line opens a new list item — a dash/asterisk/bullet
    ("-", "*", "•") OR a numbered marker ("1.", "2)", etc.). The LLM has
    been observed switching to numbered Recommendations ("1. ... 2. ...")
    without being asked to, and every filter here used to only recognize
    "-"/"*"/"•" as a bullet start. That meant a numbered list bypassed
    _group_into_bullets' grouping AND sanitize_recommendations' gate
    entirely — confirmed in testing, where a numbered recommendation
    containing "threshold" and "filter" (both blocked terms) survived
    completely unfiltered because it never matched the old check.
    """
    return bool(_LIST_ITEM_START_PATTERN.match(stripped_line))


def _group_into_bullets(lines):
    """
    Groups lines into chunks so a wrapped multi-line bullet/numbered item
    is treated as one unit instead of separate physical lines. Each chunk
    is either:
      - a list item: starts with "-"/"*"/"•" or a numbered marker like
        "1.", plus any following lines that wrap it (no marker,
        non-blank), or
      - a single non-item line (headers, blank lines, plain prose).

    Both sanitize_recommendations and strip_direction_claims used to test
    and remove one physical line at a time. That broke two ways in
    testing: (1) removing a flagged opening line left its wrapped
    continuation behind as an orphan fragment with no marker, and
    (2) worse, when the flagged word landed on a *different* physical
    line than the trigger term (e.g. "dx" on line 1, "backwards" on the
    wrapped line 2), neither line matched on its own and the whole item
    slipped through unfiltered. Testing an item's full joined text as one
    unit fixes both failure modes at once.
    """

    chunks = []
    current = None

    for line in lines:
        stripped = line.strip()
        is_item_start = _is_list_item_start(stripped)

        if is_item_start:
            if current is not None:
                chunks.append(current)
            current = [line]
        elif stripped == "":
            if current is not None:
                chunks.append(current)
                current = None
            chunks.append([line])
        else:
            if current is not None:
                current.append(line)
            else:
                chunks.append([line])

    if current is not None:
        chunks.append(current)

    return chunks


def sanitize_recommendations(llm_sections):
    """
    Strips any Recommendations item (bulleted OR numbered — see
    _is_list_item_start) that mentions an ungrounded technical fix (see
    _UNGROUNDED_RECOMMENDATION_TERMS), and notes how many were removed
    instead of silently deleting them. Operates on whole items via
    _group_into_bullets — see its docstring for why.
    """

    lines = llm_sections.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("recommendations"):
            header_idx = i
            break

    if header_idx is None:
        return llm_sections

    before = lines[: header_idx + 1]
    after_chunks = _group_into_bullets(lines[header_idx + 1 :])

    out_after = []
    removed = 0

    for chunk in after_chunks:
        first = chunk[0].strip()
        if _is_list_item_start(first):
            joined = " ".join(l.strip() for l in chunk).lower()
            if any(term in joined for term in _UNGROUNDED_RECOMMENDATION_TERMS):
                removed += 1
                continue
        out_after.extend(chunk)

    result = "\n".join(before + out_after)

    if removed:
        result += (
            f"\n\n[Note: {removed} recommendation(s) were removed because "
            f"they referenced equipment, algorithms, or settings not "
            f"present in the given features.]"
        )

    return result


# describe_feature() explicitly tells the model that the real-world
# direction dx/dy/dz point (forward, sideways, up, ...) is NOT specified
# in this data and that it must not assert one. In testing, llama3.2
# ignored that rule anyway — and even contradicted itself between
# sections, e.g. calling the same dx value "backwards" in Reasons and
# "sideways" in Recommendations within the same report. Prompt wording
# alone hasn't reliably stopped this (same story as
# _UNGROUNDED_RECOMMENDATION_TERMS below), so it's filtered here too.
_AXIS_PATTERN = re.compile(
    r"\b(dx|dy|dz|[xyz]\s*[- ]?\s*axis)\b",
    re.IGNORECASE,
)
_DIRECTION_TERMS_PATTERN = re.compile(
    r"\b(forward(s)?|backward(s)?|sideways|lateral(ly)?|leftward(s)?|"
    r"rightward(s)?|upward(s)?|downward(s)?|horizontal(ly)?|vertical(ly)?)\b",
    re.IGNORECASE,
)


def strip_direction_claims(llm_sections):
    """
    Strips any bullet that both (a) references the dx/dy/dz motion axes
    and (b) asserts a specific physical direction for them, anywhere in
    the bullet's full (wrapped) text. Notes how many were removed instead
    of silently deleting them. Operates on whole bullets via
    _group_into_bullets — see its docstring for why (this filter used to
    test one physical line at a time, which could miss a claim entirely
    when the axis name and the direction word wrapped onto different
    lines).
    """

    chunks = _group_into_bullets(llm_sections.splitlines())
    out = []
    removed = 0

    for chunk in chunks:
        joined = " ".join(l.strip() for l in chunk)
        if _AXIS_PATTERN.search(joined) and _DIRECTION_TERMS_PATTERN.search(joined):
            removed += 1
            continue
        out.extend(chunk)

    result = "\n".join(out)

    if removed:
        result += (
            f"\n\n[Note: {removed} line(s) were removed because they "
            f"asserted a specific physical direction (e.g. 'forward', "
            f"'sideways') for a motion axis whose real-world direction "
            f"isn't specified in the data.]"
        )

    return result


def build_fact_sentence(f):
    """
    States what's actually known about one feature as plain fact, written
    entirely in Python: what it measures, its value, its magnitude label,
    and its direction. This is the same idea as build_summary_section,
    extended to every feature instead of just the top one — because
    testing showed the LLM will occasionally state the WRONG direction
    for a feature (e.g. saying dx "increased" the error when it was
    given as decreasing it), directly contradicting the facts it was
    handed. A keyword filter can't reliably catch a wrong direction
    without also flagging correctly-hedged, correctly-signed sentences
    (e.g. "unlikely to have increased... it decreases" is CORRECT but
    contains the word "increased"). The only reliable fix is to never
    let the model state the direction at all — see build_reasons_section.
    """

    raw_name = f.get("raw_name", "")
    description = describe_feature(raw_name)
    magnitude_label = describe_magnitude(raw_name, f["value"])
    direction = "increases" if f["shap"] > 0 else "decreases"
    magnitude_clause = f" ({magnitude_label})" if magnitude_label else ""

    return (
        f"{f['name']}: {description}. Measured value "
        f"{f['value']:.3f}{magnitude_clause} — this {direction} the "
        f"predicted error."
    )


_DIRECTION_STEM_PATTERN = re.compile(r"\b(increas\w*|decreas\w*)\b", re.IGNORECASE)

_UNGROUNDED_HEDGE_TERMS = [
    "compensation mechanism", "error model", "correction system",
    "compensat", "over-compensat", "overcompensat",
    "solid object", "reflective", "unobstructed", "clear opening",
    "should have", "should typically", "should normally",
    "algorithm", "slam", "hardware", "software", "firmware",
]


def _split_sentences(text):
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def clean_hedge_sentence(hedge_text):
    """
    The LLM's hedge text is only supposed to propose WHY a feature had
    its effect — never restate whether it increased/decreased the error
    (build_fact_sentence already states that, correctly, in Python) and
    never invent a system component or physical property. If the model
    does either anyway, drop just that sentence rather than the whole
    hedge, and drop the whole hedge (falling back to no elaboration,
    which is always safe) only if nothing is left.
    """

    kept = []
    for sentence in _split_sentences(hedge_text):
        lower = sentence.lower()
        if _DIRECTION_STEM_PATTERN.search(sentence):
            continue
        if any(term in lower for term in _UNGROUNDED_HEDGE_TERMS):
            continue
        kept.append(sentence)

    return " ".join(kept).strip()


def parse_numbered_reasons(llm_text):
    """
    Parses the LLM's numbered "Reasons for Localization Error" list (see
    build_direct_prompt for the requested format) into {number: text},
    joining any wrapped continuation lines for that number. Returns an
    empty dict if the model didn't follow the numbered format at all, so
    the caller can degrade to fact-only bullets instead of crashing.
    """

    match = re.search(
        r"Reasons for Localization Error(.*?)(?:\nRecommendations\b|\Z)",
        llm_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return {}

    body = match.group(1)
    items = {}
    starts = list(re.finditer(r"(?m)^\s*(\d+)\.\s*", body))

    for i, m in enumerate(starts):
        num = int(m.group(1))
        start = m.end()
        end = starts[i + 1].start() if i + 1 < len(starts) else len(body)
        items[num] = re.sub(r"\s+", " ", body[start:end]).strip()

    return items


def build_reasons_section(top_features, llm_text):
    """
    Assembles "Reasons for Localization Error" as one Python-written fact
    sentence per feature (always correct, by construction) followed by
    the LLM's hedge sentence for that feature (cleaned of any restated
    direction or ungrounded claim). If the LLM's hedge for a feature is
    missing or gets fully cleaned away, the bullet is just the fact
    sentence — always safe, never wrong.
    """

    parsed = parse_numbered_reasons(llm_text)
    lines = ["Reasons for Localization Error", ""]

    for i, f in enumerate(top_features, start=1):
        fact = build_fact_sentence(f)
        hedge = clean_hedge_sentence(parsed.get(i, ""))
        lines.append(f"- {fact} {hedge}".rstrip() if hedge else f"- {fact}")

    return "\n".join(lines)


def extract_recommendations_section(llm_text):
    match = re.search(r"(Recommendations.*)", llm_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Recommendations\n\n(No recommendations were generated for this frame.)"


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
- For each feature above, in the SAME numbered order, write EXACTLY ONE
  hedged sentence proposing a plausible reason WHY it might have had
  this effect (see the GOOD/BAD examples above for what "hedged" means).
- Do NOT restate the feature's value, magnitude, or whether it increased
  or decreased the predicted error — that has already been stated
  elsewhere. State ONLY the hypothesis for WHY, nothing else.
- Format exactly like this (one numbered line per feature, matching the
  numbering above):
  1. <hedged reason for feature 1>
  2. <hedged reason for feature 2>
  ...

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
    Assembles the final report from:
      - a deterministic Python-written Summary + Risk Level
        (see build_summary_section)
      - a deterministic Python-written fact per feature in Reasons, with
        an LLM-written, cleaned hedge sentence appended to each
        (see build_reasons_section — this is what prevents the LLM from
        ever stating a feature's direction incorrectly)
      - an LLM-written Recommendations section, filtered for ungrounded
        claims and physical-direction assertions
    """

    if risk_level is None:
        risk_level = compute_risk_level(prediction)

    summary_section = build_summary_section(prediction, top_features, risk_level)

    prompt = build_direct_prompt(prediction, top_features)
    llm_text = _chat(DIRECT_SYSTEM_PROMPT, prompt).strip()

    reasons_section = build_reasons_section(top_features, llm_text)

    recommendations_section = extract_recommendations_section(llm_text)
    recommendations_section = strip_direction_claims(recommendations_section)
    recommendations_section = sanitize_recommendations(recommendations_section)

    return (
        "====================================================\n\n"
        f"{summary_section}\n\n"
        "====================================================\n\n"
        f"{reasons_section}\n\n"
        "====================================================\n\n"
        f"{recommendations_section}\n\n"
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
