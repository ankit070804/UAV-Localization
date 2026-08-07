# ==========================================================
# reasoning_engine.py
#
# Converts SHAP-important features into verified facts
# for the LLM.
#
# The reasoning engine DOES NOT generate natural language
# reports. It only produces verified facts.
#
# LLM -> Professional report
# ==========================================================


# ----------------------------------------------------------
# Semantic Class Names
# ----------------------------------------------------------

CLASS_NAMES = {

    "class_28_percent": "Furniture",
    "class_229_percent": "Shelf",
    "class_180_percent": "Carpet",
    "class_157_percent": "Wall",
    "class_64_percent": "Floor",
    "class_70_percent": "Door",
    "class_87_percent": "Window",
    "class_115_percent": "Ceiling",
    "class_125_percent": "Picture",
    "class_143_percent": "Table",

    "class_8_percent": "Chair",
    "class_27_percent": "Cabinet",
    "class_59_percent": "Curtain",
    "class_74_percent": "Desk",
    "class_119_percent": "Monitor",
    "class_123_percent": "Sofa",
    "class_132_percent": "Plant",
    "class_146_percent": "Bed",
    "class_175_percent": "Door Frame",
    "class_199_percent": "Books",
    "class_205_percent": "Lamp",
    "class_208_percent": "Computer",
    "class_214_percent": "Television",
    "class_239_percent": "Miscellaneous"

}


# ==========================================================
# Main Function
# ==========================================================

def explain(top_features):

    facts = []
    recommendations = set()

    for _, row in top_features.iterrows():

        feature = row["Feature"]
        value = float(row["Value"])
        shap_value = float(row["SHAP"])


        # -----------------------------------------------
        # Model Influence
        # -----------------------------------------------

        if shap_value > 0:

            model_influence = (
                "For this prediction, the trained Random Forest "
                "considered this feature as increasing the predicted "
                "localization error."
            )

            direction = "positive"

        else:

            model_influence = (
                "For this prediction, the trained Random Forest "
                "considered this feature as reducing the predicted "
                "localization error."
            )

            direction = "negative"


        # Default values
        observation = ""
        domain_meaning = ""
                # ==================================================
        # Brightness
        # ==================================================

        if feature == "brightness":

            observation = (
                f"The measured image brightness is {value:.1f}."
            )

            if value < 80:

                domain_meaning = (
                    "The scene is dark, making visual landmarks harder to detect."
                )

                recommendations.add(
                    "Increase camera exposure or improve scene lighting."
                )

            elif value > 200:

                domain_meaning = (
                    "The scene is very bright, which may hide important visual details."
                )

                recommendations.add(
                    "Reduce camera exposure to avoid overexposed images."
                )

            else:

                domain_meaning = (
                    "The scene brightness is within a suitable range for visual localization."
                )


        # ==================================================
        # Contrast
        # ==================================================

        elif feature == "contrast":

            observation = (
                f"The measured image contrast is {value:.1f}."
            )

            if value < 60:

                domain_meaning = (
                    "Low contrast makes object boundaries difficult to distinguish, reducing reliable feature matching."
                )

                recommendations.add(
                    "Improve illumination or adjust camera exposure."
                )

            else:

                domain_meaning = (
                    "The image contains sufficient contrast for reliable feature detection."
                )


        # ==================================================
        # Blur
        # ==================================================

        elif feature == "blur":

            observation = (
                f"The blur score is {value:.1f}."
            )

            if value > 600:

                domain_meaning = (
                    "High blur reduces image sharpness and makes visual landmarks more difficult to detect."
                )

                recommendations.add(
                    "Reduce UAV speed or increase camera shutter speed to reduce motion blur."
                )

            else:

                domain_meaning = (
                    "Image sharpness is sufficient for reliable feature extraction."
                )