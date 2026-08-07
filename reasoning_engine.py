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
        # ==================================================
        # ORB Features
        # ==================================================

        elif feature == "orb_features":

            observation = (
                f"Approximately {int(value)} ORB visual features were detected."
            )

            if value < 1500:

                domain_meaning = (
                    "Very few visual landmarks were detected. "
                    "This usually makes feature matching difficult and reduces localization reliability."
                )

                recommendations.add(
                    "Fly through areas containing richer visual textures."
                )

            elif value < 3000:

                domain_meaning = (
                    "A moderate number of visual landmarks were detected. "
                    "Localization is possible but may become unstable in challenging scenes."
                )

            else:

                domain_meaning = (
                    "A large number of visual landmarks were detected, which generally supports robust visual localization."
                )


        # ==================================================
        # Edge Density
        # ==================================================

        elif feature == "edge_density":

            observation = (
                f"The measured edge density is {value:.3f}."
            )

            if value < 0.05:

                domain_meaning = (
                    "Very few edges and corners are visible in the scene. "
                    "Texture-poor environments provide fewer stable landmarks."
                )

                recommendations.add(
                    "Avoid large texture-poor regions whenever possible."
                )

            elif value < 0.10:

                domain_meaning = (
                    "The scene contains a moderate number of edges. "
                    "Localization is generally possible but may become less stable."
                )

            else:

                domain_meaning = (
                    "The environment contains many edges and structural details that can support visual localization."
                )


        # ==================================================
        # UAV Motion (dx)
        # ==================================================

        elif feature == "dx":

            observation = (
                f"The UAV moved {abs(value):.3f} meters along the X-axis between two consecutive frames."
            )

            if abs(value) > 0.25:

                domain_meaning = (
                    "Large sideways motion increases viewpoint changes, making feature matching more challenging."
                )

                recommendations.add(
                    "Reduce UAV speed during sideways motion."
                )

            else:

                domain_meaning = (
                    "Sideways motion is relatively small and should not significantly affect localization."
                )


        # ==================================================
        # UAV Motion (dy)
        # ==================================================

        elif feature == "dy":

            observation = (
                f"The UAV moved {abs(value):.3f} meters along the Y-axis between consecutive frames."
            )

            if abs(value) > 0.25:

                domain_meaning = (
                    "Large vertical motion can increase viewpoint changes and reduce localization stability."
                )

                recommendations.add(
                    "Reduce rapid vertical movement when possible."
                )

            else:

                domain_meaning = (
                    "Vertical motion is relatively small."
                )


        # ==================================================
        # UAV Motion (dz)
        # ==================================================

        elif feature == "dz":

            observation = (
                f"The UAV moved {abs(value):.3f} meters along the Z-axis between consecutive frames."
            )

            if abs(value) > 0.30:

                domain_meaning = (
                    "Large forward or backward movement increases the difficulty of matching visual features between frames."
                )

                recommendations.add(
                    "Reduce forward speed in visually challenging environments."
                )

            else:

                domain_meaning = (
                    "Forward motion is within a normal range for visual localization."
                )
                        # ==================================================
        # Semantic Classes
        # ==================================================

        elif feature.startswith("class_"):

            name = CLASS_NAMES.get(
                feature,
                feature.replace("_percent", "").replace("_", " ")
            )

            observation = (
                f"{name} occupies approximately {value:.1f}% of the visible scene."
            )

            # Large semantic regions generally provide fewer unique landmarks

            if value < 2:

                domain_meaning = (
                    f"The proportion of {name.lower()} in the scene is small, "
                    "so its influence on localization is limited."
                )

            elif value < 10:

                domain_meaning = (
                    f"{name} forms a noticeable part of the environment. "
                    "Depending on its texture and visual structure, it may slightly influence localization."
                )

            else:

                domain_meaning = (
                    f"A significant portion of the scene consists of {name.lower()}. "
                    f"Large {name.lower()} regions often contain repeated textures or fewer distinctive visual landmarks, "
                    "making feature matching more difficult."
                )

            # ------------------------------------------------
            # Object-specific recommendations
            # ------------------------------------------------

            if "Carpet" in name:

                recommendations.add(
                    "Avoid flying very close to large carpeted areas whenever possible."
                )

            elif "Wall" in name:

                recommendations.add(
                    "Maintain sufficient distance from large walls to capture richer scene features."
                )

            elif "Floor" in name:

                recommendations.add(
                    "Include surrounding objects instead of only the floor in the camera view."
                )

            elif "Window" in name:

                recommendations.add(
                    "Avoid relying heavily on window regions due to reflections and limited texture."
                )

            elif "Ceiling" in name:

                recommendations.add(
                    "Maintain a forward-looking camera angle instead of pointing toward the ceiling."
                )

            elif "Furniture" in name:

                recommendations.add(
                    "Furniture usually provides useful landmarks. Ensure it remains visible during flight."
                )

            elif "Shelf" in name:

                recommendations.add(
                    "Shelves generally provide good structural landmarks for localization."
                )

            elif "Plant" in name:

                recommendations.add(
                    "Vegetation may move slightly and create unstable visual features."
                )

            elif "Picture" in name:

                recommendations.add(
                    "Pictures often contain rich textures that can improve feature matching."
                )
                        # ==================================================
        # Mean Depth
        # ==================================================

        elif feature == "mean_depth":

            observation = (
                f"The average distance of visible objects is {value:.2f} meters."
            )

            if value < 2:

                domain_meaning = (
                    "Most objects are very close to the UAV. Nearby objects can leave the camera view quickly, making localization more challenging."
                )

            elif value < 5:

                domain_meaning = (
                    "Objects are at a moderate distance, which generally provides good visual information for localization."
                )

            else:

                domain_meaning = (
                    "Most objects are far from the UAV. Distant objects appear smaller and may provide fewer reliable visual landmarks."
                )


        # ==================================================
        # Depth Entropy
        # ==================================================

        elif feature == "depth_entropy":

            observation = (
                f"The depth entropy is {value:.2f}."
            )

            if value < 3:

                domain_meaning = (
                    "The scene has little depth variation, so the environment contains limited three-dimensional structure."
                )

            else:

                domain_meaning = (
                    "The scene contains good depth variation, providing useful geometric information for localization."
                )


        # ==================================================
        # Valid Depth Ratio
        # ==================================================

        elif feature == "valid_depth_ratio":

            observation = (
                f"{value*100:.1f}% of depth pixels are valid."
            )

            if value < 0.90:

                domain_meaning = (
                    "Many depth measurements are missing or unreliable, reducing the quality of geometric information."
                )

                recommendations.add(
                    "Improve depth sensing or avoid regions with unreliable depth measurements."
                )

            else:

                domain_meaning = (
                    "Most depth measurements are reliable."
                )


        # ==================================================
        # Default
        # ==================================================

        else:

            observation = (
                f"The measured value of {feature} is {value:.3f}."
            )

            domain_meaning = (
                "This feature contributed to the prediction learned by the Random Forest model."
            )


        # ==================================================
        # Save Verified Fact
        # ==================================================

        facts.append({

            "feature": feature,

            "value": value,

            "observation": observation,

            "domain_meaning": domain_meaning,

            "model_influence": (
                "The Random Forest model increased the predicted localization error because of this feature."
                if shap_value > 0
                else
                "The Random Forest model reduced the predicted localization error because of this feature."
            ),

            "direction": (
                "positive"
                if shap_value > 0
                else
                "negative"
            )

        })
            # ----------------------------------------------------------
    # Default Recommendation
    # ----------------------------------------------------------

    if len(recommendations) == 0:

        recommendations.add(
            "Current flight conditions appear suitable for reliable localization."
        )

    # ----------------------------------------------------------
    # Sort Recommendations
    # ----------------------------------------------------------

    recommendations = sorted(list(recommendations))

    # ----------------------------------------------------------
    # Return verified facts to the LLM
    # ----------------------------------------------------------

    return facts, recommendations
