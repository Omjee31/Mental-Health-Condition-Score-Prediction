import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Mental Wellbeing Predictor",
    page_icon="🧠",
    layout="wide"
)

# ------------------------
# Load Model
# ------------------------

@st.cache_resource
def load_model():
    return joblib.load("Mental_Health_Score_Prediction.pkl")

model = load_model()

# ------------------------
# Sidebar
# ------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction",
        "Feature Importance",
        "About"
    ]
)

# ------------------------
# HOME
# ------------------------

if page == "Home":

    st.title("🧠 Mental Wellbeing Prediction")

    st.markdown("""
This application predicts an individual's **Wellbeing Band**
using demographic information, digital behaviour,
and psychological assessment scores.

### Model

- Random Forest Classifier
- Scikit-Learn Pipeline
- Cross Validation
- Feature Engineering
- Hyperparameter Tuning

### Dataset:
Kaggle:
https://www.kaggle.com/datasets/uditjain13/social-media-screen-time-and-mental-health-2026

Use the **Prediction** page to test the model.
""")

# ------------------------
# PREDICTION PAGE
# ------------------------

elif page == "Prediction":

    st.title("Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Age",18,80,25)

        gender = st.selectbox(
            "Gender",
            ["Male","Female","Other"]
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Student",
                "Full-Time-Employed",
                "Part-Time-Employed"
                "Self-employed",
                "Unemployed",
                "Retired"
            ]
        )

        region = st.selectbox(
            "Region",
            [
                "Latin America",
                "Oceania",
                "Africa",
                "Europe",
                "Asia",
                "North America"
            ]
        )

        daily_screen_hours = st.slider(
            "Daily Screen Hours",
            0.0,18.0,5.0
        )

        daily_notifications = st.number_input(
            "Daily Notifications",
            0,
            500,
            80
        )

        minutes_to_first_check_after_waking = st.number_input(
            "Minutes Before First Phone Check",
            0,
            180,
            15
        )

        platforms_used_count = st.slider(
            "Platforms Used",
            1,
            10,
            4
        )

        most_used_platform = st.selectbox(
            "Most Used Platform",
            [
                "Instagram",
                "Facebook",
                "WhatsApp",
                "YouTube",
                "TikTok",
                "X",
                "Snapchat",
                "LinkedIn",
                "Reddit"
            ]
        )

    with col2:

        avg_sleep_hours = st.slider(
            "Average Sleep Hours",
            2.0,
            12.0,
            7.0
        )

        anxiety_score_0to27 = st.slider(
            "Anxiety Score",
            0,
            27,
            10
        )

        low_mood_score_0to27 = st.slider(
            "Low Mood Score",
            0,
            27,
            10
        )

        life_satisfaction_1to10 = st.slider(
            "Life Satisfaction",
            1,
            10,
            5
        )

        loneliness_1to10 = st.slider(
            "Loneliness",
            1,
            10,
            5
        )

        self_esteem_1to10 = st.slider(
            "Self Esteem",
            1,
            10,
            5
        )

        fomo_1to10 = st.slider(
            "FOMO",
            1,
            10,
            5
        )

        social_comparison_1to10 = st.slider(
            "Social Comparison",
            1,
            10,
            5
        )

        uses_screen_time_limits = st.selectbox(
            "Uses Screen Time Limits",
            ["Yes","No"]
        )

        night_time_use = st.selectbox(
            "Night Time Phone Usage",
            ["Never","Often","Sometimes","Every night"]
        )

        attempted_digital_detox = st.selectbox(
            "Attempted Digital Detox",
            ["No","Yes, failed","Yes, succeeded"]
        )

        seeks_mental_health_support = st.selectbox(
            "Seeks Mental Health Support",
            ["No","Yes","Considering it"]
        )

    if st.button("Predict Wellbeing"):

        sample = pd.DataFrame({

            "daily_screen_hours":[daily_screen_hours],
            "daily_notifications":[daily_notifications],
            "minutes_to_first_check_after_waking":[minutes_to_first_check_after_waking],
            "platforms_used_count":[platforms_used_count],
            "low_mood_score_0to27":[low_mood_score_0to27],
            "age":[age],
            "avg_sleep_hours":[avg_sleep_hours],
            "anxiety_score_0to27":[anxiety_score_0to27],
            "life_satisfaction_1to10":[life_satisfaction_1to10],
            "loneliness_1to10":[loneliness_1to10],
            "self_esteem_1to10":[self_esteem_1to10],
            "fomo_1to10":[fomo_1to10],
            "social_comparison_1to10":[social_comparison_1to10],
            "gender":[gender],
            "occupation":[occupation],
            "region":[region],
            "most_used_platform":[most_used_platform],
            "uses_screen_time_limits":[uses_screen_time_limits],
            "night_time_use":[night_time_use],
            "attempted_digital_detox":[attempted_digital_detox],
            "seeks_mental_health_support":[seeks_mental_health_support]

        })

        prediction = model.predict(sample)[0]

        probability = model.predict_proba(sample).max()*100

        if prediction == "Good":
            st.success(f"Predicted Wellbeing {prediction} :- It Means You Are In Good condition ")

        elif prediction == "Moderate":
            st.warning(f"Predicted Wellbeing {prediction} :- It Means Your Condition Is Moderate")

        else:
            st.error(f"Predicted Wellbeing {prediction} :- It Means You Are At Risk (Go To Doctor For Checkup )")

        st.write(f"Confidence: {probability:.2f}%")

# ------------------------
# FEATURE IMPORTANCE
# ------------------------

elif page == "Feature Importance":

    st.title("Feature Importance")

    if os.path.exists("important_features.csv"):

        imp = pd.read_csv("important_features.csv")

        st.dataframe(imp)

        fig, ax = plt.subplots(figsize=(8,7))

        imp = imp.sort_values(
            "Importance",
            ascending=True
        )

        ax.barh(
            imp["Feature"],
            imp["Importance"]
        )

        st.pyplot(fig)

    else:

        st.info(
            "Save your feature importance dataframe as feature_importance.csv"
        )

# ------------------------
# ABOUT
# ------------------------

elif page == "About":

    st.title("About Project")

    st.markdown("""

### Project

Mental Wellbeing Prediction using Machine Learning

### Algorithms Compared

- Logistic Regression     . Accuracy 92%
- Support Vector Machine  . Accuracy 92%
- Random Forest           . Accuracy 99%

### Final Model

Random Forest Classifier

### Why Random Forest 
- Anxiety score is the most important feature, and the Random Forest model effectively captures its strong influence on health prediction.
- Since anxiety plays a crucial role in overall health, giving higher importance to this feature improves the model's decision-making. 
- The model achieved 99.71% accuracy, making it highly reliable.
- Out of 2,100 predictions, only 4 were incorrect, demonstrating excellent predictive performance.
- The model also achieved very high precision, recall, and F1-score (all around 99.7%–100%), indicating that it correctly identifies almost all cases with very few false predictions.

### Conclusion:
- Random Forest was selected because it accurately captures the strong impact of anxiety on health outcomes while delivering 99.71% accuracy with only 4 misclassifications out of 2,100 predictions, making it the best-performing and most reliable model.
### Dataset

Kaggle:
https://www.kaggle.com/datasets/uditjain13/social-media-screen-time-and-mental-health-2026

### Developed By

Om Jee Singh 

GitHub:
https://github.com/Omjee31

LinkedIn:
https://www.linkedin.com/in/omjee-singh-454b9a328/

""")