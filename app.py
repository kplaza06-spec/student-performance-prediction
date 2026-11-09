
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

model = load_model()
data = load_data()

st.title("🎓 Student Performance Prediction System")
st.caption("Predictive system for identifying students who may need academic support.")

st.divider()

# Dashboard metrics
st.subheader("📊 Dashboard Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Student Records", len(data))
c2.metric("Average Final Grade", round(data["G3"].mean(), 2))
c3.metric(
    "Historical At-Risk Students",
    int((data["G3"] < 10).sum())
)

st.divider()

# Dataset preview
st.subheader("📋 Student Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

# Download sample input
st.subheader("📥 Sample Input")

template = data.drop(columns=["G3"]).head(5)

st.download_button(
    "Download Sample Input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv"
)

st.divider()

# Upload CSV
st.subheader("🔍 Predict Student Performance")

uploaded = st.file_uploader(
    "Upload a student CSV file",
    type=["csv"]
)

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)

        if new_data.empty:
            st.warning("The uploaded CSV contains no records.")
        else:
            required_columns = list(model.feature_names_in_)

            missing = [
                col for col in required_columns
                if col not in new_data.columns
            ]

            if missing:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing)
                )
            else:
                predictions = model.predict(
                    new_data[required_columns]
                )

                results = new_data.copy()

                results["Prediction"] = [
                    "AT RISK" if p == 1 else "NOT AT RISK"
                    for p in predictions
                ]

                st.success("Prediction completed!")

                st.subheader("Prediction Results")
                st.dataframe(results, use_container_width=True)

                at_risk = int((predictions == 1).sum())
                not_at_risk = int((predictions == 0).sum())

                col1, col2 = st.columns(2)
                col1.metric("Predicted AT RISK", at_risk)
                col2.metric("Predicted NOT AT RISK", not_at_risk)

                st.download_button(
                    "Download Prediction Results",
                    results.to_csv(index=False),
                    "student_predictions.csv",
                    "text/csv"
                )

    except Exception as e:
        st.error(f"Unable to process CSV: {e}")

st.divider()

st.subheader("⚠️ Responsible Use")

st.write(
    "This is a classroom prototype using historical data. "
    "Predictions are estimates and must not be used as the "
    "sole basis for academic decisions. Faculty members "
    "should review each student's situation and provide "
    "appropriate support."
)

st.caption("Student Performance Prediction System | Data Mining Laboratory")
