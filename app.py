
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def load_and_train_model():
    data = pd.read_csv("student_data.csv")
    data = data.drop_duplicates()

    if "G3" not in data.columns:
        raise ValueError("The dataset must contain the G3 column.")

    y = (data["G3"] < 10).astype(int)
    X = data.drop(columns=["G3"])

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    return model, data, X.columns.tolist()


try:
    model, data, required_columns = load_and_train_model()

    st.title("🎓 Student Performance Prediction System")
    st.caption(
        "Predictive system for identifying students "
        "who may need academic support."
    )

    st.divider()

    st.subheader("📊 Dashboard Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Student Records", len(data))
    c2.metric("Average Final Grade", round(data["G3"].mean(), 2))
    c3.metric(
        "Historical At-Risk Students",
        int((data["G3"] < 10).sum())
    )

    st.divider()

    st.subheader("📋 Student Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("📥 Sample Input")

    template = data[required_columns].head(5)

    st.download_button(
        "Download Sample Input CSV",
        template.to_csv(index=False),
        "sample_prediction_input.csv",
        "text/csv"
    )

    st.divider()

    st.subheader("🔍 Predict Student Performance")

    uploaded = st.file_uploader(
        "Upload a student CSV file",
        type=["csv"]
    )

    if uploaded is not None:
        new_data = pd.read_csv(uploaded)

        if new_data.empty:
            st.warning("The uploaded CSV contains no records.")
        else:
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

    st.divider()

    st.subheader("⚠️ Responsible Use")
    st.write(
        "This is a classroom prototype using historical data. "
        "Predictions are estimates and must not be used as the "
        "sole basis for academic decisions. Faculty members "
        "should review each student's situation and provide "
        "appropriate support."
    )

    st.caption(
        "Student Performance Prediction System | Data Mining Laboratory"
    )

except Exception as e:
    st.error(f"Unable to start the application: {e}")

st.caption("Student Performance Prediction System | Data Mining Laboratory")
