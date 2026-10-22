
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}
h1 {
    color: #183153;
}
div[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e0e6ef;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎓 Student Performance Predictor")
st.write(
    "An interactive machine-learning application "
    "to estimate student final grades."
)

st.divider()

# Load and train model


@st.cache_data
def load_data():
    return pd.read_csv("data/student-mat.csv", sep=";")


data = load_data()

features = ["studytime", "failures", "absences", "G1", "G2"]
X = data[features]
y = data["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Dashboard metrics
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(data))
col2.metric(
    "Mean Absolute Error",
    round(mean_absolute_error(y_test, predictions), 2)
)
col3.metric(
    "R² Score",
    round(r2_score(y_test, predictions), 2)
)

st.caption(
    "Metrics are calculated on a held-out test set. "
    "Predictions are estimates, not guaranteed grades."
)

st.divider()

# Two-column layout
left, right = st.columns([1, 1.3], gap="large")

with left:
    st.subheader("📝 Student Details")
    st.write("Enter the student's information below.")

    with st.form("student_form"):
        studytime = st.selectbox(
            "Weekly study time",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2–5 hours",
                3: "5–10 hours",
                4: "More than 10 hours"
            }[x]
        )

        failures = st.number_input(
            "Past class failures",
            min_value=0,
            max_value=3,
            value=0
        )

        absences = st.number_input(
            "School absences",
            min_value=0,
            max_value=93,
            value=2
        )

        G1 = st.slider("First-period grade", 0, 20, 12)
        G2 = st.slider("Second-period grade", 0, 20, 14)

        submitted = st.form_submit_button(
            "✨ Predict Final Grade",
            use_container_width=True
        )

    if submitted:
        student = pd.DataFrame(
            [[studytime, failures, absences, G1, G2]],
            columns=features
        )

        result = model.predict(student)[0]
        result = max(0, min(20, result))

        st.session_state["prediction"] = round(result, 2)

with right:
    st.subheader("📈 Actual vs Predicted Grades")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, predictions, alpha=0.7)
    ax.plot([0, 20], [0, 20], linestyle="--")
    ax.set_xlabel("Actual Grade")
    ax.set_ylabel("Predicted Grade")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# Prediction result
if "prediction" in st.session_state:
    st.divider()
    st.subheader("🎯 Prediction Result")
    st.success(
        f"Estimated Final Grade: "
        f"{st.session_state['prediction']:.2f} / 20"
    )
    st.progress(
        int(st.session_state["prediction"] * 5)
    )

st.divider()
st.caption(
    "Built with Python, Pandas, Scikit-learn, Matplotlib "
    "and Streamlit | Student Performance Predictor"
)
