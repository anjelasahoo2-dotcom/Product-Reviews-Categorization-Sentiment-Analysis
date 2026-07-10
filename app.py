import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    page_icon="⭐",
    layout="wide"
)

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("dataset/Participants_Data_DCW/train.csv")
    return data

data = load_data()

# ---------------------------
# Text Cleaning
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["Review"] = data["Review"].fillna("").apply(clean_text)

# ---------------------------
# Train Model
# ---------------------------
X = data["Review"]
y = data["Polarity"]

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select",
    ["Home", "Dataset", "Predict Sentiment"]
)

# ---------------------------
# Home Page
# ---------------------------
if page == "Home":

    st.title("🛍️ Product Review Sentiment Analysis")

    st.write("""
This application predicts whether a product review is **Positive** or **Negative**.

### Features
- TF-IDF Vectorization
- Logistic Regression Model
- Live Prediction
- Dataset Preview

Dataset Size:
""")

    col1, col2 = st.columns(2)

    col1.metric("Reviews", len(data))
    col2.metric("Features", data.shape[1])

# ---------------------------
# Dataset Page
# ---------------------------
elif page == "Dataset":

    st.title("📊 Dataset")

    st.dataframe(data.head(20), use_container_width=True)

    st.subheader("Sentiment Distribution")

    sentiment = data["Polarity"].value_counts()

    st.bar_chart(sentiment)

# ---------------------------
# Prediction Page
# ---------------------------
elif page == "Predict Sentiment":

    st.title("🔍 Predict Product Review")

    review = st.text_area(
        "Enter Product Review",
        height=200,
        placeholder="Example: This product is amazing and works perfectly."
    )

    if st.button("Predict"):

        if review.strip() == "":
            st.warning("Please enter a review.")
        else:

            cleaned = clean_text(review)

            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)[0]

            probability = model.predict_proba(vector)[0]

            if prediction == 1:
                st.success("😊 Positive Review")
                st.write(f"Confidence: **{probability[1]*100:.2f}%**")
            else:
                st.error("☹️ Negative Review")
                st.write(f"Confidence: **{probability[0]*100:.2f}%**")

            st.subheader("Prediction Probability")

            df = pd.DataFrame({
                "Sentiment": ["Negative", "Positive"],
                "Probability": probability
            })

            st.bar_chart(df.set_index("Sentiment"))

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Product Review Sentiment Analysis using Streamlit")