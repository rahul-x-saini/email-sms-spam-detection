import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk.download("punkt_tab")
nltk.download("stopwords")

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))
    y = [ps.stem(i) for i in text if i not in stop_words]

    return " ".join(y)

# -----------------------------
# Load saved vectorizer and model
# -----------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="📩 Spam Classifier",
    page_icon="📨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title
st.markdown("<h1 style='text-align:center; color:#4B0082;'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("---")

# Main input area
st.subheader("Enter your message below:")
input_sms = st.text_area("Type your email or SMS message here:", height=150)

# Sidebar with optional examples
st.sidebar.header("Try Example Messages")
example_sms = st.sidebar.selectbox(
    "Select an example message:",
    ["", "Win cash now!", "Are we meeting today?", "Congratulations! You won a free ticket"]
)

# Only use example if main input is empty
if input_sms.strip() == "" and example_sms != "":
    input_sms = example_sms

# Predict button in main area
if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms]).toarray()

        # Predict
        result = model.predict(vector_input)[0]
        try:
            prob = model.predict_proba(vector_input)[0][1]  # probability of spam
        except:
            prob = None

        # Display results in columns
        col1, col2 = st.columns([2,1])  # wider left column for prediction

        # Prediction Badge
        if result == 1:
            col1.markdown("<h2 style='color:red; text-align:center;'>🚨 SPAM!</h2>", unsafe_allow_html=True)
        else:
            col1.markdown("<h2 style='color:green; text-align:center;'>✅ NOT SPAM</h2>", unsafe_allow_html=True)

        # Confidence bar
        if prob is not None:
            col2.markdown("<h4 style='text-align:center;'>Spam Probability</h4>", unsafe_allow_html=True)
            st.progress(int(prob * 100))
            col2.markdown(f"<p style='text-align:center;'>{prob * 100:.1f}%</p>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color: gray;'>Developed by Rahul Saini | <a href='https://github.com/rahul-x-saini'>GitHub</a></p>",
    unsafe_allow_html=True
)

