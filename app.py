import streamlit as st
import pandas as pd
import os
import nltk
import warnings
from nltk.corpus import stopwords
from spam import SpamClassifier
from eda import EDA

# ✅ Ignore warnings
warnings.filterwarnings("ignore")

# ✅ Manually set the NLTK data path
nltk.data.path.append("C:\\nltk_data")

# ✅ Function to check and download only if needed
def download_nltk_packages():
    required_packages = ["stopwords", "punkt", "wordnet"]
    
    for package in required_packages:
        try:
            nltk.data.find(f"corpora/{package}")
            print(f"{package} is already installed.")
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, download_dir="C:\\nltk_data")

# ✅ Run the function
download_nltk_packages()

# ✅ Use Fixed File Path for Dataset
file_path = r"C:\Users\mk744\OneDrive - Poornima University\Desktop\spam_classifiers\Email_spam_data.csv"
classifier = SpamClassifier(file_path=file_path)
results = classifier.train_models()
eda = EDA(classifier.df)

# ✅ Streamlit UI
st.title("📩 Spam Email Classifier App")

st.sidebar.header("Navigation")
option = st.sidebar.radio("Select Option", ["Dataset Overview", "EDA & Visualization", "Spam Prediction"])

if option == "Dataset Overview":
    eda.show_dataset_summary()
    eda.plot_spam_distribution()
    eda.plot_message_length()

elif option == "EDA & Visualization":
    eda.generate_wordclouds()
    eda.show_top_spam_words()
    for model_name, result in results.items():
        st.subheader(f"📌 {model_name} Performance")
        st.write(f"**Accuracy:** {result['accuracy']:.2%}")
        st.dataframe(result["report"])

elif option == "Spam Prediction":
    model_choice = st.selectbox("Choose Model", ["Naïve Bayes", "SVM", "Random Forest"])
    user_input = st.text_area("Enter an email message below:")
    if st.button("Predict"):
        prediction = classifier.predict(user_input, model_choice)
        result = "✅ Ham (Not Spam)" if prediction == 0 else "🚨 Spam Detected!"
        st.subheader("Prediction Result:")
        st.write(f"**{result}**")
