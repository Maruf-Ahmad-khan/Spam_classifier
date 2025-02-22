import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ Download NLTK Resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class SpamClassifier:
    def __init__(self, file_path):
        """Initialize the Spam Classifier with data loading and preprocessing."""
        self.file_path = file_path
        self.df = self.load_data()
        self.vectorizer = TfidfVectorizer(max_features=3000, max_df=0.8, min_df=1)
        
        # Initialize models
        self.models = {
            "Naïve Bayes": MultinomialNB(alpha=0.5),
            "SVM": SVC(kernel="linear"),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.trained_models = {}

    def load_data(self):
        """Load and preprocess the dataset."""
        df = pd.read_csv(self.file_path)
        df.dropna(inplace=True)
        df['message'] = df['message'].astype(str).replace('', 'NoText')
        df['label'] = df['label'].map({'ham': 0, 'spam': 1}).astype(int)
        return df
    
    def preprocess_text(self, text):
        """Text Preprocessing."""
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()

        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(text)
        words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]
        return " ".join(words) if words else "NoText"

    def train_models(self):
        """Train multiple ML models."""
        self.df['cleaned_message'] = self.df['message'].apply(self.preprocess_text)
        X = self.vectorizer.fit_transform(self.df['cleaned_message'])
        y = self.df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        results = {}
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
            results[model_name] = {
                "accuracy": accuracy,
                "report": pd.DataFrame({
                    "Class": ["Ham (0)", "Spam (1)"],
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                    "Support": support
                })
            }
            self.trained_models[model_name] = model

        return results

    def predict(self, message, model_name="Naïve Bayes"):
        """Predict using the selected model."""
        if model_name not in self.trained_models:
            return "Model not trained!"
        
        processed_message = self.preprocess_text(message)
        transformed_message = self.vectorizer.transform([processed_message])
        return self.trained_models[model_name].predict(transformed_message)[0]
