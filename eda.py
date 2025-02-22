import streamlit as st
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from collections import Counter

class EDA:
    def __init__(self, df):
        self.df = df

    def show_dataset_summary(self):
        """Display Dataset Overview."""
        st.header("📌 Dataset Summary")
        st.write(self.df.head())

    def plot_spam_distribution(self):
        """Visualize Spam vs. Ham Count."""
        st.subheader("Spam vs. Ham Count")
        fig, ax = plt.subplots()
        sns.countplot(x=self.df['label'], palette='viridis', ax=ax)
        ax.set_xticklabels(["Ham (0)", "Spam (1)"])
        st.pyplot(fig)

    def plot_message_length(self):
        """Plot Message Length Distribution."""
        self.df['message_length'] = self.df['message'].apply(len)
        st.subheader("Message Length Distribution")
        fig, ax = plt.subplots()
        sns.histplot(self.df['message_length'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    def generate_wordclouds(self):
        """Generate WordClouds for Spam & Ham Messages."""
        st.header("📊 Exploratory Data Analysis")

        spam_messages = self.df[self.df['label'] == 1]['message']
        ham_messages = self.df[self.df['label'] == 0]['message']
        
        spam_words = " ".join(spam_messages) if not spam_messages.empty else "NoSpamMessages"
        ham_words = " ".join(ham_messages) if not ham_messages.empty else "NoHamMessages"

        # WordCloud for Spam
        spam_wc = WordCloud(width=800, height=400, background_color="black").generate(spam_words)
        st.subheader("Most Common Words in Spam Messages")
        fig, ax = plt.subplots()
        ax.imshow(spam_wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # WordCloud for Ham
        ham_wc = WordCloud(width=800, height=400, background_color="white").generate(ham_words)
        st.subheader("Most Common Words in Ham Messages")
        fig, ax = plt.subplots()
        ax.imshow(ham_wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    def show_top_spam_words(self):
        """Show most common words in spam messages."""
        spam_messages = self.df[self.df['label'] == 1]['message']
        words = " ".join(spam_messages).split()
        common_words = Counter(words).most_common(10)

        st.subheader("📌 Top 10 Spam Trigger Words")
        st.table(pd.DataFrame(common_words, columns=["Word", "Count"]))
