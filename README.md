## 📩 Spam Email Classifier - Project Overview 

### 📌 Project Purpose

This **Spam Email Classifier**
is a **Machine Learning (ML) application**
that detects spam emails using multiple ML models. 
The project allows users to:  
✅ **Upload a dataset** (CSV file) of spam and ham emails.  
✅ **Explore the dataset** with visualizations & insights.  
✅ **Train multiple ML models** (Naïve Bayes, SVM, Random Forest).  
✅ **Predict whether an email is spam or ham** using trained models.  

## **🛠️ How This Project Works?**
### **1️⃣ Uploading the Dataset**
- Users can **upload their CSV file** via the Streamlit UI.  
- The dataset should contain:
  - `message` column (email text).
  - `label` column (`ham` for normal emails, `spam` for spam emails).
- The system reads the file and processes the text data.  

### **2️⃣ Text Preprocessing**
The dataset undergoes **Natural Language Processing (NLP)** steps:  
✔ Convert text to **lowercase**.  
✔ **Remove punctuation & stopwords**.  
✔ **Tokenization** (breaking text into words).  
✔ **Lemmatization & Stemming** (reducing words to their root forms).  
✔ Convert text into **numerical vectors** using **TF-IDF (Term Frequency - Inverse Document Frequency)**.

### **3️⃣ Model Training & Evaluation**
The system **trains three different ML models**:  
- **Multinomial Naïve Bayes (NB)**: Good for text classification.  
- **Support Vector Machine (SVM)**: Performs well for spam detection.  
- **Random Forest (RF)**: A tree-based ensemble learning method.  

Each model is trained, tested, and evaluated using:
- **Accuracy**
- **Precision, Recall, and F1-Score** (displayed in a table).

### **4️⃣ Data Exploration & Visualizations**
Users can explore their dataset using:  
📊 **Spam vs. Ham Distribution** (bar chart).  
📏 **Message Length Distribution** (histogram).  
🌐 **Word Clouds** (most frequent spam & ham words).  
📌 **Top Spam Trigger Words** (table of common spam words).  

### **5️⃣ Spam Prediction**
- Users can **input any email message** into the system.  
- Select **which model** to use for prediction.  
- The system **predicts if it's spam or not** with 🚨 **Spam Detected!** or ✅ **Ham (Not Spam)**.

---

## **📊 How is the Model Performing?**
📌 **Performance Summary:**  
- **Accuracy (Approximate Values)**
  - **Naïve Bayes:** **97-98%**
  - **SVM:** **98-99%**
  - **Random Forest:** **96-97%**

📌 **Classification Report Sample (Example Output):**  

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|----------|
| Ham (0)  | 0.98       | 1.00   | 0.99     | 965      |
| Spam (1) | 0.99       | 0.85   | 0.92     | 149      |

📌 **Insights:**  
✔ **Naïve Bayes** performs well but is slightly sensitive to word frequency.  
✔ **SVM** has the **highest precision**, meaning it is less likely to misclassify spam.  
✔ **Random Forest** is robust but may slightly overfit.  

---

## **🎯 Why is This Project Useful?**
✅ **Helps in spam detection for businesses & individuals.**  
✅ **Prevents phishing, fraud, and email overload.**  
✅ **Gives users the ability to explore & visualize spam trends.**  
✅ **Allows selection of different ML models for comparison.**  
✅ **Supports custom datasets through file uploads.**  


