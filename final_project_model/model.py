import pandas as pd 
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import unidecode
import string
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
import re
import pickle

df = pd.read_csv('train.csv')

class_columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
df['toxic_classification'] = df[class_columns].max(axis=1)

# Data cleaning, tokenization, and stop words 
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove accents
    text = unidecode.unidecode(text)
    # Normalize numbers: Replace digits with a special token, e.g., "NUM"
    text = re.sub(r'\d+', 'NUM', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Special characters 
    text = re.sub(r'[^\w\sáéíóúüñ¿¡]', '', text) 
    # Tokenize the text into words
    words = word_tokenize(text)
    # Define stop words for English
    stop_words = set(stopwords.words('english'))
    # Remove stop words from the tokenized words
    words = [word for word in words if word not in stop_words]
    # Join the words back into a single string with spaces
    return ' '.join(words)

df['cleaned_text'] = df['comment_text'].apply(preprocess_text)

# TF-IDF Vectorization function
def tfidf_vectorize(data, text_column):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data[text_column])
    return X, vectorizer

# Logistic Regression Classification function
def logistic_regression_classification(X, y):
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # random_state=42)
    
    # Train the classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    return classifier

# Main function to tie everything together
def main_tfidf_log_reg(data, text_column):
    # TF-IDF Vectorization
    X, vectorizer = tfidf_vectorize(data, text_column)
    y = data['toxic_classification']

    # Classification
    classifier = logistic_regression_classification(X, y)

    # Save the model
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(classifier, model_file)
    
    # Save the vectorizer
    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Model and vectorizer saved successfully.")

# Execute the main function
main_tfidf_log_reg(df, 'cleaned_text')
