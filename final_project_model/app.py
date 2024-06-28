import streamlit as st
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import unidecode
import re
import os
import pickle

# Load the model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    extract_path = './final_project_model/'  # Directory where model and vectorizer files are stored
    model_path = os.path.join(extract_path, 'model.pkl')
    vectorizer_path = os.path.join(extract_path, 'vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None, "Model file or vectorizer file not found. Ensure they are in the correct directory."
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer, None

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

def main():
    nltk.download('stopwords')
    nltk.download('punkt')
    
    st.set_page_config(layout="wide")
    
    st.title('Final Project - Natural Language Processing')

    
    options = st.sidebar.radio('', ['🚩 Red Flag Detector ', '📓 About '])

    if options == '🚩 Red Flag Detector ':
        st.subheader('Red Flag Detector 🚩')
        st.write("Ready to spot red flags? Use the Red Flag Detector to foster a positive vibe and suss out any toxicity in your interactions. Enter text below to check for warning signs in conversations with friends or partners. Let's keep it healthy! 💬")
        
        # Load model and vectorizer
        model, vectorizer = load_model_and_vectorizer()
        
        # User input text area
        user_inputs = st.text_area('Enter as many comment as you need, please make sure there is one per line 😉 ', height=200).split('\n')
        
        if st.button('Classify'):
            if not user_inputs:
                st.warning('Please enter some text.')
            else:
                results = []
                for user_input in user_inputs:
                    if user_input.strip() == '':
                        results.append('Please enter some text.')
                    else:
                        processed_input = preprocess_text(user_input)
                        input_vector = vectorizer.transform([processed_input])
                        prediction = model.predict(input_vector)[0]
                        if prediction == 1:
                            results.append(f'Text: "{user_input}" - <span style="color:red; font-weight:bold;">Toxic</span>')
                        else:
                            results.append(f'Text: "{user_input}" - <span style="color:green; font-weight:bold;">Non-Toxic</span>')
                
                for result in results:
                    st.markdown(result, unsafe_allow_html=True)
    
    elif options == '📓 About ':
        st.subheader('About 📓')
        st.markdown("""
        **The Author**
                    
        Lydia Manzanares is a highly dedicated and accomplished professional, who successfully transitioned from a thriving career in product operations management to follow her passion for data analysis and machine learning. With a robust background in product development, Lydia brings a distinctive perspective to data science, seamlessly integrating her expertise in understanding user needs and delivering strategic outcomes. Her educational journey continues in the completion of the rigorous Data Science and Machine Learning bootcamp at Ironhack, where she refined her skills in developing advanced algorithms and harnessing data to drive well-informed decisions.

        Her overarching goal is to empower businesses and organizations by giving a compelling voice to data, thereby enabling them to make pivotal, data-driven decisions.

        <a href="https://www.linkedin.com/in/lymanzanares" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/LinkedIn_icon.svg/1200px-LinkedIn_icon.svg.png" alt="LinkedIn" style="width:32px;height:32px;border:0;"></a>
        <a href="https://github.com/lilimanz/final-project.git" target="_blank"><img src="https://static-00.iconduck.com/assets.00/github-icon-2048x2048-823jqxdr.png" alt="GitHub" style="width:32px;height:32px;border:0;"></a>
                    
        **The Red Flag Detector**

        The Red Flag Detector, Lydia Manzanares's final project at Ironhack, embodies her dedication to leveraging data science for societal benefit. This innovative app utilizes advanced logistic classification algorithms to analyze and flag potentially toxic comments in real time. It features a user-friendly interface that allows users to paste multiple comments for simultaneous analysis, empowering them to promote and recognize a safe and respectful online environments.

        **Key Features:**

        - **Advanced Classification Technology:** Utilizes state-of-the-art logistic classifiers to assess comments for toxicity.
        - **Real-time Analysis:** Provides instant feedback on the presence of harmful content.
        - **User-Friendly Interface:** Simple and accessible, designed for ease of use.
        - **Bulk Comment Analysis:** Users can conveniently paste multiple comments into the app for simultaneous flagging and assessment.

        **Why Choose Our Red Flag Detector?**

        Empower yourself to take proactive steps against harmful speech with our reliable and straightforward tool.
        
        
        """,unsafe_allow_html=True)

if __name__ == '__main__':
    main()
