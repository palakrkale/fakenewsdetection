import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Custom stopwords
custom_stopwords = [
    'said', 'says', 'say', 'according', 'also', 'could', 'would', 'like', 'even', 'still', 
    'much', 'many', 'may', 'might', 'often', 'every', 'however', 'since', 'though', 'whether', 
    'yet', 'us', 'one', 'two', 'three', 'first', 'last', 'new', 'old', 'year', 'years', 'time', 
    'times', 'day', 'days', 'week', 'weeks', 'month', 'months', 'today', 'yesterday', 'tomorrow', 
    'just', 'now', 'then', 'here', 'there', 'where', 'why', 'how', 'what', 'when', 'who', 'whom', 
    'whose', 'which', 'that', 'this', 'these', 'those', 'their', 'them', 'they', 'he', 'she', 'it', 
    'we', 'our', 'you', 'your', 'its', 'his', 'her', 'him', 'my', 'me', 'myself', 'yourself', 
    'themselves', 'itself', 'himself', 'herself', 'ourselves', 'yourselves', 'again', 'back', 
    'well', 'really', 'very', 'quite', 'rather', 'some', 'any', 'no', 'not', 'only', 'more', 
    'most', 'less', 'least', 'such', 'both', 'either', 'neither', 'none', 'nothing', 'something', 
    'anything', 'everything', 'someone', 'anyone', 'everyone', 'noone', 'nobody', 'somebody', 
    'anybody', 'everybody', 'else', 'other', 'another', 'others', 'each', 'few', 'several', 
    'enough', 'too', 'so', 'as', 'because', 'although', 'though', 'while', 'until', 'unless', 
    'without', 'within', 'upon', 'among', 'between', 'about', 'above', 'below', 'under', 'over', 
    'after', 'before', 'during', 'through', 'into', 'onto', 'toward', 'towards', 'against', 
    'amongst', 'along', 'around', 'across', 'beyond', 'behind', 'beside', 'besides', 'except', 
    'including', 'regarding', 'despite', 'via', 'per'
]
stop_words.update(custom_stopwords)

# Function to preprocess text
def wordopt(text):
    # Lowercase the text
    text = text.lower()

    # Remove text within square brackets
    text = re.sub('\[.*?\]', '', text)

    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)

    # Remove newline characters
    text = re.sub('\n', '', text)

    # Remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)

    # Remove non-alphanumeric characters
    text = re.sub("\\W", " ", text)

    # Remove extra whitespace
    text = re.sub('\s+', ' ', text).strip()

    # Expand contractions
    text = contractions.fix(text)

    # Lemmatization
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

# Function for tokenization and preprocessing
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
    return " ".join(result)

# Load the vectorizer and model
with open("/Users/pk/Desktop/vectorization_0.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("/Users/pk/Desktop/LR_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app
st.title("Fake News Detection using ML")
st.write("Enter the news article text below to check if it's FAKE or REAL.")

# Input text area
user_input = st.text_area("Type your news article: ", "")

if st.button("Predict"):
    if user_input:
        # Preprocess the input text
        cleaned_input = wordopt(user_input)
        preprocessed_input = preprocess(cleaned_input)

        # Vectorize the input text
        input_vector = vectorizer.transform([preprocessed_input])

        # Make prediction
        prediction = model.predict(input_vector)

        # Display the result
        if prediction[0] == 0:
            st.error("This news is likely to be FAKE.")
        else:
            st.success("This news is likely to be REAL.")
    else:
        st.warning("Please enter some text to predict.")
