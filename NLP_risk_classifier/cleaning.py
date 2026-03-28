# src/cleaning.py
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def clean_text(text: str):
    """Clean text → tokens."""
    import nltk
    tokens = nltk.word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemm.lemmatize(t) for t in tokens]
    return tokens
