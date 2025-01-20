import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Referenced from https://www.datacamp.com/tutorial/text-analytics-beginners-nltk

# Downloads additional data that includes pre-trained models, corpora and other services required for NLP.
nltk.download('all')

# Initialize the sentiment analyzer from NLTK
analyzer = SentimentIntensityAnalyzer()


def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment


def apply_nltk_analysis(file, field_to_analyse):
    df = pd.read_csv(file.file)
    df[f'{field_to_analyse}'] = df[f'{field_to_analyse}'].apply(preprocess_text)
    df['sentiment'] = df[f'{field_to_analyse}'].apply(get_sentiment)
    average_sentiment = float(df['sentiment'].mean())
    if average_sentiment >= 0.8:
        sentiment = "Very Positive"
    elif average_sentiment >= 0.6:
        sentiment = "Positive"
    elif average_sentiment >= 0.5:
        sentiment = "Neutral"
    elif average_sentiment >= 0.4:
        sentiment = "Negative"
    else:
        sentiment = "Very Negative"

    return sentiment