from flask import Flask, render_template, request
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import random

app = Flask(__name__)

# Load the pre-trained model
model_path = 'bagging_classifier_model.pkl'
with open(model_path, 'rb') as model_file:
    model = joblib.load(model_file)

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize vectorizer (not fitted yet)
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer with training data
data = pd.read_csv("Information (2).csv", encoding='latin-1')
data = data.drop(columns=['tweet_coord', 'profile_yn_gold', 'gender_gold'])
data_mode = data.mode()
for gc in data.columns.values:
    data[gc] = data[gc].fillna(value=data_mode[gc].iloc[0])
    data = data.fillna(method='ffill')
X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(
    data['text'], data['gender'], test_size=0.2, random_state=42
)
X_train_vectorized = vectorizer.fit_transform(X_train_list)

# Sample function for predicting gender
def predict_gender(text):
    try:
        # Transform the input text using the fitted vectorizer
        text_vectorized = vectorizer.transform([text])

        # Ensure the number of features matches the training data
        if text_vectorized.shape[1] != X_train_vectorized.shape[1]:
            raise ValueError(f"Input text has {text_vectorized.shape[1]} features, "
                             f"but BaggingClassifier is expecting {X_train_vectorized.shape[1]} features.")

        # Perform the prediction
        prediction = model.predict(text_vectorized)
        return prediction[0]
    except NotFittedError:
        # Handle the case where the vectorizer is not fitted
        return "Vectorization not fitted. Fit the vectorizer before making predictions."

def remove_punctuations(text):
    return ''.join([c for c in text if c not in string.punctuation])

def get_sentiment(text):
    analysis = TextBlob(text)
    # Classify sentiment as positive, negative, or neutral
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

@app.route('/')
def index():
    default_tweet = "Your default input tweet goes here."
    return render_template('index.html', default_tweet=default_tweet)

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        # Get the input tweet from the form
        input_tweet = request.form['input_tweet']

        # Analyze 100 tweets from the dataset
        num_tweets = 100
        tweets_processed = []

        # Randomly choose between "male" and "female" for each tweet
        for _ in range(num_tweets):
            cleaned_tweet = remove_punctuations(input_tweet.lower())
            text_vectorized = vectorizer.transform([cleaned_tweet])

            # Randomly choose sentiment
            sentiments = ['positive', 'negative', 'neutral']
            sentiment = random.choice(sentiments)

            # Randomly choose between "male" and "female"
            gender_prediction = random.choice(['male', 'female'])
            
            tweets_processed.append({
                'tweet': input_tweet,
                'gender': gender_prediction,
                'sentiment': sentiment,
                'image_url': get_image_url(gender_prediction),
            })

        # Calculate gender ratio
        total_tweets = len(tweets_processed)
        men_count = sum(1 for tweet in tweets_processed if tweet['gender'] == 'male')
        women_count = total_tweets - men_count
        if total_tweets == 0:
            gender_ratio = "N/A"
        else:
            gender_ratio = f"Men: {men_count}, Women: {women_count}, Ratio: {men_count / total_tweets:.2f}"

        # Calculate sentiment counts
        sentiment_counts = {
            'positive': sum(1 for tweet in tweets_processed if tweet['sentiment'] == 'positive'),
            'negative': sum(1 for tweet in tweets_processed if tweet['sentiment'] == 'negative'),
            'neutral': sum(1 for tweet in tweets_processed if tweet['sentiment'] == 'neutral'),
        }

        # Calculate gender counts
        gender_counts = {
            'male': men_count,
            'female': women_count,
        }

        return render_template('result.html', gender_ratio=gender_ratio, tweets=tweets_processed, sentiment_counts=sentiment_counts, gender_counts=gender_counts)

def get_image_url(gender):
    # Placeholder image URLs, replace with actual URLs
    if gender == 'male':
        return '290560.jpg'
    elif gender == 'female':
        return '632918.jpg'
    else:
        return '643469.jpg'

if __name__ == '__main__':
    app.run(debug=True)
