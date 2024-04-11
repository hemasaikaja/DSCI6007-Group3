import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn import tree


# Load the data
data = pd.read_csv("Information (2).csv", encoding='latin-1')

# Drop unnecessary columns
data = data.drop(columns=['tweet_coord', 'profile_yn_gold', 'gender_gold'])

# Handle missing values
data_mode = data.mode()
for gc in data.columns.values:
    data[gc] = data[gc].fillna(value=data_mode[gc].iloc[0])
data = data.fillna(method='ffill')

# Train-Test Split
X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(
    data['text'], data['gender'], test_size=0.2, random_state=42
)

# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')  # You can customize stop_words if needed
X_train_vectorized = vectorizer.fit_transform(X_train_list)

# Build the Bagging Classifier
model_bagging = BaggingClassifier(tree.DecisionTreeClassifier(random_state=70))
model_bagging.fit(X_train_vectorized, y_train_list)

# Save the Bagging Classifier Model
joblib.dump(model_bagging, 'bagging_classifier_model.pkl')

# Build the Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=4)
model_rf.fit(X_train_vectorized, y_train_list)

# Save the Random Forest Classifier Model
joblib.dump(model_rf, 'random_forest_classifier_model.pk')
