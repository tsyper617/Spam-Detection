import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords

import joblib
import colorama
from colorama import Fore, Style

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


nltk.download('stopwords')


# Read CSV file and select the first two columns
df = pd.read_csv('spam.csv', encoding='latin1', usecols=[0, 1])

# Rename the columns
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# Check for null values
df.isnull().sum()

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # convert to lowercase
    text = text.lower()
    # remove special characters
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # remove stopwords
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text

# Clean the messages
df['clean_text'] = df['message'].apply(clean_text)
df.head()

X = df['clean_text']  # Use clean text for input
y = df['label']  # Use original labels for target variable

def classify(model, X, y):
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)

    # Model training
    pipeline_model = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf', model)])
    pipeline_model.fit(x_train, y_train)

    print('Accuracy:', pipeline_model.score(x_test, y_test) * 100)

    y_pred = pipeline_model.predict(x_test)
    #print(classification_report(y_test, y_pred))

    joblib.dump(pipeline_model, 'sms_spam_svc_model.pkl')


model = LogisticRegression()
classify(model, X, y)

model = MultinomialNB()
classify(model, X, y)

thirdmodel = SVC(C=3)
classify(model, X, y)

model = RandomForestClassifier()
classify(model, X, y)

colorama.init(autoreset=True)  # Initialize colorama

# Load the trained model
loaded_model = joblib.load('sms_spam_svc_model.pkl', mmap_mode='r')

# Read the new CSV file with the messages
new_data = pd.read_csv('spamraw.csv')
actual_labels = new_data['type']
messages = new_data['text']

STOPWORDS = set(stopwords.words('english'))

num_messages = 1000
correct_predictions = 0

for index, (message, label) in enumerate(zip(messages.head(num_messages), actual_labels.head(num_messages))):
    cleaned_message = clean_text(message)
    prediction = loaded_model.predict([cleaned_message])[0]
    
    if prediction == 'ham':
        predicted_label = "Regular Message"
    else:
        predicted_label = "Spam"
    
    output = f"Message {index + 1}: Predicted - {predicted_label}, Actual - {label}"
    
    if prediction == label:
        output = Fore.GREEN + output  # Correct prediction in green
        correct_predictions += 1
    else:
        output = Fore.RED + output    # Incorrect prediction in red
    
    print(output)

accuracy = correct_predictions / num_messages * 100
print(Fore.CYAN + f"\nAccuracy: {accuracy:.2f}%")
