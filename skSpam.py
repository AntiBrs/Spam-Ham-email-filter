import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import string
import nltk

def preprocess_text(text):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokens = [word for word in tokens if word not in stopwords]
    tokens = [word for word in tokens if word not in string.punctuation and not word.isdigit()]
    return " ".join(tokens)

data_files = glob("*.csv")
data = pd.concat((pd.read_csv(filename) for filename in data_files), ignore_index=True)

X = data["CONTENT"].apply(preprocess_text)
y = data["CLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)

classifiers = [
    ("MultinomialNB", MultinomialNB()),
    ("ComplementNB", ComplementNB()),
    ("SGDClassifier (log_loss)", SGDClassifier(loss="log_loss")),
    ("SGDClassifier (hinge loss)", SGDClassifier(loss="hinge"))
]

for name, classifier in classifiers:
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", classifier)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion_matrix_result = confusion_matrix(y_test, y_pred)

    print(f"\n{name}:")
    print(f"Accuracy: {accuracy}")
    print(confusion_matrix_result)
