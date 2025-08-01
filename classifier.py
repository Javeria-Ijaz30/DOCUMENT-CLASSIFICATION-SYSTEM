import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class DocumentClassifier:
    def __init__(self, k=3):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.classifier = KNeighborsClassifier(n_neighbors=k)

    def train(self, documents, labels):
        X = self.vectorizer.fit_transform(documents)
        self.classifier.fit(X, labels)

    def predict(self, new_doc):
        X_new = self.vectorizer.transform([new_doc])
        return self.classifier.predict(X_new)

    def evaluate(self, documents, labels):
        X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.3)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.classifier.fit(X_train_vec, y_train)
        y_pred = self.classifier.predict(X_test_vec)

        print(classification_report(y_test, y_pred))
