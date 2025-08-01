from scraper import scrape_website
from classifier import DocumentClassifier

# Sample training data
documents = [
    "The stock market crashed due to economic slowdown.",
    "New algorithm improves the accuracy of neural networks.",
    "The government passed a new education bill today.",
    "Deep learning revolutionizes image classification."
]
labels = ["News", "Research", "News", "Research"]

# Train the model
model = DocumentClassifier()
model.train(documents, labels)

# Scrape and predict
url = "https://www.bbc.com/news/technology"  # example site
text = scrape_website(url)
predicted = model.predict(text)

print("Predicted Category:", predicted[0])
