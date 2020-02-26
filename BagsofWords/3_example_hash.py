from sklearn.feature_extraction.text import HashingVectorizer

text = ["The quick brown fox jumped over the lazy dog."]

vectorizer = HashingVectorizer(n_features=20)
vector = vectorizer.transform(text)
print(vector.shape)
print(vector.toarray())