from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
doc1 = "Information Retrieval is fun"
doc2 = "Information Extraction is interesting"
documents = [doc1, doc2]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("\nVocabulary (terms):")
print(vectorizer.get_feature_names_out())
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print("\nCosine Similarity between Document 1 and Document 2:")
print(similarity[0][0])

"""
In the Vector Space Model, a document is represented as a vector of weights, where:
 Each dimension corresponds to a unique term (word) in the overall vocabulary (the
corpus).
 The value along each dimension represents the weight of that term in the document —
commonly calculated using TF-IDF.
TF-IDF stands for:
 TF (Term Frequency): how often the term appears in the document.
 IDF (Inverse Document Frequency): how unique the term is across the whole
collection.
So, each document is like a point in a high-dimensional space, where each dimension is a
word.
"""
