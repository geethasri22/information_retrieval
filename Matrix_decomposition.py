import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(newsgroups.data)
print("Step 1: TF-IDF Matrix Shape =", X.shape)
n_components = 100
svd = TruncatedSVD(n_components=n_components)
X_lsi = svd.fit_transform(X)
print("Step 2: Reduced LSI Matrix Shape =", X_lsi.shape)
explained_variance = svd.explained_variance_ratio_.sum()
print(f"Step 3: Total Explained Variance by {n_components} topics = {explained_variance:.2f}")
terms = vectorizer.get_feature_names_out()
print("\nStep 4: Top Terms for First 5 Latent Topics:\n")
for i, comp in enumerate(svd.components_[:5]):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:10]
    print(f"Topic {i+1}: ", [t[0] for t in sorted_terms])
cum_var = svd.explained_variance_ratio_.cumsum()
plt.figure(figsize=(8,6))
plt.plot(np.arange(1, n_components+1), cum_var, marker='o', markersize=3, linewidth=1.2)
plt.xlabel("Number of Topics (Components)")
plt.ylabel("Cumulative Explained Variance")
plt.title("LSI - Variance Explained Curve")
plt.grid(True)
elbow = np.argmax(cum_var > 0.6) + 1
plt.axvline(x=elbow, color='red', linestyle='--', label=f'Elbow ~{elbow} topics')
plt.legend()
plt.show()
