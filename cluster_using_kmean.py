from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import confusion_matrix
categories = ['alt.atheism', 'comp.graphics', 'sci.space', 'rec.sport.hockey']
dataset = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
true_labels = dataset.target
num_clusters = len(categories)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5)
X = vectorizer.fit_transform(dataset.data)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
predicted_clusters = kmeans.fit_predict(X)
def purity_score(y_true, y_pred):
    contingency_matrix = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        cluster_i_indices = np.where(y_pred == i)[0]
        true_labels_i = y_true[cluster_i_indices]
        most_common = np.bincount(true_labels_i).max()
        contingency_matrix[i, :] = np.bincount(true_labels_i, minlength=num_clusters)
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)
purity = purity_score(true_labels, predicted_clusters)
cm = confusion_matrix(true_labels, predicted_clusters)
indexes = linear_assignment(-cm)
mapping = {cluster: label for cluster, label in indexes}
mapped_preds = np.array([mapping[cluster] for cluster in predicted_clusters])
precision = precision_score(true_labels, mapped_preds, average='macro')
recall = recall_score(true_labels, mapped_preds, average='macro')
f1 = f1_score(true_labels, mapped_preds, average='macro')
print(f"Purity: {purity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-measure: {f1:.4f}")
