"""
Stop words are common words in a language that usually carry little meaningful information
for tasks like indexing or searching. Examples include:
 English: “a”, “the”, “and”, “is”, “in”
Stop words are removed because
 They do not help in distinguishing between documents.
 They reduce storage and speed up search operations.
 Improves focus on the meaningful terms (keywords).
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
text = "Students are studying studies in universities."
words = word_tokenize(text)
print("Original Words:", words)
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.lower() not in stop_words]
print("After Stop Word Removal:", filtered_words)
ps = PorterStemmer()
stemmed_words = [ps.stem(w) for w in filtered_words]
print("After Stemming:", stemmed_words)
