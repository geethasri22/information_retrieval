import os
import nltk
import json
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
docs_folder = './docs'
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
inverted_index = defaultdict(set)
total_docs = 0
for filename in os.listdir(docs_folder):
    if filename.endswith('.txt'):
        total_docs += 1
        doc_id = total_docs
        file_path = os.path.join(docs_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = word_tokenize(text.lower())
        tokens = [
            stemmer.stem(word)
            for word in tokens
            if word.isalnum() and word not in stop_words
        ]
        for word in set(tokens):
            inverted_index[word].add(doc_id)
print(f"Total documents indexed: {total_docs}")
print(f"Vocabulary size: {len(inverted_index)}")
inverted_index = {term: sorted(list(doc_ids)) for term, doc_ids in inverted_index.items()}
with open('inverted_index.json', 'w', encoding='utf-8') as f:
    json.dump(inverted_index, f, indent=4)
print(f"Inverted index saved to 'inverted_index.json'.")
example_terms = list(inverted_index.keys())[:10]
for term in example_terms:
    print(f"'{term}': {inverted_index[term]}")
