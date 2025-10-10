import re
from collections import Counter
import spacy
tweets=["AI is transforming the future of technology in New York!","OpenAI released ChatGPT, and Microsoft is investing heavily.","Elon Musk talks about artificial intelligence and Tesla.","Google DeepMind achieves new breakthroughs in AI research.","The AI Act in the United States will regulate artificial intelligence."]
def clean_text(text):
    text=re.sub(r"http\S+|@\w+|#","",text)
    return re.sub(r"[^A-Za-z\s]","",text).lower().strip()
cleaned=[clean_text(t) for t in tweets]
words=" ".join(cleaned).split()
print("Top words:",Counter(words).most_common(5))
nlp=spacy.load("en_core_web_sm")
entities=[]
for t in tweets:
    for ent in nlp(t).ents:
        entities.append((ent.text,ent.label_))
print("\nTop entities:",Counter(entities).most_common(5))
