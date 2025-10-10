import xml.etree.ElementTree as ET
from collections import defaultdict
def parse_xml(path):
    root = ET.parse(path).getroot()
    pages = {}
    for p in root.findall('page'):
        pid = p.get('id')
        title = (p.findtext('title') or '').lower()
        text = (p.findtext('text') or '').lower()
        out = [l.get('to') or (l.text or '') for l in p.find('links').findall('link')] if p.find('links') else []
        pages[pid] = {'title': title, 'text': text, 'out': out}
    return pages
def build_graph(pages):
    G = defaultdict(list)
    for pid, p in pages.items():
        for d in p['out']:
            if d in pages:
                G[pid].append(d)
    return G
def personalize(pages, topic):
    topic = set(topic.lower().split())
    scores = {pid: sum(w in topic for w in (p['title']+p['text']).split()) for pid,p in pages.items()}
    s = sum(scores.values()) or len(pages)
    return {k: v/s if s else 1/len(pages) for k,v in scores.items()}
def pagerank(G, p, alpha=0.85, iters=50):
    nodes = list(G.keys())
    n = len(nodes)
    pr = {u: 1/n for u in nodes}
    for _ in range(iters):
        new = {u: (1-alpha)*p.get(u,0) for u in nodes}
        for u in nodes:
            if G[u]:
                share = alpha*pr[u]/len(G[u])
                for v in G[u]: new[v]+=share
            else:
                for v in nodes: new[v]+=alpha*pr[u]/n
        pr = new
    return dict(sorted(pr.items(), key=lambda x:-x[1]))
if __name__ == "__main__":
    xml = """<corpus>
<page id='A'><title>Neural Networks</title><text>deep learning</text><links><link to='B'/></links></page>
<page id='B'><title>Graph Theory</title><text>pagerank random walk</text><links><link to='A'/></links></page>
</corpus>"""
    root = ET.fromstring(xml)
    pages = {}
    for p in root.findall('page'):
        pid = p.get('id')
        title = (p.findtext('title') or '').lower()
        text = (p.findtext('text') or '').lower()
        out = [l.get('to') or (l.text or '') for l in p.find('links').findall('link')] if p.find('links') else []
        pages[pid] = {'title': title, 'text': text, 'out': out}
    G = build_graph(pages)
    pvec = personalize(pages, 'graph pagerank')
    scores = pagerank(G, pvec)
    print("\nTopic-Specific PageRank Results")
    print("-" * 45)
    print(f"{'Page':<6} {'Title':<20} {'Score'}")
    print("-" * 45)
    for pid, score in scores.items():
        title = pages[pid]['title'].title()
        print(f"{pid:<6} {title:<20} {score:.6f}")
