import requests
from bs4 import BeautifulSoup
from googlesearch import search
def fetch_news(topic, limit_websites=None, num_results=10):
    query = topic + " news"
    results = []
    for url in search(query, num_results=num_results):
        if limit_websites:
            if not any(site in url for site in limit_websites):
                continue
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.find("title").get_text() if soup.find("title") else "No Title"
            description = ""
            meta = soup.find("meta", attrs={"name": "description"})
            if meta:
                description = meta.get("content", "")
            results.append({"title": title, "url": url, "description": description})
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
    return results
if __name__ == "__main__":
    topic = input("Enter the topic to search news for: ")
    choice = input("Do you want to restrict to specific websites? (y/n): ").lower()
    websites = None
    if choice == "y":
        websites = input("Enter websites separated by comma (e.g., bbc.com,reuters.com): ").split(",")
    news_list = fetch_news(topic, limit_websites=websites, num_results=15)
    print("\n News Stories Found:\n")
    for i, news in enumerate(news_list, 1):
        print(f"{i}. {news['title']}")
        print(f"  {news['url']}")
        if news['description']:
            print(f"  -- {news['description']}")
        print("-" * 80)
