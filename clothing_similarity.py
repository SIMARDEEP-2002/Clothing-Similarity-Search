import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

def get_similar_items(query, url, n):


    # Check if the URL is correct.
    if not url:
        raise ValueError("Please provide a valid URL.")

    # Check if the website is up and running.
    try:
        response = requests.get(url)
    except requests.exceptions.ConnectionError:
        raise ValueError("The website is not up and running.")

    # Check if the website allows scraping.
    if response.status_code == 403:
        raise ValueError("The website does not allow scraping.")

    # Scrape the data from the website.
    soup = BeautifulSoup(response.content, "html.parser")

    # Get the clothing item descriptions and URLs.
    descriptions = []
    urls = []
    for product in soup.find_all("a", class_="product-description"):
        try:
            descriptions.append(product.text)
            urls.append(product["href"])
        except:
            pass

    # Clean the data.
    descriptions = [description.lower() for description in descriptions]
    descriptions = [description.replace("[^a-zA-Z]", " ") for description in descriptions]

    # Remove stop words.
    stop_words = set(stopwords.words("english"))
    descriptions = [" ".join([word for word in description.split() if word not in stop_words]) for description in descriptions]

    # Check if there are any non-empty descriptions.
    if not descriptions:
        return []

    # Create a vectorizer.
    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # Vectorize the descriptions.
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # Transform the query.
    query_vector = vectorizer.transform([query])

    # Compute cosine similarities between the query and descriptions.
    similarities = cosine_similarity(query_vector, tfidf_matrix)

    # Get the indices of the top-N most similar items.
    indices = similarities.argsort()[0][-n:][::-1]

    # Get the URLs of the top-N most similar items.
    top_n_urls = [urls[index] for index in indices]

    return top_n_urls

# Get the user's query.
query = input("Enter your query: ")

# Define the URL of the website to scrape.
url = "https://www.asos.com/women/dresses/cat/?cid=8799"

# Define the number of items to return.
n = 3

# Get the top-N most similar items.
similar_items = get_similar_items(query, url, n)

# Print the URLs of the top-N most similar items.
if similar_items:
    for url in similar_items:
        print(url)
else:
    print("No similar items found.")
