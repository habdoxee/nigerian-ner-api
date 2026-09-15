import os
import tweepy
from dotenv import load_dotenv

load_dotenv()

bearer_token = os.getenv("X_BEARER_TOKEN")

client = tweepy.Client(bearer_token=bearer_token)

response = client.search_recent_tweets(
    query="Nigeria",
    max_results=10
)

if response.data:
    for tweet in response.data:
        print(tweet.text)
else:
    print("No tweets found.")