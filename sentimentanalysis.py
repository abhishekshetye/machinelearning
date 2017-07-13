import tweepy
from textblob import TextBlob

consumer_key = 'UaTbauUSnv2oH6tYZyhRxGv9b'
consumer_secret = 'bQgcG0mxGae8wCWYQ3DxMA2iMYf4zYFggpllw7DqwPlZTQwn8t'

access_token = '4031525956-Du3NnkacWsagdMUcWZBAoAT6d3xqHGAyJJvITq6'
access_secret = '4S9xnxbaLYhxOULdJjMKid1I2e7c6C2dbkd87kZRbAx6q'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

for tweet in public_tweets:
	print(tweet.text)
	analysis = TextBlob(tweet.text)
	print(analysis.sentiment)