

```python
# Dependencies
import json
from pprint import pprint
from datetime import datetime as dt
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import (consumer_key, consumer_secret,
                    access_token, access_token_secret)
```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Function to analyze 500 tweets of the target mention and return a dataframe
def vader(target_handle):
    
    # Create list of dictionaries
    sentiment = []

    # Instantiate tweet count
    tweet_count = 1
    
    # Paginate through 5 pages
    for x in range(25):
       
        # Get all tweets from home feed (for each page specified)
        public_tweets = api.user_timeline(target_handle, page=x)
        
        # Loop through all tweets
        for tweet in public_tweets:
            
            # Grab tweet data
            name = tweet['user']['name']
            tweet_text = tweet['text']
            date = tweet['created_at']
            
            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            compound = results["compound"]
            positive = results['pos']
            neutral = results['neu']
            negative = results['neg']
            
            # Track tweet count
            tweets_ago = tweet_count
            
            # Create dictionary holding tweet data
            tweet_dict = {'Media Source': name, 'Tweet': tweet_text, 'Date': date, 'Compound': compound, 
                          'Positive': positive, 'Neutral': neutral, 'Negative': negative, 'Tweets Ago': tweet_count}
            
            # Increment tweet_count
            tweet_count += 1
            
            # Append tweet dictionary to sentiment list
            sentiment.append(tweet_dict)
            
    # Create dataframe
    sentiment_df = pd.DataFrame(sentiment)

    # Reorder columns
    sentiment_df = sentiment_df.iloc[:,[2,6,1,0,5,4,3,7]]

    # Grab datetime to interpolate into csv name
    today = dt.now().strftime('%Y%m%d') 

    # Save to csv
    sentiment_df.to_csv(f"tweet_data/{today}_{target_handle}_sentiment.csv", encoding='utf-8', index=False)
        
    # Return sentiment analysis
    return sentiment_df
```


```python
# Create function to plot vader sentiment analysis and return image path
def plotSentiment(sentiment_df, target_handle):
    
    # Set axes
    tweets_ago = np.arange(1, 501)
    target = sentiment_df['Compound']

    # Set figure size
    plt.figure(figsize=(10, 7))

    # Plot
    target_plot, = plt.plot(tweets_ago, target, marker='o', ms=7.5, color='steelblue', 
                            alpha=0.8, lw=0.5, label=f"{target_handle}")

    # Set limits
    plt.xlim(505, -5)
    plt.ylim(-1.05, 1.05)

    # Set axes background color
    ax = plt.gca()
    ax.set_facecolor('whitesmoke')

    # Insert grid lines and set behind plot elements
    ax.grid(color='white')
    ax.set_axisbelow(True)

    # Specify max number of ticks in x-axis
    plt.locator_params(axis='y', numticks=1)

    # Get datetime
    today = dt.now().strftime('%m/%d/%y')

    # Labels
    plt.title(f"Sentiment Analysis of Tweets ({today})", fontsize=16)
    plt.xlabel('Tweets Ago', fontsize=14)
    plt.ylabel('Tweet Polarity', fontsize=14)

    # Legend
    lgd = plt.legend(handles=[target_plot], title='Tweets', loc=1, bbox_to_anchor=(1.15, 1))
    
    # Format datetime for saving image
    today = dt.now().strftime('%Y%m%d')

    # Save image and show
    image_path = f"images/{today}_{target_handle}_analysis.png"
    plt.savefig(image_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    # Return image location
    return image_path
```


```python
# Function to reply to sentiment analysis requests
def PlotBot():
    
    # Assign bot screen name
    bot_screen_name = "JustinL64157813"
    
    # Check bot's timeline to see new analysis requests in the last 50 tweets
    public_tweets = api.home_timeline(count=50)

    # Loop through tweets
    for tweet in public_tweets:
        
        # Check bot screen name as first mention
        bot_mention = tweet["entities"]["user_mentions"][0]["screen_name"]
        
        # Varify bot mention
        if bot_mention == bot_screen_name:
        
            # Varify analysis requests in tweet text
            if f"@{bot_mention} Analyze: " in tweet["text"]:
            
                # Grab variables for conditional checks
                target_mention = tweet["entities"]["user_mentions"][1]["screen_name"]
                requester = tweet["user"]["screen_name"]
                tweet_id = tweet["id"]
            
                # Check if target has been analyzed
                if target_mention not in target_handles:
            
                    # Call vader function to generate sentiment dataframe
                    sentiment_df = vader(target_mention)
            
                    # Call plot function to generate plot and image path
                    image_path = plotSentiment(sentiment_df, target_mention)
                                    
                    # Reply to request by tweet id with sentiment analysis and thank you message
                    api.update_with_media(image_path, f"New Tweet Analysis: @{target_mention} (Thx @{requester}!)", 
                                          in_reply_to_status_id=tweet_id)
                
                    # Append target to analyzed handles list
                    target_handles.append(target_mention)
               
                    # Print debug statement
                    print(f"Tweet Analysis for @{target_mention} complete.")
```


```python
# Keep track of analyzed targets for the program
target_handles = []

# Run sentiment analysis bot every 5 minutes
while True:
    
    # Call bot function, append to list above, and print analyzed targets
    PlotBot()
    
    # Print target handles list
    print(f"List of analyzed targets: {target_handles}\n")
    
    # Wait for 5 minutes
    time.sleep(300)
```

    Tweet Analysis for @FoxNews complete.
    Tweet Analysis for @ABC complete.
    Tweet Analysis for @CNN complete.
    List of analyzed targets: ['FoxNews', 'ABC', 'CNN']
    
    Tweet Analysis for @BBCNews complete.
    List of analyzed targets: ['FoxNews', 'ABC', 'CNN', 'BBCNews']
    
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-36-040d0a652007> in <module>()
         12 
         13     # Wait for 5 minutes
    ---> 14     time.sleep(180)
    

    KeyboardInterrupt: 

