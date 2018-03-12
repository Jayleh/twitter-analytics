
# Twitter News Sentiment Analysis
##### Observable Trends:
###### 1. The scatter plot shows tweet sentiments of each news source scattered across vader's tweet polarity. As you look closer at news sources individually you can see that some have more positive tweets than negative, or vice versa. 
###### 2. As of March 12th, 2018, BBCNews has the most negative average compound score while CBS has the most positive average compound score out of the 5 news sources analyzed.
###### 3. However, according to the vader sentiment documentation, the average compound scores for these news sources fall in the range of neutral sentiment.


```python
# Dependencies
import json
from pprint import pprint
from datetime import datetime as dt
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
# Target user
target_list = ['@BBCNews', '@CBS', '@CNN', '@FoxNews', '@nytimes']

# Create list of dictionaries
sentiment = []

tweet_count = 1

for target_user in target_list:
    
    # Paginate through 5 pages
    for x in range(5):
       
        # Get all tweets from home feed (for each page specified)
        public_tweets = api.user_timeline(target_user, page=x)
        
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
            
            # Only count up to 100 for each target user
            if tweets_ago == 100:
                # Reset tweet counter
                tweet_count = 1
            else:
                tweet_count += 1
            
            # Append tweet dictionary to sentiment list
            sentiment.append(tweet_dict)
```


```python
# Create dataframe
media_tweets = pd.DataFrame(sentiment)

# Reorder columns
media_tweets = media_tweets.iloc[:,[2,6,1,0,5,4,3,7]]

# Grab datetime to interpolate into csv name
today = dt.now().strftime('%Y%m%d') 

# Save to csv
media_tweets.to_csv(f'tweet_data/{today}_news_sentiment.csv', encoding='utf-8', index=False)
media_tweets
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Media Source</th>
      <th>Tweet</th>
      <th>Date</th>
      <th>Compound</th>
      <th>Positive</th>
      <th>Neutral</th>
      <th>Negative</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCSport: Jamie Carragher has apologised f...</td>
      <td>Mon Mar 12 20:25:56 +0000 2018</td>
      <td>0.1027</td>
      <td>0.065</td>
      <td>0.935</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BBC News (UK)</td>
      <td>Chloe Miazek death: Mark Bruce admits Aberdeen...</td>
      <td>Mon Mar 12 19:17:43 +0000 2018</td>
      <td>-0.7964</td>
      <td>0.133</td>
      <td>0.364</td>
      <td>0.503</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BBC News (UK)</td>
      <td>Mesh surgeon removed ovaries without prior con...</td>
      <td>Mon Mar 12 19:12:59 +0000 2018</td>
      <td>-0.1695</td>
      <td>0.000</td>
      <td>0.808</td>
      <td>0.192</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCSport: Eddie Jones is open to bringing ...</td>
      <td>Mon Mar 12 19:08:15 +0000 2018</td>
      <td>-0.4019</td>
      <td>0.108</td>
      <td>0.677</td>
      <td>0.215</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BBC News (UK)</td>
      <td>Coronation Street unveils new on-set memorial ...</td>
      <td>Mon Mar 12 18:45:31 +0000 2018</td>
      <td>-0.3182</td>
      <td>0.000</td>
      <td>0.813</td>
      <td>0.187</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BBC News (UK)</td>
      <td>RT @bbcouch: Why is drinking water so bad for ...</td>
      <td>Mon Mar 12 18:34:24 +0000 2018</td>
      <td>-0.6696</td>
      <td>0.000</td>
      <td>0.757</td>
      <td>0.243</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCWorldatOne: A comedy “northern powerhou...</td>
      <td>Mon Mar 12 18:28:59 +0000 2018</td>
      <td>0.8689</td>
      <td>0.366</td>
      <td>0.634</td>
      <td>0.000</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCJonSopel: Now that @theresa_may has con...</td>
      <td>Mon Mar 12 18:23:43 +0000 2018</td>
      <td>-0.5859</td>
      <td>0.000</td>
      <td>0.840</td>
      <td>0.160</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BBC News (UK)</td>
      <td>Russia reacts to @theresa_may statement on Sal...</td>
      <td>Mon Mar 12 17:51:05 +0000 2018</td>
      <td>-0.5859</td>
      <td>0.000</td>
      <td>0.703</td>
      <td>0.297</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BBC News (UK)</td>
      <td>Salisbury poisoning was "indiscriminate and re...</td>
      <td>Mon Mar 12 17:38:37 +0000 2018</td>
      <td>-0.5719</td>
      <td>0.138</td>
      <td>0.542</td>
      <td>0.320</td>
      <td>10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCAlagiah: Theresa May says “no question ...</td>
      <td>Mon Mar 12 17:23:58 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>11</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BBC News (UK)</td>
      <td>Highly likely Russia behind spy attack, says T...</td>
      <td>Mon Mar 12 17:19:37 +0000 2018</td>
      <td>-0.4767</td>
      <td>0.000</td>
      <td>0.744</td>
      <td>0.256</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCBreaking: Former Russian spy &amp;amp; his ...</td>
      <td>Mon Mar 12 17:15:51 +0000 2018</td>
      <td>-0.4939</td>
      <td>0.000</td>
      <td>0.868</td>
      <td>0.132</td>
      <td>13</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BBC News (UK)</td>
      <td>Celibacy and abstinence should be promoted as ...</td>
      <td>Mon Mar 12 16:31:07 +0000 2018</td>
      <td>0.7506</td>
      <td>0.368</td>
      <td>0.632</td>
      <td>0.000</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>BBC News (UK)</td>
      <td>Tube push attempted murder: Alan Alencar secti...</td>
      <td>Mon Mar 12 16:27:03 +0000 2018</td>
      <td>-0.6908</td>
      <td>0.000</td>
      <td>0.598</td>
      <td>0.402</td>
      <td>15</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BBC News (UK)</td>
      <td>Giant panda breeding programme at Edinburgh Zo...</td>
      <td>Mon Mar 12 16:09:35 +0000 2018</td>
      <td>-0.4767</td>
      <td>0.000</td>
      <td>0.721</td>
      <td>0.279</td>
      <td>16</td>
    </tr>
    <tr>
      <th>16</th>
      <td>BBC News (UK)</td>
      <td>Kathmandu airport crash: Footage shows wreckag...</td>
      <td>Mon Mar 12 15:58:17 +0000 2018</td>
      <td>-0.8720</td>
      <td>0.000</td>
      <td>0.568</td>
      <td>0.432</td>
      <td>17</td>
    </tr>
    <tr>
      <th>17</th>
      <td>BBC News (UK)</td>
      <td>Millennial 26-30 railcard to launch nationwide...</td>
      <td>Mon Mar 12 15:47:44 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>18</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BBC News (UK)</td>
      <td>Meghan Markle joins Queen for Commonwealth Day...</td>
      <td>Mon Mar 12 15:34:09 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>19</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCParliament: Speaker John Bercow is chai...</td>
      <td>Mon Mar 12 15:21:12 +0000 2018</td>
      <td>0.2023</td>
      <td>0.083</td>
      <td>0.917</td>
      <td>0.000</td>
      <td>20</td>
    </tr>
    <tr>
      <th>20</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCSport: Jamie Carragher has apologised f...</td>
      <td>Mon Mar 12 20:25:56 +0000 2018</td>
      <td>0.1027</td>
      <td>0.065</td>
      <td>0.935</td>
      <td>0.000</td>
      <td>21</td>
    </tr>
    <tr>
      <th>21</th>
      <td>BBC News (UK)</td>
      <td>Chloe Miazek death: Mark Bruce admits Aberdeen...</td>
      <td>Mon Mar 12 19:17:43 +0000 2018</td>
      <td>-0.7964</td>
      <td>0.133</td>
      <td>0.364</td>
      <td>0.503</td>
      <td>22</td>
    </tr>
    <tr>
      <th>22</th>
      <td>BBC News (UK)</td>
      <td>Mesh surgeon removed ovaries without prior con...</td>
      <td>Mon Mar 12 19:12:59 +0000 2018</td>
      <td>-0.1695</td>
      <td>0.000</td>
      <td>0.808</td>
      <td>0.192</td>
      <td>23</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCSport: Eddie Jones is open to bringing ...</td>
      <td>Mon Mar 12 19:08:15 +0000 2018</td>
      <td>-0.4019</td>
      <td>0.108</td>
      <td>0.677</td>
      <td>0.215</td>
      <td>24</td>
    </tr>
    <tr>
      <th>24</th>
      <td>BBC News (UK)</td>
      <td>Coronation Street unveils new on-set memorial ...</td>
      <td>Mon Mar 12 18:45:31 +0000 2018</td>
      <td>-0.3182</td>
      <td>0.000</td>
      <td>0.813</td>
      <td>0.187</td>
      <td>25</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BBC News (UK)</td>
      <td>RT @bbcouch: Why is drinking water so bad for ...</td>
      <td>Mon Mar 12 18:34:24 +0000 2018</td>
      <td>-0.6696</td>
      <td>0.000</td>
      <td>0.757</td>
      <td>0.243</td>
      <td>26</td>
    </tr>
    <tr>
      <th>26</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCWorldatOne: A comedy “northern powerhou...</td>
      <td>Mon Mar 12 18:28:59 +0000 2018</td>
      <td>0.8689</td>
      <td>0.366</td>
      <td>0.634</td>
      <td>0.000</td>
      <td>27</td>
    </tr>
    <tr>
      <th>27</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCJonSopel: Now that @theresa_may has con...</td>
      <td>Mon Mar 12 18:23:43 +0000 2018</td>
      <td>-0.5859</td>
      <td>0.000</td>
      <td>0.840</td>
      <td>0.160</td>
      <td>28</td>
    </tr>
    <tr>
      <th>28</th>
      <td>BBC News (UK)</td>
      <td>Russia reacts to @theresa_may statement on Sal...</td>
      <td>Mon Mar 12 17:51:05 +0000 2018</td>
      <td>-0.5859</td>
      <td>0.000</td>
      <td>0.703</td>
      <td>0.297</td>
      <td>29</td>
    </tr>
    <tr>
      <th>29</th>
      <td>BBC News (UK)</td>
      <td>Salisbury poisoning was "indiscriminate and re...</td>
      <td>Mon Mar 12 17:38:37 +0000 2018</td>
      <td>-0.5719</td>
      <td>0.138</td>
      <td>0.542</td>
      <td>0.320</td>
      <td>30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>The New York Times</td>
      <td>At restaurants across America, servers calcula...</td>
      <td>Mon Mar 12 14:21:03 +0000 2018</td>
      <td>-0.5423</td>
      <td>0.000</td>
      <td>0.837</td>
      <td>0.163</td>
      <td>71</td>
    </tr>
    <tr>
      <th>471</th>
      <td>The New York Times</td>
      <td>Goldman Sachs cleared the way for a successor ...</td>
      <td>Mon Mar 12 14:10:06 +0000 2018</td>
      <td>0.3182</td>
      <td>0.163</td>
      <td>0.837</td>
      <td>0.000</td>
      <td>72</td>
    </tr>
    <tr>
      <th>472</th>
      <td>The New York Times</td>
      <td>Morning Briefing: Here's what you need to know...</td>
      <td>Mon Mar 12 14:00:24 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>73</td>
    </tr>
    <tr>
      <th>473</th>
      <td>The New York Times</td>
      <td>10 myths about your NCAA bracket https://t.co/...</td>
      <td>Mon Mar 12 13:44:07 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>74</td>
    </tr>
    <tr>
      <th>474</th>
      <td>The New York Times</td>
      <td>"Does anyone here know who was imprisoned here...</td>
      <td>Mon Mar 12 13:30:14 +0000 2018</td>
      <td>-0.4588</td>
      <td>0.000</td>
      <td>0.857</td>
      <td>0.143</td>
      <td>75</td>
    </tr>
    <tr>
      <th>475</th>
      <td>The New York Times</td>
      <td>What's the actual story behind the O.J. Simpso...</td>
      <td>Mon Mar 12 13:14:06 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>76</td>
    </tr>
    <tr>
      <th>476</th>
      <td>The New York Times</td>
      <td>If President Trump pulls out of the Iran deal,...</td>
      <td>Mon Mar 12 13:00:14 +0000 2018</td>
      <td>0.2732</td>
      <td>0.087</td>
      <td>0.913</td>
      <td>0.000</td>
      <td>77</td>
    </tr>
    <tr>
      <th>477</th>
      <td>The New York Times</td>
      <td>A passenger plane from Bangladesh slammed into...</td>
      <td>Mon Mar 12 12:47:04 +0000 2018</td>
      <td>-0.4939</td>
      <td>0.000</td>
      <td>0.819</td>
      <td>0.181</td>
      <td>78</td>
    </tr>
    <tr>
      <th>478</th>
      <td>The New York Times</td>
      <td>Dropbox said that it hoped to raise as much as...</td>
      <td>Mon Mar 12 12:30:10 +0000 2018</td>
      <td>0.3818</td>
      <td>0.126</td>
      <td>0.874</td>
      <td>0.000</td>
      <td>79</td>
    </tr>
    <tr>
      <th>479</th>
      <td>The New York Times</td>
      <td>The White House said it wanted to partner with...</td>
      <td>Mon Mar 12 12:15:05 +0000 2018</td>
      <td>-0.2732</td>
      <td>0.000</td>
      <td>0.900</td>
      <td>0.100</td>
      <td>80</td>
    </tr>
    <tr>
      <th>480</th>
      <td>The New York Times</td>
      <td>Morning Briefing: Here's what you need to know...</td>
      <td>Mon Mar 12 12:00:12 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>81</td>
    </tr>
    <tr>
      <th>481</th>
      <td>The New York Times</td>
      <td>Virginia was named the top overall seed in the...</td>
      <td>Mon Mar 12 11:44:01 +0000 2018</td>
      <td>0.2023</td>
      <td>0.107</td>
      <td>0.893</td>
      <td>0.000</td>
      <td>82</td>
    </tr>
    <tr>
      <th>482</th>
      <td>The New York Times</td>
      <td>The passengers of the helicopter that crashed ...</td>
      <td>Mon Mar 12 11:31:02 +0000 2018</td>
      <td>-0.2732</td>
      <td>0.000</td>
      <td>0.909</td>
      <td>0.091</td>
      <td>83</td>
    </tr>
    <tr>
      <th>483</th>
      <td>The New York Times</td>
      <td>Under guard at his home, the Venezuelan opposi...</td>
      <td>Mon Mar 12 11:15:09 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>84</td>
    </tr>
    <tr>
      <th>484</th>
      <td>The New York Times</td>
      <td>China's National People’s Congress opened the ...</td>
      <td>Mon Mar 12 11:00:22 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>85</td>
    </tr>
    <tr>
      <th>485</th>
      <td>The New York Times</td>
      <td>A commercial flight from Bangladesh crashed at...</td>
      <td>Mon Mar 12 10:46:04 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>86</td>
    </tr>
    <tr>
      <th>486</th>
      <td>The New York Times</td>
      <td>Morning Briefing: Here's what you need to know...</td>
      <td>Mon Mar 12 10:30:13 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>87</td>
    </tr>
    <tr>
      <th>487</th>
      <td>The New York Times</td>
      <td>“The level of tension between the United State...</td>
      <td>Mon Mar 12 10:13:02 +0000 2018</td>
      <td>-0.2944</td>
      <td>0.105</td>
      <td>0.709</td>
      <td>0.186</td>
      <td>88</td>
    </tr>
    <tr>
      <th>488</th>
      <td>The New York Times</td>
      <td>Stormy Daniels and her lawyer are pursuing an ...</td>
      <td>Mon Mar 12 09:59:02 +0000 2018</td>
      <td>-0.4215</td>
      <td>0.000</td>
      <td>0.774</td>
      <td>0.226</td>
      <td>89</td>
    </tr>
    <tr>
      <th>489</th>
      <td>The New York Times</td>
      <td>RT @nytimesworld: Brussels is astir after a bu...</td>
      <td>Mon Mar 12 09:45:07 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>90</td>
    </tr>
    <tr>
      <th>490</th>
      <td>The New York Times</td>
      <td>Marine Le Pen hopes a rebranding may reinvigor...</td>
      <td>Mon Mar 12 09:30:09 +0000 2018</td>
      <td>0.7184</td>
      <td>0.427</td>
      <td>0.437</td>
      <td>0.136</td>
      <td>91</td>
    </tr>
    <tr>
      <th>491</th>
      <td>The New York Times</td>
      <td>How to stop eating sugar https://t.co/PdGvwhkpaH</td>
      <td>Mon Mar 12 09:00:06 +0000 2018</td>
      <td>-0.2960</td>
      <td>0.000</td>
      <td>0.694</td>
      <td>0.306</td>
      <td>92</td>
    </tr>
    <tr>
      <th>492</th>
      <td>The New York Times</td>
      <td>RT @nytimesworld: He wears tennis shoes and ri...</td>
      <td>Mon Mar 12 08:44:04 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>93</td>
    </tr>
    <tr>
      <th>493</th>
      <td>The New York Times</td>
      <td>In the early days of the crackdown, at least 1...</td>
      <td>Mon Mar 12 08:31:03 +0000 2018</td>
      <td>-0.8316</td>
      <td>0.000</td>
      <td>0.719</td>
      <td>0.281</td>
      <td>94</td>
    </tr>
    <tr>
      <th>494</th>
      <td>The New York Times</td>
      <td>RT @nytimesworld: British counterterrorism pol...</td>
      <td>Mon Mar 12 08:26:05 +0000 2018</td>
      <td>-0.5267</td>
      <td>0.000</td>
      <td>0.841</td>
      <td>0.159</td>
      <td>95</td>
    </tr>
    <tr>
      <th>495</th>
      <td>The New York Times</td>
      <td>Low-effort, high-reward cooking for when you'r...</td>
      <td>Mon Mar 12 08:11:27 +0000 2018</td>
      <td>-0.3612</td>
      <td>0.000</td>
      <td>0.800</td>
      <td>0.200</td>
      <td>96</td>
    </tr>
    <tr>
      <th>496</th>
      <td>The New York Times</td>
      <td>Today, forgive yourself for putting off that o...</td>
      <td>Mon Mar 12 07:52:51 +0000 2018</td>
      <td>0.2732</td>
      <td>0.110</td>
      <td>0.890</td>
      <td>0.000</td>
      <td>97</td>
    </tr>
    <tr>
      <th>497</th>
      <td>The New York Times</td>
      <td>News Analysis: A Sudden Promotion Raises Quest...</td>
      <td>Mon Mar 12 07:34:57 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>98</td>
    </tr>
    <tr>
      <th>498</th>
      <td>The New York Times</td>
      <td>The phone booth is back. Kind of. https://t.co...</td>
      <td>Mon Mar 12 07:20:42 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>99</td>
    </tr>
    <tr>
      <th>499</th>
      <td>The New York Times</td>
      <td>It’s Tax Time! Here’s What to Know This Year h...</td>
      <td>Mon Mar 12 07:02:43 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 8 columns</p>
</div>




```python
# Set media source as index
media_index = media_tweets.set_index(['Media Source'])

# Subset new data for each media source
bbc_sent = media_index.loc['BBC News (UK)',['Compound']]
cbs_sent = media_index.loc['CBS',['Compound']]
cnn_sent = media_index.loc['CNN',['Compound']]
fox_sent = media_index.loc['Fox News',['Compound']]
nyt_sent = media_index.loc['The New York Times',['Compound']]
```

# Sentiment Analysis of Media Tweets


```python
# Set axes
tweets_ago = np.arange(1, 101)
bbc = bbc_sent['Compound']
cbs = cbs_sent['Compound']
cnn = cnn_sent['Compound']
fox = fox_sent['Compound']
nyt = nyt_sent['Compound']

# Set figure size
plt.figure(figsize=(10, 7))

# Plot
bbc_plot = plt.scatter(tweets_ago, bbc, facecolors='b', edgecolors='black', s=150, alpha=0.8, label="@BBCNews")
cbs_plot = plt.scatter(tweets_ago, cbs, facecolors='g', edgecolors='black', s=150, alpha=0.8, label="@CBS")
cnn_plot = plt.scatter(tweets_ago, cnn, facecolors='r', edgecolors='black', s=150, alpha=0.8, label="@CNN")
fox_plot = plt.scatter(tweets_ago, fox, facecolors='gold', edgecolors='black', s=150, alpha=0.8, label="@FoxNews")
nyt_plot = plt.scatter(tweets_ago, nyt, facecolors='darkslategray', edgecolors='black', s=150, alpha=0.8, label="@nytimes")

# Set limits
plt.xlim(105, -5)
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
plt.title(f'Sentiment Analysis of Media Tweets ({today})', fontsize=16)
plt.xlabel('Tweets Ago', fontsize=14)
plt.ylabel('Tweet Polarity', fontsize=14)

# Legend
lgd = plt.legend(handles=[bbc_plot, cbs_plot, cnn_plot, fox_plot, nyt_plot], 
                 title='Media Sources', loc=1, bbox_to_anchor=(1.2, 1))

# Format datetime for saving image
today = dt.now().strftime('%Y%m%d')

# Save image and show
plt.savefig(f'images/{today}_sent_media_tweets.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
```


![png](/images/20180312_sent_media_tweets.png)


# Overall Media Sentiment Based on Twitter



```python
# Group by media source
media_group = media_tweets.groupby(['Media Source'])

# Grab mean of compound sentiment values
avg_comp = media_group['Compound'].mean()

# Convert to dataframe
media_sent = pd.DataFrame({'Average Compound': avg_comp})

# Reset index
media_sent = media_sent.reset_index()
media_sent
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Media Source</th>
      <th>Average Compound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC News (UK)</td>
      <td>-0.183014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBS</td>
      <td>0.289191</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN</td>
      <td>-0.082892</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fox News</td>
      <td>0.051733</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The New York Times</td>
      <td>-0.049292</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Set axes
x_axis = np.arange(len(media_sent['Media Source']))
compound_sent = media_sent['Average Compound']

# Set figure size
plt.figure(figsize=(10, 7))

# color=tum_vol_chg['Positive?'].map({True: 'r', False: 'g'})
# Set bar colors
colors = {"BBC News (UK)": "b", "CBS": "g", "CNN": "r", 
          "Fox News": "gold", "The New York Times": "darkslategray"}


# Plot bars
sent_bar = plt.bar(x_axis, compound_sent, color=media_sent['Media Source'].map(colors), 
                  edgecolor='black', alpha=1, align="edge")

# Place tick locations and label
tick_locations = [value+0.4 for value in x_axis]
plt.xticks(tick_locations, ["BBC News (UK)", "CBS", "CNN", "Fox News", "The New York Times"])

# Set the x and y limits
#plt.xlim(-0.25, len(x_axis))
#plt.ylim(-25, max(drucompound_sent))

# Draw horizontal line
plt.axhline(y=0, xmin=0, xmax=1, linestyle='-', color='black')

# Get datetime
today = dt.now().strftime('%m/%d/%y')

# Labels
plt.title(f'Overall Media Sentiment Based on Twitter ({today})', fontsize=16)
plt.xlabel('Media Sources', fontsize=14)
plt.ylabel('Tweet Polarity', fontsize=14)

# Insert grid lines and set behind plot elements
# ax = plt.gca()
# ax.grid(linestyle='--')
# ax.grid(color='black', alpha=0.7)
# ax.set_axisbelow(True)

# Annotate bars
perc_tum_vol_chg = [f'{chg:.2f}' for chg in compound_sent]
for i, rect in enumerate(sent_bar):
    height = 0
    plt.text(rect.get_x() + rect.get_width()/2.0, height, perc_tum_vol_chg[i], 
             color='black', fontsize='13', ha='center', va='bottom')

# Format datetime for saving image
today = dt.now().strftime('%Y%m%d')    

# Save and show
plt.savefig(f'images/{today}_sent_media_overall.png')
plt.show()
```


![png](/images/20180312_sent_media_overall.png)

