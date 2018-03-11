
# Twitter News Sentiment Analysis
##### Observable Trends:
###### 1.
###### 2.
###### 3.


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
# today = dt.now().strftime('%Y%m%d') 

# Save to csv
media_tweets.to_csv('tweet_data/news_sentiment.csv', encoding='utf-8', index=False)
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
      <td>Regulator to quiz water firms over shortages h...</td>
      <td>Sun Mar 11 04:33:27 +0000 2018</td>
      <td>-0.1531</td>
      <td>0.000</td>
      <td>0.814</td>
      <td>0.186</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BBC News (UK)</td>
      <td>Winter Paralympics: GB skiers win super-G silv...</td>
      <td>Sun Mar 11 01:53:45 +0000 2018</td>
      <td>0.5859</td>
      <td>0.297</td>
      <td>0.703</td>
      <td>0.000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BBC News (UK)</td>
      <td>Man restrained by police in Lewisham dies http...</td>
      <td>Sun Mar 11 00:16:38 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BBC News (UK)</td>
      <td>Anti-Islamic letters probed by terror police h...</td>
      <td>Sat Mar 10 23:19:34 +0000 2018</td>
      <td>-0.5267</td>
      <td>0.000</td>
      <td>0.638</td>
      <td>0.362</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BBC News (UK)</td>
      <td>Russian spy: Traces of nerve agent 'found at Z...</td>
      <td>Sat Mar 10 22:04:35 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BBC News (UK)</td>
      <td>Totnes landslide damages cars and shuts road h...</td>
      <td>Sat Mar 10 19:45:36 +0000 2018</td>
      <td>-0.4404</td>
      <td>0.000</td>
      <td>0.707</td>
      <td>0.293</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BBC News (UK)</td>
      <td>Mother's Day protest for Nazanin Zaghari-Ratcl...</td>
      <td>Sat Mar 10 19:45:36 +0000 2018</td>
      <td>-0.2500</td>
      <td>0.000</td>
      <td>0.750</td>
      <td>0.250</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BBC News (UK)</td>
      <td>Student asked to leave Rush bar in Mansfield o...</td>
      <td>Sat Mar 10 19:45:35 +0000 2018</td>
      <td>-0.0516</td>
      <td>0.000</td>
      <td>0.893</td>
      <td>0.107</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BBC News (UK)</td>
      <td>Six Nations: Ireland win 2018 title after Engl...</td>
      <td>Sat Mar 10 18:44:16 +0000 2018</td>
      <td>0.2023</td>
      <td>0.226</td>
      <td>0.595</td>
      <td>0.179</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BBC News (UK)</td>
      <td>Six Nations: France 22-16 England https://t.co...</td>
      <td>Sat Mar 10 18:39:51 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BBC News (UK)</td>
      <td>RT @bbcrugbyunion: FT: France 22-16 England\n\...</td>
      <td>Sat Mar 10 18:33:24 +0000 2018</td>
      <td>0.1386</td>
      <td>0.106</td>
      <td>0.811</td>
      <td>0.083</td>
      <td>11</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCBreaking: More than 200 witnesses ident...</td>
      <td>Sat Mar 10 17:18:57 +0000 2018</td>
      <td>-0.5859</td>
      <td>0.000</td>
      <td>0.826</td>
      <td>0.174</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BBC News (UK)</td>
      <td>Six Nations: Ireland 28-8 Scotland https://t.c...</td>
      <td>Sat Mar 10 16:29:41 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>13</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BBC News (UK)</td>
      <td>Luke Atherton reunited with RAF 'hero who save...</td>
      <td>Sat Mar 10 16:16:26 +0000 2018</td>
      <td>0.7506</td>
      <td>0.416</td>
      <td>0.584</td>
      <td>0.000</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>BBC News (UK)</td>
      <td>Woman robbed on motorway hard shoulder https:/...</td>
      <td>Sat Mar 10 16:16:26 +0000 2018</td>
      <td>-0.1027</td>
      <td>0.000</td>
      <td>0.811</td>
      <td>0.189</td>
      <td>15</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BBC News (UK)</td>
      <td>Man shot in head during 'brutal' attack in Wes...</td>
      <td>Sat Mar 10 16:16:26 +0000 2018</td>
      <td>-0.4767</td>
      <td>0.000</td>
      <td>0.763</td>
      <td>0.237</td>
      <td>16</td>
    </tr>
    <tr>
      <th>16</th>
      <td>BBC News (UK)</td>
      <td>RT @BBCPolitics: To mark one year to go until ...</td>
      <td>Sat Mar 10 13:56:26 +0000 2018</td>
      <td>0.4019</td>
      <td>0.105</td>
      <td>0.895</td>
      <td>0.000</td>
      <td>17</td>
    </tr>
    <tr>
      <th>17</th>
      <td>BBC News (UK)</td>
      <td>RT @BBC: 🙌 1 year ago today the world became a...</td>
      <td>Sat Mar 10 13:49:23 +0000 2018</td>
      <td>0.4404</td>
      <td>0.209</td>
      <td>0.791</td>
      <td>0.000</td>
      <td>18</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BBC News (UK)</td>
      <td>Friend tells court Jackson would be 'last pers...</td>
      <td>Sat Mar 10 12:33:48 +0000 2018</td>
      <td>-0.3612</td>
      <td>0.179</td>
      <td>0.559</td>
      <td>0.263</td>
      <td>19</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BBC News (UK)</td>
      <td>Elizabeth Hurley's nephew stabbed in Wandswort...</td>
      <td>Sat Mar 10 11:28:29 +0000 2018</td>
      <td>-0.4404</td>
      <td>0.000</td>
      <td>0.674</td>
      <td>0.326</td>
      <td>20</td>
    </tr>
    <tr>
      <th>20</th>
      <td>BBC News (UK)</td>
      <td>Regulator to quiz water firms over shortages h...</td>
      <td>Sun Mar 11 04:33:27 +0000 2018</td>
      <td>-0.1531</td>
      <td>0.000</td>
      <td>0.814</td>
      <td>0.186</td>
      <td>21</td>
    </tr>
    <tr>
      <th>21</th>
      <td>BBC News (UK)</td>
      <td>Winter Paralympics: GB skiers win super-G silv...</td>
      <td>Sun Mar 11 01:53:45 +0000 2018</td>
      <td>0.5859</td>
      <td>0.297</td>
      <td>0.703</td>
      <td>0.000</td>
      <td>22</td>
    </tr>
    <tr>
      <th>22</th>
      <td>BBC News (UK)</td>
      <td>Man restrained by police in Lewisham dies http...</td>
      <td>Sun Mar 11 00:16:38 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>23</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BBC News (UK)</td>
      <td>Anti-Islamic letters probed by terror police h...</td>
      <td>Sat Mar 10 23:19:34 +0000 2018</td>
      <td>-0.5267</td>
      <td>0.000</td>
      <td>0.638</td>
      <td>0.362</td>
      <td>24</td>
    </tr>
    <tr>
      <th>24</th>
      <td>BBC News (UK)</td>
      <td>Russian spy: Traces of nerve agent 'found at Z...</td>
      <td>Sat Mar 10 22:04:35 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BBC News (UK)</td>
      <td>Totnes landslide damages cars and shuts road h...</td>
      <td>Sat Mar 10 19:45:36 +0000 2018</td>
      <td>-0.4404</td>
      <td>0.000</td>
      <td>0.707</td>
      <td>0.293</td>
      <td>26</td>
    </tr>
    <tr>
      <th>26</th>
      <td>BBC News (UK)</td>
      <td>Mother's Day protest for Nazanin Zaghari-Ratcl...</td>
      <td>Sat Mar 10 19:45:36 +0000 2018</td>
      <td>-0.2500</td>
      <td>0.000</td>
      <td>0.750</td>
      <td>0.250</td>
      <td>27</td>
    </tr>
    <tr>
      <th>27</th>
      <td>BBC News (UK)</td>
      <td>Student asked to leave Rush bar in Mansfield o...</td>
      <td>Sat Mar 10 19:45:35 +0000 2018</td>
      <td>-0.0516</td>
      <td>0.000</td>
      <td>0.893</td>
      <td>0.107</td>
      <td>28</td>
    </tr>
    <tr>
      <th>28</th>
      <td>BBC News (UK)</td>
      <td>Six Nations: Ireland win 2018 title after Engl...</td>
      <td>Sat Mar 10 18:44:16 +0000 2018</td>
      <td>0.2023</td>
      <td>0.226</td>
      <td>0.595</td>
      <td>0.179</td>
      <td>29</td>
    </tr>
    <tr>
      <th>29</th>
      <td>BBC News (UK)</td>
      <td>Six Nations: France 22-16 England https://t.co...</td>
      <td>Sat Mar 10 18:39:51 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
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
      <td>Modern Love wants to hear about how college st...</td>
      <td>Sat Mar 10 17:30:08 +0000 2018</td>
      <td>0.8074</td>
      <td>0.390</td>
      <td>0.610</td>
      <td>0.000</td>
      <td>71</td>
    </tr>
    <tr>
      <th>471</th>
      <td>The New York Times</td>
      <td>Mired in poverty not long ago, Ghana’s economi...</td>
      <td>Sat Mar 10 17:15:02 +0000 2018</td>
      <td>-0.1779</td>
      <td>0.131</td>
      <td>0.704</td>
      <td>0.166</td>
      <td>72</td>
    </tr>
    <tr>
      <th>472</th>
      <td>The New York Times</td>
      <td>Donald Trump’s victory shook him. Badly. And s...</td>
      <td>Sat Mar 10 17:00:03 +0000 2018</td>
      <td>-0.5423</td>
      <td>0.000</td>
      <td>0.791</td>
      <td>0.209</td>
      <td>73</td>
    </tr>
    <tr>
      <th>473</th>
      <td>The New York Times</td>
      <td>The vandals attacked a statue of Gandhi in the...</td>
      <td>Sat Mar 10 16:45:07 +0000 2018</td>
      <td>-0.4588</td>
      <td>0.000</td>
      <td>0.842</td>
      <td>0.158</td>
      <td>74</td>
    </tr>
    <tr>
      <th>474</th>
      <td>The New York Times</td>
      <td>RT @samdolnick: I went to Ohio to profile a ma...</td>
      <td>Sat Mar 10 16:30:08 +0000 2018</td>
      <td>-0.0258</td>
      <td>0.077</td>
      <td>0.843</td>
      <td>0.080</td>
      <td>75</td>
    </tr>
    <tr>
      <th>475</th>
      <td>The New York Times</td>
      <td>Migrants have created some of the works themse...</td>
      <td>Sat Mar 10 16:15:06 +0000 2018</td>
      <td>0.3612</td>
      <td>0.171</td>
      <td>0.829</td>
      <td>0.000</td>
      <td>76</td>
    </tr>
    <tr>
      <th>476</th>
      <td>The New York Times</td>
      <td>Prime Minister Justin Trudeau of Canada has ap...</td>
      <td>Sat Mar 10 16:00:08 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>77</td>
    </tr>
    <tr>
      <th>477</th>
      <td>The New York Times</td>
      <td>RT @EricLiptonNYT: JUST POSTED: A NYT journey ...</td>
      <td>Sat Mar 10 15:46:12 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>78</td>
    </tr>
    <tr>
      <th>478</th>
      <td>The New York Times</td>
      <td>A traditional museum experience using video-ga...</td>
      <td>Sat Mar 10 15:45:05 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>79</td>
    </tr>
    <tr>
      <th>479</th>
      <td>The New York Times</td>
      <td>Here are some of the highlights of the past we...</td>
      <td>Sat Mar 10 15:30:08 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>80</td>
    </tr>
    <tr>
      <th>480</th>
      <td>The New York Times</td>
      <td>Serve it straight out of the oven, when it's a...</td>
      <td>Sat Mar 10 15:17:03 +0000 2018</td>
      <td>0.2263</td>
      <td>0.137</td>
      <td>0.863</td>
      <td>0.000</td>
      <td>81</td>
    </tr>
    <tr>
      <th>481</th>
      <td>The New York Times</td>
      <td>Adam Rippon: "There are other athletes who are...</td>
      <td>Sat Mar 10 15:02:01 +0000 2018</td>
      <td>-0.2529</td>
      <td>0.087</td>
      <td>0.785</td>
      <td>0.128</td>
      <td>82</td>
    </tr>
    <tr>
      <th>482</th>
      <td>The New York Times</td>
      <td>Unsubscribing from IRL junk mail is almost as ...</td>
      <td>Sat Mar 10 14:41:06 +0000 2018</td>
      <td>0.0314</td>
      <td>0.145</td>
      <td>0.717</td>
      <td>0.138</td>
      <td>83</td>
    </tr>
    <tr>
      <th>483</th>
      <td>The New York Times</td>
      <td>The dispute over the Trump hotel in Panama Cit...</td>
      <td>Sat Mar 10 14:21:03 +0000 2018</td>
      <td>-0.4019</td>
      <td>0.000</td>
      <td>0.816</td>
      <td>0.184</td>
      <td>84</td>
    </tr>
    <tr>
      <th>484</th>
      <td>The New York Times</td>
      <td>RT @nytimes: Kevin Zeich had 3 and a half year...</td>
      <td>Sat Mar 10 14:05:25 +0000 2018</td>
      <td>-0.4588</td>
      <td>0.044</td>
      <td>0.845</td>
      <td>0.111</td>
      <td>85</td>
    </tr>
    <tr>
      <th>485</th>
      <td>The New York Times</td>
      <td>John Kelly thwarted a plan by the EPA chief Sc...</td>
      <td>Sat Mar 10 14:02:05 +0000 2018</td>
      <td>0.1280</td>
      <td>0.086</td>
      <td>0.856</td>
      <td>0.059</td>
      <td>86</td>
    </tr>
    <tr>
      <th>486</th>
      <td>The New York Times</td>
      <td>"Identity politics, religion, architecture, ch...</td>
      <td>Sat Mar 10 13:41:02 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>87</td>
    </tr>
    <tr>
      <th>487</th>
      <td>The New York Times</td>
      <td>How one child’s sickle cell mutation helped pr...</td>
      <td>Sat Mar 10 13:21:04 +0000 2018</td>
      <td>0.3818</td>
      <td>0.178</td>
      <td>0.822</td>
      <td>0.000</td>
      <td>88</td>
    </tr>
    <tr>
      <th>488</th>
      <td>The New York Times</td>
      <td>The excavation of Rome’s newest subway line ha...</td>
      <td>Sat Mar 10 13:02:03 +0000 2018</td>
      <td>0.6486</td>
      <td>0.249</td>
      <td>0.751</td>
      <td>0.000</td>
      <td>89</td>
    </tr>
    <tr>
      <th>489</th>
      <td>The New York Times</td>
      <td>Al Sharpton, Reconsidered https://t.co/Sn5DSYqVG7</td>
      <td>Sat Mar 10 12:50:55 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>90</td>
    </tr>
    <tr>
      <th>490</th>
      <td>The New York Times</td>
      <td>"It's always good news when you find new pengu...</td>
      <td>Sat Mar 10 12:41:06 +0000 2018</td>
      <td>0.4404</td>
      <td>0.244</td>
      <td>0.756</td>
      <td>0.000</td>
      <td>91</td>
    </tr>
    <tr>
      <th>491</th>
      <td>The New York Times</td>
      <td>RT @NYTHealth: Stop obsessing about your own h...</td>
      <td>Sat Mar 10 12:21:02 +0000 2018</td>
      <td>-0.8720</td>
      <td>0.000</td>
      <td>0.623</td>
      <td>0.377</td>
      <td>92</td>
    </tr>
    <tr>
      <th>492</th>
      <td>The New York Times</td>
      <td>In an effort to better understand the top chie...</td>
      <td>Sat Mar 10 12:02:04 +0000 2018</td>
      <td>0.5719</td>
      <td>0.190</td>
      <td>0.810</td>
      <td>0.000</td>
      <td>93</td>
    </tr>
    <tr>
      <th>493</th>
      <td>The New York Times</td>
      <td>RT @NYTNational: Megachurches around the count...</td>
      <td>Sat Mar 10 11:38:51 +0000 2018</td>
      <td>0.5719</td>
      <td>0.156</td>
      <td>0.844</td>
      <td>0.000</td>
      <td>94</td>
    </tr>
    <tr>
      <th>494</th>
      <td>The New York Times</td>
      <td>Fiction: A Mother and Daughter and a Town Made...</td>
      <td>Sat Mar 10 11:21:51 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>95</td>
    </tr>
    <tr>
      <th>495</th>
      <td>The New York Times</td>
      <td>After Demanding Local Control, DeVos Finds Tha...</td>
      <td>Sat Mar 10 11:18:23 +0000 2018</td>
      <td>-0.2263</td>
      <td>0.000</td>
      <td>0.853</td>
      <td>0.147</td>
      <td>96</td>
    </tr>
    <tr>
      <th>496</th>
      <td>The New York Times</td>
      <td>Trump Rules: In Decline, Offshore Drillers Fin...</td>
      <td>Sat Mar 10 11:14:39 +0000 2018</td>
      <td>0.5994</td>
      <td>0.245</td>
      <td>0.755</td>
      <td>0.000</td>
      <td>97</td>
    </tr>
    <tr>
      <th>497</th>
      <td>The New York Times</td>
      <td>In Britain’s Playgrounds, ‘Bringing in Risk’ t...</td>
      <td>Sat Mar 10 11:10:59 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>98</td>
    </tr>
    <tr>
      <th>498</th>
      <td>The New York Times</td>
      <td>In one of China’s biggest cities, the women-on...</td>
      <td>Sat Mar 10 10:56:12 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>99</td>
    </tr>
    <tr>
      <th>499</th>
      <td>The New York Times</td>
      <td>RT @nytpolitics: Stephen K. Bannon's next proj...</td>
      <td>Sat Mar 10 10:38:04 +0000 2018</td>
      <td>0.2960</td>
      <td>0.121</td>
      <td>0.879</td>
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

# Save image and show
plt.savefig('images/sent_media_tweets.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
```


![png](images/sent_media_tweets.png)


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
      <td>-0.124556</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBS</td>
      <td>0.314749</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN</td>
      <td>0.006238</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fox News</td>
      <td>0.201252</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The New York Times</td>
      <td>0.039298</td>
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

# Save and show
plt.savefig('images/sent_media_overall.png')
plt.show()
```


![png](images/sent_media_overall.png)

