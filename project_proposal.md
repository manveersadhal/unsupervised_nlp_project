# Reddit WallStreetBets Unsupervised Learning Proposal
#### Manveer Sadhal
#### Nov 3, 2021
## Question
Reddit's WallStreetBets subreddit has been widely discussed in news media from 2020-2021 for performing in-depth analyses of the stock market and loosely organizing [short squeezes](https://www.investopedia.com/terms/s/shortsqueeze.asp) as a protest against the large financial institutions holding the short positions. Some Redditors have made a fortune almost overnight by taking part.

The goal of this project is to use unsupervised learning to evaluate trends in topics discussed as well as the sentiment of the discussion.
## Data Description
Reddit's WallStreetBets subreddit. Retrieve 1,000 most recent posts and their associated comments using an API.

A sample would consist of a single post. Each comment on the post would be a separate sample.

## Tools
- Data Acquisition
    - PushShift API to retrieve posts from the subreddit
- Data Cleaning
    - Pandas
    - NumPy
- Text Preprocessing
    - NLTK
    - spaCy
    - SciKit-Learn
- Modeling
    - SciKit-Learn
- Visualization
    - Matplotlib
    - Seaborn
    - Plotly

## MVP Goal
Retrieve text data from Reddit using the API. Process the text and count vectorize or TF-IDF vectorize it. Generate preliminary topic models.