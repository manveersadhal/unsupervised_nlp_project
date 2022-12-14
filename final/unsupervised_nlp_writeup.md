# r/WallStreetBets Unsupervised Topic Modeling

Manveer Sadhal

## Abstract
The goal of this project was to use unsupervised learning models and natural language processing to uncover the most prominent topics discussed on Reddit's WallStreetBets subreddit.

22,500 submissions of suitable length were retrieved from January 2019 to November 2021, using the first week of each month as a sample.

Raw text was processed prior to modeling, primarily using SciKit-Learn. After count vectorization or TF-IDF vectorization, dimensionality reduction and topic modeling was performed using latent semantic analysis (LSA), non-negative matrix factorization (NMF), and latent Dirichlet allocation (LDA) models.

The LSA and NMF models were packaged into an interactive app to allow for live feedback regarding hyperparameter changes as well as to investigate different time periods.

## Design
The [WallStreetBets subreddit](https://www.reddit.com/r/wallstreetbets/) has soared in popularity in 2020-2021 and has been covered in news media for short squeezes aimed against large financial institutions holding short positions. Trends in topics were evaluated from January 2019 to November 2021.

One application of this unsupervised learning model could be to use the topics identified as features in a supervised learning model aimed at predicting stock market movements.

## Data
22,500 submissions were retrieved from the WallStreetBets subreddit from January 2019 to November 2021. Only submissions categorized as "deep dive" and "discussion" were included. Any remaining submissions with a text body under 100 characters were also excluded.

Data was retrieved using the PushShift API. The retrieval and filtering of submissions was combined into one function to allow for additional data to be added in the future.

## Algorithms
### Text Preprocessing
- Removing special characters:
    - Newline, ampersands, greater than, less than, zero-width space characters
- Removing links
- Removing punctuation
- Removing capitalization
- Removing digits
- Removing extra spaces
- Replacing popular stock symbols with company names
- Removing stop words
    - Standard SciKit-Learn English stop words with 162 additional stop words added.

### Model Selection
The LSA model was used as a baseline, using both count and TF-IDF vectorization. LDA and NMF models were also evaluated.

LSA and NMF models both performed well in identifying topics based on the processed text. NMF with TF-IDF vectorization ultimately was chosen on the basis that the the topics identified through analysis of the most prominent tokens were the most coherent, with minimal overlap.

### Model Tuning and Evaluation
The number of topics to identify as well as minimum and maximum document frequencies were varied and the generated topics observed. Since the model was deployed in an app to evaluate numerous time periods, the same process was repeated for randomly selected date ranges in the data set. The maximum number of features defaults to approximately 75% of the number of documents being evaluated.

### Interactive App
The NMF and LSA models were deployed in an app to allow for the date range to be dynamically selected and hyperparameters tuned, with resulting tokens for topics identified displayed.

The app also features the ability to compare word clouds from two months side-by-side for a high level comparison of the discussions taking place.

## Tools
- Data Acquisition
    - PushShift API (using PSAW library) to retrieve submissions
- Text Preprocessing
    - SciKit-Learn
    - Contractions
    - String
    - RegEx
    - Pandas
    - NumPy
- Modeling
    - SciKit-Learn
    - pyLDAvis
    - Gensim
- Visualization
    - Matplotlib
    - WordCloud
- Production
    - Streamlit
    - SQLAlchemy

## Communication
A summary of the modeling process and results will be communicated during a 5-minute slide presentation.
