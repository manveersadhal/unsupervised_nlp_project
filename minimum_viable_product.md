# Reddit WallStreetBets Unsupervised Learning
#### Manveer Sadhal
#### Nov 8, 2021
## Question
The question this project intends to answer is what the major topics being discussed on the WallStreetBets subreddit are for a given time period.

## Data
### Retrieval
The PushShift API was used to retrieve 10,000 of the most recent submissions in the WallStreetBets subreddit. Despite the volume of submissions retrieved, the date range only spans October 27, 2021 to November 5, 2021.

### Filtering
Many of the submissions did not contain any text for various reasons (memes, deleted and removed submissions). The submissions were filtered based on the flair tag (deep dive or discussion) as well as a minimum length of 100 characters in the body of the submission. Only 1,632 submissions remained after filtering.

### Text Preprocessing
The following preprocessing steps were taken:
- Removing special characters:
    - Newline, ampersands, greater than, less than, zero-width space characters
- Removing links
- Removing punctuation
- Removing capitalization
- Removing digits
- Removing extra spaces
- Removing stop words
    - Standard SciKit-Learn English stop words with additional stop words added.

## Vectorization
Count vectorization and TF-IDF vectorization were used in initial topic modeling using latent semantic analysis (LSA) as a baseline.

## Topic Modeling
### LSA
TF-IDF with LSA using five topics and an n-gram range from 1-3 highlights the following topics and top tokens:

- Topic  1: Matterport - 3D Media Company
    - stock, short, company, market, earnings, matterport, price

- Topic  2: Tilray - Cannabis Research Company
    - citadel, cannabis, tilray, tlry, jpm, banking, archegos

- Topic  3: Pfizer and COVID-19 Vaccines
    - fy2022, 120b, vaccine, boosters, 40b, pfizer, current guidance

- Topic  4: Matterport as a part of Metaverse
    - matterport, metaverse, scanning, spatial, scan, mttr, facebook

- Topic  5: Grab and Altimeter Growth Corp (AGC) Merger
    - grab, agc, merger, spac, shorts, float, ride

### Latent Dirichlet Allocation

Topic modeling was attempted using LDA (pyLDAvis library). The topics identified are less clear than those using LSA.

## Next Steps
- Additional preprocessing
    - Stemming or lemmatization
    - Adding more stop words
    - Combining stock symbols and company names into the same token
- Evaluation of non-negative matrix factorization (NMF) and possibly principal component analysis (PCA)
- Collection of data from other time periods to compare topics over time
- Identify most popular topics over time
- Sentiment analysis, possibly over time