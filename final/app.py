import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import datetime as dt
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# import pyLDAvis
# import pyLDAvis.sklearn


def display_wordcloud(text):
    wc_text = ' '.join(text)
    wordcloud = WordCloud(stopwords = stop_words.union('s'), random_state=137, background_color='white',
                      collocations=True).generate(wc_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    st.pyplot(fig)

def display_topics(model, feature_names, no_top_words):
    output_list = []
    for ix, topic in enumerate(model.components_, 1):
        output_list.append(f"Topic {ix}: ")
        output_list.append(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        output_list.append(' <br>')
    return ''.join(output_list)


my_stop_words = ['like', 'likes', 'just', 'shares', 'guys', 'click', 'x200b', 'going', 'went', 'stock', 'stocks', 'market', 'know', 'link', 'think',\
                'want', 'ape', 'apes', 'company', 'companies', 'earnings', 'short', 'shit', 'fucking', 'million', 'billion', 'revenue', 'let', 'got',\
                 'fuck', 'fucking', 'fucked', 'retard', 'retards', 'retarded', 'people', 'mods', 'sub', 'post', 'did', 'ass', 'make', 'maybe',\
                 'tell', 'getting', 'autists', 'autistic', 'tendies', 'listen', 'remember', 'talking', 'buy', 'hold', 'sell', 'good', 'new', 'time', 'today',\
                'week', 'use', 'invest', 'bought', 'need', 'day', 'really', 'right', 'high', 'low', 'big', 'need', 'right', 'money', 'cash',\
                'lot', 'need', 'does', 'day', 'right', 'way', 'price', 'trade', 'trading', 'trades', 'calls', 'option', 'options', 'short',\
                 'shorts', 'shorted', 'long', 'share', 'thoughts', 'missed', 'come', 'coming', 'doing', 'thing', 'trying', 'say', 'feel', 'goes', 'actually', 'porn',\
                 'meme', 'squeeze', 'holding', 'buying', 'wsb', 'days', 'lose', 'look', 'looking', 'knows', 'believe', 'possibly', 'said', 'nbsp',\
                'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'thinking',\
                'month', 'year', 'years', 'probably', 'better', 'tl', 'dr', 'likely', 'sure', 'great', 'months', 'things', 'means', 'started', 'business',\
                 'industry', 'based', 'pretty', 'total', 'best', 'real', 'currently', 'play', 'kind', 'little']
stop_words = ENGLISH_STOP_WORDS.union(my_stop_words)

#create db connection
engine = create_engine('sqlite:///data.db')

#get user input for dates to display
st.title('WallStreetBets Topic Comparison')
st.write("Word Cloud Comparison")
date_ranges = [date.date() for date in pd.date_range('2019-01-01', periods=35, freq="MS")]

col1, col2 = st.beta_columns(2)

with col1:
    left_date = st.selectbox('First Date', date_ranges)

with col2:
    right_date = st.selectbox('Second Date', date_ranges)


# if date1 != date2, query db and create wordclouds

table = 'submissions'
start_date = left_date
end_date = left_date + dt.timedelta(days=7)
sql = f"""
    SELECT docs_clean
    FROM {table}
    WHERE date BETWEEN '{start_date}' AND '{end_date}';
    """

left_df = pd.read_sql(sql, engine)

table = 'submissions'
start_date = right_date
end_date = right_date + dt.timedelta(days=7)
sql = f"""
    SELECT docs_clean
    FROM {table}
    WHERE date BETWEEN '{start_date}' AND '{end_date}';
    """

right_df = pd.read_sql(sql, engine)

with col1:
    display_wordcloud(left_df['docs_clean'])
with col2:
    display_wordcloud(right_df['docs_clean'])

# min_df = 0.05
# max_df = 0.8
# max_features = len(left_df['docs_clean']) // 2
# num_topics = 5
# words_to_display = 7

st.sidebar.title("LSA Topic Modeling")
topic_model_date = st.sidebar.selectbox('Month to Topic Model', date_ranges)

table = 'submissions'
start_date = topic_model_date
end_date = topic_model_date + dt.timedelta(days=7)
sql = f"""
    SELECT docs_clean
    FROM {table}
    WHERE date BETWEEN '{start_date}' AND '{end_date}';
    """

model_df = pd.read_sql(sql, engine)

min_df = st.sidebar.number_input('Minimum Document Frequency', min_value=0.0, max_value=1.0, value=0.05)
max_df = st.sidebar.number_input("Maximum Document Frequency", min_value=0.0, max_value=1.0, value=0.8)
max_features = st.sidebar.number_input("Maximum Features", min_value=1, max_value=int(len(model_df['docs_clean'])), value=int(len(model_df['docs_clean']) // 2))
num_topics = st.sidebar.number_input("Number of Topics", min_value=1, max_value=30, value=5)
words_to_display = st.sidebar.number_input("Words per topic to display", min_value=3, max_value=10, value=5)


tfidf = TfidfVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df, max_features=max_features)
df_fit = tfidf.fit_transform(model_df['docs_clean'])
tfidf_df = pd.DataFrame(df_fit.toarray(), columns=tfidf.get_feature_names_out())
lsa = TruncatedSVD(num_topics)
doc_topic = lsa.fit_transform(tfidf_df)

st.write("Model Results:")
st.markdown(display_topics(lsa, tfidf.get_feature_names_out(), words_to_display), unsafe_allow_html=True)

# tf_vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.9, max_features=1000)
# dtm_tf = tf_vectorizer.fit_transform(left_df['docs_clean'])
# lda_tf = LatentDirichletAllocation(n_components=5, random_state=137)
# lda_tf.fit(dtm_tf)
# pyLDA_prepared = pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)
# st.markdown(pyLDAvis.prepared_data_to_html(pyLDA_prepared, template_type='simple'), unsafe_allow_html=True)
