import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import datetime as dt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

def display_wordcloud(text):
    wc_text = ' '.join(text)
    wordcloud = WordCloud(stopwords = stop_words.union('s'), random_state=137, background_color='white',
                      collocations=True).generate(wc_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


def display_topics(model, feature_names, no_top_words):
    output_list = []
    for ix, topic in enumerate(model.components_, 1):
        output_list.append(f"Topic {ix}: ")
        output_list.append(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        output_list.append(' <br>')
    return ''.join(output_list)


def create_lsa_tfidf(docs, num_topics, stop_words, min_df, max_df, max_features):
    tfidf = TfidfVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df, max_features=max_features)
    df_fit = tfidf.fit_transform(docs)
    tfidf_df = pd.DataFrame(df_fit.toarray(), columns=tfidf.get_feature_names_out())
    lsa = TruncatedSVD(num_topics)
    doc_topic = lsa.fit_transform(tfidf_df)
    return lsa, tfidf


def create_nmf_model(docs, num_topics, stop_words, min_df, max_df, max_features):
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features, min_df=min_df, max_df=max_df)
    doc_word = vectorizer.fit_transform(docs)
    nmf_model = NMF(num_topics)
    doc_topic = nmf_model.fit_transform(doc_word)
    return nmf_model, vectorizer


my_stop_words = ['like', 'likes', 'just', 'shares', 'guys', 'click', 'x200b', 'going', 'went', 'stock', 'stocks', 'market', 'know', 'link', 'think',\
                'want', 'ape', 'apes', 'company', 'companies', 'earnings', 'short', 'shit', 'fucking', 'million', 'billion', 'revenue', 'let', 'got',\
                 'fuck', 'fucking', 'fucked', 'retard', 'retards', 'retarded', 'people', 'mods', 'sub', 'post', 'did', 'ass', 'make', 'maybe',\
                 'tell', 'getting', 'autist', 'autists', 'autistic', 'tendies', 'listen', 'remember', 'talking', 'buy', 'hold', 'sell', 'good', 'new', 'time', 'today',\
                'week', 'use', 'invest', 'bought', 'need', 'day', 'really', 'right', 'high', 'low', 'big', 'need', 'right', 'money', 'cash',\
                'lot', 'need', 'does', 'day', 'right', 'way', 'price', 'trade', 'trading', 'trades', 'calls', 'option', 'options', 'short',\
                 'shorts', 'shorted', 'long', 'share', 'thoughts', 'missed', 'come', 'coming', 'doing', 'thing', 'trying', 'say', 'feel', 'goes', 'actually', 'porn',\
                 'meme', 'squeeze', 'holding', 'buying', 'wsb', 'days', 'lose', 'look', 'looking', 'knows', 'believe', 'possibly', 'said', 'nbsp',\
                'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'thinking',\
                'month', 'year', 'years', 'probably', 'better', 'tl', 'dr', 'likely', 'sure', 'great', 'months', 'things', 'means', 'started', 'business',\
                 'industry', 'based', 'pretty', 'total', 'best', 'real', 'currently', 'play', 'kind', 'little', 'yeah', 'nbsp', 'hello', 'hey', 'dd', 'tldr']
stop_words = ENGLISH_STOP_WORDS.union(my_stop_words)

#create db connection
engine = create_engine('sqlite:///data.db')
periods=35
date_ranges = [date.date() for date in pd.date_range('2019-01-01', periods=periods, freq="MS")]

st.sidebar.title("Menu")
display_option = st.sidebar.radio("Select Module:", options=['Word Clouds', 'Topic Modeling'])

if display_option == 'Word Clouds':
    #get user input for dates to display
    st.title('WallStreetBets Keywords')
    st.markdown("### Word Cloud Comparison")

    left_date_index = periods - 20
    right_date_index = periods - 1

    if st.button('Click Here to Randomly Select Two Months to Compare'):
        left_date_index = np.random.choice(periods - 1)
        right_date_index = np.random.choice(periods - 1)
        while left_date_index == right_date_index:
            right_date_index = np.random.choice(periods - 1)

    col1, col2 = st.columns(2)
    
    with col1:
        left_date = st.selectbox('First Month', date_ranges, format_func=lambda x: x.strftime('%B %Y'), index=left_date_index)

    with col2:
        right_date = st.selectbox('Second Month', date_ranges, format_func=lambda x: x.strftime('%B %Y'), index=right_date_index)

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

else:
    st.title("WallStreetBets Topic Modeling")
    st.markdown("#### Choose Month, Model Type, and Model Parameters Below")

    topic_model_date = st.selectbox('Month to Model', date_ranges, format_func=lambda x: x.strftime('%B %Y'), index=periods - 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("Type of Model", ['NMF', 'LSA'])    

    table = 'submissions'
    start_date = topic_model_date
    end_date = topic_model_date + dt.timedelta(days=7)
    sql = f"""
        SELECT docs_clean
        FROM {table}
        WHERE date BETWEEN '{start_date}' AND '{end_date}';
        """

    model_df = pd.read_sql(sql, engine)

    default_max_features = len(model_df['docs_clean']) // 4 * 3
    max_features_max = len(model_df['docs_clean'])
    default_num_topics = 7

    with col1:
        min_df = st.number_input('Minimum Document Frequency', min_value=0.0, max_value=1.0, value=0.0)
        max_df = st.number_input("Maximum Document Frequency", min_value=0.0, max_value=1.0, value=0.9)
    
    with col2:
        max_features = st.number_input("Maximum Features", format='%i', min_value=1, max_value=max_features_max, value=default_max_features, step=1)
        num_topics = st.number_input("Number of Topics", min_value=1, max_value=30, value=default_num_topics)
        words_to_display = st.number_input("Words per Topic to Display", min_value=3, max_value=10, value=5)

    if model_type == 'LSA':
        model, vectorizer = create_lsa_tfidf(docs=model_df['docs_clean'], num_topics=num_topics, stop_words=stop_words, min_df=min_df, max_df=max_df, max_features=max_features)
    else:
        model, vectorizer = create_nmf_model(docs=model_df['docs_clean'], num_topics=num_topics, stop_words=stop_words, min_df=min_df, max_df=max_df, max_features=max_features)
    
    st.markdown("### Model Results:")
    st.markdown(display_topics(model, vectorizer.get_feature_names_out(), words_to_display), unsafe_allow_html=True)
