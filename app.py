import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
from collections import Counter
from datetime import datetime
import emoji
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import langdetect
import io
from fpdf import FPDF, XPos, YPos
import tempfile
import os
from langdetect import detect, DetectorFactory,LangDetectException
DetectorFactory.seed = 0
import re
from urllib.parse import urlparse
from detoxify import Detoxify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import streamlit as st


st.set_page_config(page_title="WhatsApp Chat Analyzer Advanced", layout="wide")

vader = SentimentIntensityAnalyzer()

# -------------------------
# Helper Functions
# -------------------------

def preprocess(data):
    # WhatsApp message regex pattern (date, user, message)
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:am|pm))\s-\s([^:]+):\s(.*)'
    messages = re.findall(pattern, data)
    df = pd.DataFrame(messages, columns=['Datetime', 'User', 'Message'])
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y, %I:%M %p', errors='coerce')
    df = df.dropna().reset_index(drop=True)
    df = df[df['Message'] != '']
    df['IsMedia'] = df['Message'].str.contains('<Media omitted>', case=False)
    df['IsLink'] = df['Message'].str.contains('http', case=False)
    df['Word Count'] = df['Message'].apply(lambda x: len(str(x).split()))
    df['Char Count'] = df['Message'].apply(len)
    df['Emoji'] = df['Message'].apply(lambda x: [c for c in x if c in emoji.EMOJI_DATA])
    df['Mentions'] = df['Message'].apply(lambda x: re.findall(r'@\w+', x))
    df['Date'] = df['Datetime'].dt.date
    df['Hour'] = df['Datetime'].dt.hour
    df['Weekday'] = df['Datetime'].dt.day_name()
    df['Month'] = df['Datetime'].dt.to_period('M')
    return df

def get_sentiment_scores(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    vader_scores = vader.polarity_scores(text)
    compound = vader_scores['compound']
    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    return polarity, subjectivity, compound, label

def detect_language(text):
    try:
        if len(text.strip()) < 3:
            return "unknown"
        return langdetect.detect(text)
    except:
        return "unknown"

def calculate_response_times(df):
    df_sorted = df.sort_values('Datetime')
    response_times = []
    last_message_time = None
    last_user = None
    for _, row in df_sorted.iterrows():
        if last_message_time and row['User'] != last_user:
            diff = (row['Datetime'] - last_message_time).total_seconds() / 60
            if diff > 0:
                response_times.append(diff)
        last_message_time = row['Datetime']
        last_user = row['User']
    avg_response = np.mean(response_times) if response_times else None
    return avg_response, response_times

def calculate_chat_streaks(df):
    streaks = {}
    users = df['User'].unique()
    for user in users:
        user_dates = sorted(df[df['User'] == user]['Date'].unique())
        max_streak = 1
        current_streak = 1
        for i in range(1, len(user_dates)):
            if (user_dates[i] - user_dates[i-1]).days == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        streaks[user] = max_streak
    return streaks

def plot_activity_heatmap(df):
    heatmap_data = df.groupby(['Weekday', 'Hour']).size().reset_index(name='MessageCount')
    # Reorder weekdays to Monday-Sunday
    weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data['Weekday'] = pd.Categorical(heatmap_data['Weekday'], categories=weekdays_order, ordered=True)
    heatmap_pivot = heatmap_data.pivot(index='Hour', columns='Weekday', values='MessageCount').fillna(0)
    plt.figure(figsize=(12,6))
    sns.heatmap(heatmap_pivot, cmap='YlGnBu', linewidths=.5)
    plt.title('Message Activity Heatmap by Hour and Weekday')
    plt.ylabel('Hour of Day')
    plt.xlabel('Day of Week')
    st.pyplot(plt)

def emoji_usage_stats(df):
    all_emojis = []
    user_emojis = {}

    for user in df['User'].unique():
        user_msgs = df[df['User'] == user]['Message']
        emojis = [c for msg in user_msgs for c in msg if c in emoji.EMOJI_DATA]
        user_emojis[user] = Counter(emojis)
        all_emojis.extend(emojis)

    overall_counts = Counter(all_emojis)
    return overall_counts, user_emojis

def plot_message_length_distribution(df):
    plt.figure(figsize=(10,5))
    sns.histplot(df['Char Count'], bins=50, kde=True)
    plt.title("Message Length Distribution (Characters)")
    plt.xlabel("Message Length")
    plt.ylabel("Frequency")
    st.pyplot(plt)

def plot_active_hours_per_user(df):
    users = df['User'].unique()
    fig, axes = plt.subplots(len(users), 1, figsize=(12, 4 * len(users)), sharex=True)
    if len(users) == 1:
        axes = [axes]
    for ax, user in zip(axes, users):
        user_hours = df[df['User'] == user]['Hour'].value_counts().sort_index()
        ax.bar(user_hours.index, user_hours.values, color='mediumseagreen')
        ax.set_title(f"Active Hours for {user}")
        ax.set_ylabel("Messages")
        ax.set_xlabel("Hour of Day")
    plt.tight_layout()
    st.pyplot(plt)
from langdetect import detect, LangDetectException

from langdetect import detect, LangDetectException

def language_usage(df):
    def is_telugu_transliterated(text):
        telugu_keywords = [
            'naa', 'nee', 'naku', 'neeku', 'nenu', 'meeru', 'meeru', 'vadu', 'avadu', 'vachindi',
            'cheyyadam', 'choodu', 'ra', 'em', 'pelli', 'vachindi', 'ellu', 'unda', 'undi', 'ledu',
            'aithe', 'kani', 'dinni', 'manam', 'avunu', 'kaadu', 'kaani', 'kuda', 'chala', 'chusthunnanu',
            'chesthunnanu', 'mimmalni', 'meeku', 'mari', 'entha', 'tappu', 'adhi', 'maata', 'thappu',
            'pillalu', 'bala', 'baga', 'poyindi', 'poyanu', 'poye', 'okkati', 'pade', 'vaddu', 'pilla',
            'chinnadi', 'pedda', 'podam', 'raavu', 'raav', 'eesu', 'pitta', 'addam', 'mariyu', 'appudu',
            'cheppu', 'chey', 'tinu', 'tinnanu', 'vinanu', 'sare', 'santoshanga', 'telusu', 'raju',
            'vadini', 'chestha', 'ente', 'okkadu', 'thappadu', 'kaadhu', 'ayyina', 'pandaga', 'dodda',
            'chinna', 'vellipoyi', 'kadu', 'madhya', 'manchi', 'thakkuva', 'pattu', 'koti', 'kadha',
            'nikitha', 'unna', 'vunte', 'okka', 'kaani', 'kaavalante', 'okasari', 'aithe', 'kudirithe',
            'kotha', 'chakkani', 'meeru', 'meeku', 'edho', 'okkaadu', 'aadukunnadu', 'chaalu', 'adugutunnanu',
            'dinna', 'dariki', 'dhoriki', 'poyinadhi', 'modati', 'kalavani', 'kudire', 'choosthunnaru',
            'okka', 'padam', 'padina', 'padutunna', 'chudandi', 'emito', 'okkate', 'padaga', 'kalisi',
            'thodalu', 'dini', 'chidabothunnaru', 'adhi', 'anni', 'poyina', 'neethi', 'tappaka', 'paaru',
            'koothuru', 'puttindi', 'pakka', 'madi', 'mukhyam', 'sambandham', 'chala', 'nunchi', 'neellu',
            'navvu', 'paruvam', 'mattladadam', 'navvutunnaru', 'pelli', 'prema', 'saayam', 'kaasam', 'kavacham',
            'kothaga', 'nela', 'talliki', 'cheppi', 'thalli', 'talli', 'mama', 'mami', 'thammudu', 'cheyadam',
            'cheyyaledu', 'ra', 'gaa', 'madhuram', 'bommala', 'vidudala', 'vachindi', 'elanti', 'anta',
            'addam', 'bhandaru', 'dabbulu', 'thagulu', 'cheyyandi', 'nirdeshalu', 'kanipistunnayi', 'putti',
            'paaru', 'dhiniki', 'dina', 'pothundi', 'madhyalo', 'preminchadam', 'naku', 'nee', 'mimmalni',
            'pilusthunnanu', 'adugu', 'adigithe', 'chitram', 'padam', 'chuse', 'poyina', 'vinipistundi',
            'nirnayam', 'poyavu', 'teliyadu', 'matladatam', 'niluvani', 'niluvadam', 'poyindi', 'kalagadam',
            'chese', 'avakasam', 'lanti', 'madhya', 'mukhya', 'chudandi', 'mimmalni', 'pilavandi', 'thodu',
            'daani', 'chakkaga', 'kottaga', 'kaadu', 'kaani', 'chaala', 'chota', 'tinnanu', 'pinchadam',
            'padukoni', 'vesi', 'chusthunnaru', 'madhuram', 'goppatanam', 'kala', 'saayam', 'chusthunnanu',
            'ga', 'chala', 'cheppandi', 'chinna', 'gundello', 'pelli', 'thalli', 'pillalu', 'aduguta', 'vinadam'
        ]
        text_lower = text.lower()
        return any(word in text_lower for word in telugu_keywords)

    def safe_detect(text):
        try:
            if len(text.strip()) > 20:
                if is_telugu_transliterated(text):
                    return 'telugu'
                elif detect(text) == 'en':
                    return 'english'
        except LangDetectException:
            if is_telugu_transliterated(text):
                return 'telugu'
        return None  # Exclude anything that’s not telugu or english

    langs = df['Message'].dropna().apply(safe_detect)
    langs = langs[langs.notna()]  # Drop None entries
    return langs.value_counts(normalize=True) * 100

def frequent_urls_domains(df):
    url_pattern = r'(https?://\S+)'
    urls = []
    for msg in df['Message'].dropna():
        urls.extend(re.findall(url_pattern, msg))
    domains = [urlparse(url).netloc for url in urls]
    domain_counts = Counter(domains).most_common(10)
    return domain_counts

def conversation_starters(df):
    df = df.sort_values('Datetime')
    df['time_diff'] = df['Datetime'].diff().dt.total_seconds().fillna(0)
    df['starter'] = df['time_diff'] > 1800  # 30 minutes threshold
    starters_count = df[df['starter']].groupby('User').size().sort_values(ascending=False)
    return starters_count

def export_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# -------------------------
# App UI
# -------------------------

st.title("🚀 WhatsApp Chat Analyzer Advanced — All Features in One Page")

uploaded_file = st.file_uploader("Upload your WhatsApp chat export (.txt)", type=["txt"])

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8", errors='ignore')
    df = preprocess(raw_text)

    if df.empty:
        st.error("No valid messages found in uploaded file.")
    else:
        sentiment_results = df['Message'].apply(get_sentiment_scores)
        df[['Polarity', 'Subjectivity', 'VADER_Compound', 'Sentiment']] = pd.DataFrame(sentiment_results.tolist(), index=df.index)

        if len(df) > 1000:
            sample_langs = df['Message'].sample(1000, random_state=42).apply(detect_language)
            most_common_lang = sample_langs.mode()[0] if not sample_langs.mode().empty else "unknown"
            st.info(f"Most common language detected (sample): {most_common_lang}")
        else:
            df['Language'] = df['Message'].apply(detect_language)

        # Filters sidebar
        st.sidebar.header("Filters & Controls")
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        user_list = df['User'].unique()
        selected_users = st.sidebar.multiselect("Select Users", user_list, default=user_list.tolist())
        df = df[df['User'].isin(selected_users)]

        show_media = st.sidebar.checkbox("Include Media Messages", True)
        if not show_media:
            df = df[~df['IsMedia']]
        show_links = st.sidebar.checkbox("Include Link Messages", True)
        if not show_links:
            df = df[~df['IsLink']]

        search_kw = st.sidebar.text_input("Search Messages (keyword):").strip()
        if search_kw:
            df = df[df['Message'].str.contains(search_kw, case=False, na=False)]

        st.sidebar.markdown("---")

        # Overview
        st.header("📊 Overview & Basic Stats")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", len(df))
        col2.metric("Total Words", df['Word Count'].sum())
        col3.metric("Media Messages", df['IsMedia'].sum())
        col4.metric("Links Shared", df['IsLink'].sum())

        st.subheader("Message Type Distribution")
        type_counts = {
            "Text": len(df) - df['IsMedia'].sum() - df['IsLink'].sum(),
            "Media": df['IsMedia'].sum(),
            "Links": df['IsLink'].sum()
        }
        fig_type = px.pie(names=list(type_counts.keys()), values=list(type_counts.values()), title="Message Type Distribution")
        st.plotly_chart(fig_type, use_container_width=True)

        st.subheader("Mentions Summary")
        mentions = [m for sublist in df['Mentions'] for m in sublist]
        mention_counts = Counter(mentions)
        if mention_counts:
            mention_df = pd.DataFrame(mention_counts.items(), columns=["Mention", "Count"]).sort_values(by="Count", ascending=False)
            st.dataframe(mention_df.head(10))
        else:
            st.write("No mentions found.")

        # User Activity
        st.header("👥 User Activity & Stats")
        user_counts = df['User'].value_counts()
        fig_user = px.bar(user_counts, title="Messages per User")
        st.plotly_chart(fig_user, use_container_width=True)

        st.subheader("Words per User")
        words_per_user = df.groupby('User')['Word Count'].sum().sort_values(ascending=False)
        st.bar_chart(words_per_user)

        st.subheader("Average Response Time (Minutes)")
        avg_resp, resp_times = calculate_response_times(df)
        st.write(f"Average response time between users: {avg_resp:.2f} minutes" if avg_resp else "Insufficient data for response time calculation.")

        # Sentiment Analysis
        st.header("😊 Sentiment Analysis")
        sentiment_counts = df['Sentiment'].value_counts(normalize=True).mul(100).round(2)
        fig_sentiment = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values, title="Sentiment Distribution (%)")
        st.plotly_chart(fig_sentiment, use_container_width=True)

        st.subheader("Sentiment Over Time")
        sentiment_time = df.groupby('Date')['VADER_Compound'].mean()
        fig_sent_time = px.line(sentiment_time, labels={"Date":"Date", "VADER_Compound":"Avg VADER Compound Score"})
        st.plotly_chart(fig_sent_time, use_container_width=True)

        st.subheader("Polarity vs Subjectivity")
        fig_ps = px.scatter(df, x='Polarity', y='Subjectivity', color='Sentiment', title="Polarity vs Subjectivity")
        st.plotly_chart(fig_ps, use_container_width=True)

        # Chat Streaks
        st.header("🔥 Chat Streaks")
        st.write("Longest consecutive days each user sent messages:")
        st.table(pd.DataFrame(calculate_chat_streaks(df).items(), columns=['User', 'Longest Streak (days)']))

        # Activity Heatmap
        st.header("⏰ Activity Heatmap")
        plot_activity_heatmap(df)

        st.header("😄 Emoji Usage Stats")

        overall_counts, user_emojis = emoji_usage_stats(df)

        st.subheader("Top 10 Emojis Overall")
        overall_df = pd.DataFrame(overall_counts.most_common(10), columns=['Emoji', 'Count'])
        st.table(overall_df)

        st.subheader("Top 5 Emojis Per User")
        for user, counts in user_emojis.items():
            st.write(f"**{user}**")
            user_df = pd.DataFrame(counts.most_common(5), columns=['Emoji', 'Count'])
            st.table(user_df)

        st.header("✍️ Message Length Distribution")
        plot_message_length_distribution(df)

        st.header("⏳ Most Active Hours Per User")
        plot_active_hours_per_user(df)

        st.header("🌐 Language Detection & Usage")
        lang_dist = language_usage(df)
        st.bar_chart(lang_dist)

        st.header("🔗 Frequent URLs & Domains Shared")
        domains = frequent_urls_domains(df)
        st.table(domains)

        st.header("🗣️ Conversation Starters")
        starters = conversation_starters(df)
        st.table(starters)

        # Wordcloud
        st.header("🔤 Wordcloud of Most Common Words")
        all_text = " ".join(df['Message'].tolist()).lower()
        stopwords = set(STOPWORDS)
        wc = WordCloud(stopwords=stopwords, background_color='white', max_words=200, width=800, height=400)
        wc.generate(all_text)
        plt.figure(figsize=(15,7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Get most used words (top 10 for example)
        words = re.findall(r'\w+', all_text)
        word_counts = Counter(words)
        most_used_words = [w for w, c in word_counts.most_common(10)]

        # Prepare wordcloud figure for PDF (save figure)
        fig_wc = plt.figure(figsize=(15,7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.close(fig_wc)  # Close so it doesn't display twice

        # Example for language distribution figure
        fig_lang, ax = plt.subplots(figsize=(6,4))
        ax.bar(['English', 'Spanish'], [100, 50])
        ax.set_title("Languages Used")
        plt.close(fig_lang)  # Prevents figure from displaying twice in Streamlit


        # Export options
        st.header("📥 Export Data")
        csv_data = export_csv(df)
        st.download_button("Download CSV", data=csv_data, file_name="whatsapp_chat_data.csv", mime="text/csv")

