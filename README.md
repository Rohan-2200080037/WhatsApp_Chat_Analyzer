# 🚀 WhatsApp Chat Analyzer Advanced

A powerful Streamlit-based web app that performs deep analysis of WhatsApp chat exports with advanced features like sentiment analysis, chat streaks, emoji tracking, word clouds, multilingual detection, and more.

## 🔧 Features

### 📊 General Chat Statistics
- Total messages, words, links, and media shared
- Message type distribution
- User-wise message and word contributions

### 😊 Sentiment Analysis
- Polarity, Subjectivity (TextBlob)
- VADER Sentiment (Compound Score)
- Sentiment over time visualizations
- Sentiment scatter plots (Polarity vs Subjectivity)

### 🔥 Engagement Metrics
- Average user response time
- Longest chat streaks per user
- Conversation starters (30+ min gaps)

### 🕒 Activity Patterns
- Activity heatmap (hour vs weekday)
- Hour-wise message frequency per user
- Message length distribution

### 😄 Emoji Analysis
- Top emojis overall and per user

### 🌐 Language Analysis
- Detection of primary languages
- Telugu transliterated language identification
- Language usage distribution

### 🧠 Keyword & Word Cloud
- Word cloud of common words (excluding stopwords)
- Most frequent words

### 🔗 Shared Content
- Most frequently shared URLs and domains

### 📁 Export Capabilities
- Export full dataset as CSV
- Ready for PDF integration (with saved plots)

##📦 Technologies Used

Python, Streamlit
pandas, numpy, seaborn, plotly, matplotlib
wordcloud, emoji, TextBlob, VADER, langdetect
Detoxify, scikit-learn (LDA support)


