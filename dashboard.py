# theme_dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# Load cleaned dataframe
@st.cache_data
def load_dataframe():
    # Make sure 'processed_reviews.csv' exists in your working directory
    return pd.read_csv("processed_reviews.csv")  # This CSV should contain 'processed_view' and 'theme' columns

df = load_dataframe()

# Page title
st.title("NLP Review Theme Analysis Dashboard")

# Sidebar theme filter
theme_filter = st.sidebar.selectbox("Filter by Theme", ["All"] + sorted(df["theme"].unique()))

# Filtered dataframe
if theme_filter != "All":
    df = df[df["theme"] == theme_filter]

# --- Theme Counts ---
st.subheader("Theme Distribution")
theme_counts = df["theme"].value_counts().reset_index()
theme_counts.columns = ["Theme", "Count"]
fig = px.bar(theme_counts, x="Theme", y="Count", color="Theme", title="Number of Reviews per Theme")
st.plotly_chart(fig)

# --- Word Cloud ---
st.subheader("Word Cloud")
text = " ".join(df["processed_review"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# --- Table of Reviews ---
st.subheader("Sample Reviews")
st.dataframe(df[["processed_review", "theme"]].reset_index(drop=True).head(10))