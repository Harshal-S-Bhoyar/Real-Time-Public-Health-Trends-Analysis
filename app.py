# app.py
# environment name is redditproj
# activate redditproj\Scripts\activate
# to run write streamlit run app.py
import streamlit as st
import praw
import pandas as pd
import os
from dotenv import load_dotenv
import logging
from geopy.geocoders import Nominatim
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium  # For displaying Folium maps in Streamlit

# ----------------------------
# Setup
# ----------------------------

# Load environment variables (optional)
load_dotenv()

# Configure logging
logging.basicConfig(
    filename="health_trends.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Reddit credentials (from .env or hardcoded)
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "_r-auBrrOUMdCwudxBtZJw")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "ldQbRtpOltzZZHbJfDiuZGZq0s30bg")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "windows:my-cool-app:v1.0 (by /u/Happy-Syllabub-6994)")

# Initialize geolocator
geolocator = Nominatim(user_agent="health_trends_app")
LOCATIONS = ["New York", "California", "London", "Toronto", "Delhi", "Texas", "Florida", "Paris", "Tokyo", "Mumbai"]

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Reddit Health Trends Analyzer", layout="wide")
st.title("🩺 Reddit Health Trends Analyzer")
st.markdown("Enter a keyword to analyze health-related discussions on Reddit.")

# Input widget (replaces ipywidgets.Text)
keyword = st.text_input("Keyword", value="flu", placeholder="e.g., flu, anxiety, headache")

# Button (replaces ipywidgets.Button)
if st.button(" Search and Analyze", type="primary"):
    if not keyword.strip():
        st.warning("Please enter a keyword.")
    else:
        keyword = keyword.strip()
        st.info(f"Collecting posts for keyword: **{keyword}**")
        logging.info(f"Collecting posts for keyword: {keyword}")

        try:
            # Authenticate with Reddit
            reddit = praw.Reddit(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                user_agent=USER_AGENT
            )
            reddit.read_only = True
            try:
                user = reddit.user.me()
                st.success(f"Authentication successful! (User: {user})")
                logging.info("Authentication successful")
            except Exception:
                st.success("Authentication successful! (Read-only mode)")

            # Collect posts
            posts = []
            subreddit = reddit.subreddit("health+anxiety+mentalhealth+askdocs")
            for post in subreddit.search(keyword, limit=100):
                posts.append({
                    "date": post.created_utc,
                    "text": post.title + " " + (post.selftext or ""),
                    "subreddit": post.subreddit.display_name,
                    "username": post.author.name if post.author else "Unknown",
                    "keyword": keyword
                })

            if not posts:
                st.warning("No posts found for this keyword.")
                st.stop()

            df = pd.DataFrame(posts)
            df["date"] = pd.to_datetime(df["date"], unit="s")
            st.write(f"Collected **{len(posts)}** posts")
            logging.info(f"Collected {len(posts)} posts")

            # ----------------------------
            # Data Cleaning
            # ----------------------------
            def clean_text(text):
                text = re.sub(r"http\S+|www\S+|https\S+", "", text)
                text = re.sub(r"@\w+|\#\w+", "", text)
                text = re.sub(r"[^\w\s]", "", text)
                return text.strip()

            df["cleaned_text"] = df["text"].apply(clean_text)
            df["subreddit"] = df["subreddit"].fillna("Unknown")
            df = df.drop_duplicates(subset=["text"])
            st.write("Data cleaned")
            logging.info("Data cleaned")

            # ----------------------------
            # Location Inference
            # ----------------------------
            def extract_location(text):
                for location in LOCATIONS:
                    if re.search(r'\b' + re.escape(location) + r'\b', text, re.IGNORECASE):
                        try:
                            geo = geolocator.geocode(location)
                            if geo:
                                return location, geo.latitude, geo.longitude
                        except:
                            continue
                return None, None, None

            df["location"], df["latitude"], df["longitude"] = zip(*df["text"].apply(extract_location))
            st.write("Location data added")
            logging.info("Location data added")

            # ----------------------------
            # Sentiment Analysis
            # ----------------------------
            def get_sentiment(text):
                return TextBlob(text).sentiment.polarity

            df["sentiment"] = df["cleaned_text"].apply(get_sentiment)
            df["sentiment_category"] = df["sentiment"].apply(
                lambda x: "Positive" if x > 0.1 else "Negative" if x < -0.1 else "Neutral"
            )
            st.write(" Sentiment analysis completed")
            logging.info("Sentiment analysis completed")

            # ----------------------------
            # Topic Modeling
            # ----------------------------
            texts = df["cleaned_text"].dropna().tolist()
            if not texts:
                st.warning("No valid texts for topic modeling.")
                df["topic"] = None
            else:
                vectorizer = CountVectorizer(stop_words="english", max_df=0.95, min_df=2)
                doc_term_matrix = vectorizer.fit_transform(texts)
                lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
                lda_output = lda_model.fit_transform(doc_term_matrix)
                df["topic"] = [None] * len(df)
                valid_indices = df["cleaned_text"].dropna().index
                df.loc[valid_indices, "topic"] = lda_output.argmax(axis=1)

                st.write("Topic modeling completed")
                feature_names = vectorizer.get_feature_names_out()
                topics_info = []
                for topic_idx, topic in enumerate(lda_model.components_):
                    top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
                    topics_info.append(f"Topic {topic_idx}: {', '.join(top_words)}")
                st.text("\n".join(topics_info))
                logging.info("Topic modeling completed")

            # ----------------------------
            # Time Series Analysis
            # ----------------------------
            daily_posts = df.groupby(df["date"].dt.date).size().reset_index(name="post_count")
            daily_posts["date"] = pd.to_datetime(daily_posts["date"])
            daily_posts.set_index("date", inplace=True)
            if len(daily_posts) > 7:
                decomposition = seasonal_decompose(daily_posts["post_count"], model="additive", period=7)
            else:
                decomposition = None
            st.write("Time series analysis completed")
            logging.info("Time series analysis completed")

            # ----------------------------
            # Visualizations
            # ----------------------------

            # 1. Posts over time
            st.subheader("Posts Over Time")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            daily_posts.plot(ax=ax1)
            plt.title(f"{keyword.capitalize()}-Related Posts Over Time")
            st.pyplot(fig1)

            # 2. Sentiment distribution
            st.subheader("Sentiment Distribution")
            sentiment_counts = df["sentiment_category"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sentiment_counts.plot(kind="bar", ax=ax2, color=["green", "gray", "red"])
            plt.xticks(rotation=0)
            st.pyplot(fig2)

            # 3. Topic distribution
            if "topic" in df.columns and df["topic"].notna().any():
                st.subheader("Topic Distribution")
                topic_counts = df["topic"].value_counts()
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                topic_counts.plot(kind="bar", ax=ax3)
                plt.xticks(rotation=0)
                st.pyplot(fig3)

            # 4. Map
            location_df = df.dropna(subset=["latitude", "longitude"])
            if not location_df.empty:
                st.subheader("Post Locations")
                m = folium.Map(location=[20, 0], zoom_start=2)
                marker_cluster = MarkerCluster().add_to(m)
                for _, row in location_df.iterrows():
                    folium.Marker(
                        location=[row["latitude"], row["longitude"]],
                        popup=f"{row['location']}: {row['text'][:100]}...",
                        tooltip=row["location"]
                    ).add_to(marker_cluster)
                st_folium(m, width=700, height=400)
            else:
                st.info("No location data available for mapping.")

            # 5. Time series decomposition
            if decomposition is not None:
                st.subheader("Time Series Decomposition")
                fig4, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                decomposition.observed.plot(ax=axes[0], title="Observed")
                decomposition.trend.plot(ax=axes[1], title="Trend")
                decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
                decomposition.resid.plot(ax=axes[3], title="Residual")
                plt.tight_layout()
                st.pyplot(fig4)
            else:
                st.info("Not enough data for time series decomposition.")

            # Save results
            df.to_csv("analyzed_posts.csv", index=False)
            st.success("Analysis complete! Results saved to `analyzed_posts.csv`")
            logging.info("Analysis complete")

            # Optional: Allow download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "analyzed_posts.csv", "text/csv")

        except praw.exceptions.RedditAPIException as e:
            st.error(f"Reddit API error: {e}")
            logging.error(f"Reddit API error: {e}")
        except Exception as e:
            st.exception(f"An error occurred: {e}")
            logging.error(f"Error: {e}")
else:
    st.info("Enter a keyword and click **Search and Analyze** to begin.")