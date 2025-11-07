# Real-Time-Public-Health-Trends-Analysis
A Python-based project to analyze public health trends from Reddit data using PRAW, with sentiment analysis (TextBlob), topic modeling (scikit-learn LDA), location inference (GeoPy), time series analysis (Statsmodels), and visualizations (Matplotlib, Folium) in JupyterLab

# Overview
This project analyzes public health trends by collecting data from Reddit using the Reddit API (PRAW). It performs sentiment analysis with TextBlob, topic modeling with scikit-learn’s LDA, location inference with GeoPy, and time series analysis with Statsmodels. Results are visualized through interactive maps (Folium) and plots (Matplotlib), with data stored in SQLite for further analysis. The project is implemented in Python and executed in JupyterLab, providing insights into health-related discussions on social media.

# Features
- Data Collection: Scrapes Reddit posts from health-related subreddits using PRAW.
- Sentiment Analysis: Analyzes post sentiment using TextBlob, categorizing them as Positive, Negative, or Neutral.
- Topic Modeling: Identifies key topics in posts using scikit-learn’s Latent Dirichlet Allocation (LDA).
- Location Inference: Extracts and maps locations mentioned in posts using GeoPy and Folium.
- Time Series Analysis: Examines posting trends over time with Statsmodels’ seasonal decomposition.
- Visualizations: Generates interactive maps (Folium) and plots (Matplotlib) for data insights.
- Data Storage: Saves processed data in an SQLite database for future analysis.

# Prerequisites
- Python 3.9 or higher
- A Reddit account to obtain API credentials (Client ID, Client Secret, User Agent)
- Git (to clone the repository)
- JupyterLab (to run the notebook)

# Installation

## Clone the Repository:
- git clone https://github.com/yourusername/health-trends-analysis.git
cd health-trends-analysis
- Replace yourusername with your GitHub username.

## Set Up a Virtual Environment (Recommended):
- python -m virtualenv health_trends_env
- health_trends_env\Scripts\activate  # On Windows

# Install Dependencies:
- pip install scikit-learn scipy numpy pandas matplotlib statsmodels
- pip install praw textblob geopy folium python-dotenv
- pip install sqlalchemy streamlit streamlit_folium
- pip install jupyterlab ipywidgets jupyterlab-widgets

# Set Up Reddit API Credentials:
- Create a Reddit app to get your API credentials (see Reddit API Documentation).
+ REDDIT_CLIENT_ID=your_client_id
+ REDDIT_CLIENT_SECRET=your_client_secret
+ REDDIT_USER_AGENT=your_user_agent

## Note : Create an Virtual environment and activate it and after installing libraries Then launch
jupyter lab

# Usage
- Open JupyterLab in your browser (typically at http://localhost:8888/lab).
- Open the health_trends_dashboard.ipynb notebook from the File Browser.
- Run the notebook cells to:
- Authenticate with the Reddit API using your credentials.
- Input a keyword (e.g., "flu") via the interactive widget.
- Collect and analyze Reddit posts, generating visualizations and saving results.
- Results are saved as:
- analyzed_posts.csv: Processed data in CSV format.
- health_trends.log: Log file for debugging.
- SQLite database (optional, if implemented).

# File Structure

- health_analysis.ipynb: Main JupyterLab notebook with the analysis pipeline.
- health_analysis.log: Log file generated during execution.
- analyzed_posts.csv: Output CSV file with analyzed data.
- README.md: Project documentation (this file).

# Requirements

## The project uses the following Python libraries (installed during setup):
- praw: Reddit API access
- pandas: Data manipulation
- python-dotenv: Environment variable management
- logging: Event logging
- ipywidgets: Interactive widgets in JupyterLab
- IPython: Display utilities
- geopy: Location inference
- re: Text cleaning
- textblob: Sentiment analysis
- scikit-learn: Topic modeling
- matplotlib: Visualizations
- statsmodels: Time series analysis
- folium: Interactive maps
- sqlalchemy: SQLite database storage
- sqlite3: SQLite database queries
- jupyterlab: Notebook environment
- scipy, numpy: Numerical operations

# Contributing
## Contributions are welcome! To contribute:
- Fork the repository.
- Create a new branch (git checkout -b feature/your-feature).
- Make your changes and commit (git commit -m "Add your feature").
- Push to your branch (git push origin feature/your-feature).
- Open a pull request with a detailed description of your changes.
