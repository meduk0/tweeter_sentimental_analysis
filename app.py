"""
Twitter Sentiment Analysis Web Application
A Streamlit-based interface for sentiment prediction and data visualization
"""

import os
import sys

# Set Java options BEFORE importing PySpark
os.environ['_JAVA_OPTIONS'] = '-Duser.country=US -Duser.language=en -Djavax.security.auth.useSubjectCredsOnly=false'
os.environ['JAVA_TOOL_OPTIONS'] = '-Djavax.security.auth.useSubjectCredsOnly=false'

import streamlit as st
import json
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import (
    StringIndexerModel, Tokenizer, StopWordsRemover,
    HashingTF, IDFModel, NGram, VectorAssembler
)
from pyspark.ml.classification import LogisticRegressionModel

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_spark():
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName("TwitterSentimentWeb") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.extraJavaOptions",
                "-Duser.country=US -Duser.language=en -Djavax.security.auth.useSubjectCredsOnly=false") \
        .config("spark.executor.extraJavaOptions",
                "-Duser.country=US -Duser.language=en -Djavax.security.auth.useSubjectCredsOnly=false") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .config("spark.metrics.conf.*.sink.jmx.class", "org.apache.spark.metrics.sink.JmxSink") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark


@st.cache_resource
def load_models(_spark):
    """Load all trained models and preprocessing components"""
    models_dir = "saved_models"
    
    try:
        # Load models
        lr_model = LogisticRegressionModel.load(os.path.join(models_dir, "best_lr_model"))
        label_indexer = StringIndexerModel.load(os.path.join(models_dir, "label_indexer"))
        tokenizer = Tokenizer.load(os.path.join(models_dir, "tokenizer"))
        stop_words_remover = StopWordsRemover.load(os.path.join(models_dir, "stop_words_remover"))
        hashing_tf = HashingTF.load(os.path.join(models_dir, "hashing_tf"))
        hashing_tf_bigram = HashingTF.load(os.path.join(models_dir, "hashing_tf_bigram"))
        hashing_tf_trigram = HashingTF.load(os.path.join(models_dir, "hashing_tf_trigram"))
        idf_model = IDFModel.load(os.path.join(models_dir, "idf_model"))
        bigram = NGram.load(os.path.join(models_dir, "bigram"))
        trigram = NGram.load(os.path.join(models_dir, "trigram"))
        vector_assembler = VectorAssembler.load(os.path.join(models_dir, "vector_assembler"))
        
        # Get label mapping from the StringIndexerModel
        # labels attribute contains the ordered list of sentiment labels
        labels_array = label_indexer.labels
        label_to_sentiment = {int(idx): label for idx, label in enumerate(labels_array)}
        
        return {
            'lr_model': lr_model,
            'label_indexer': label_indexer,
            'tokenizer': tokenizer,
            'stop_words_remover': stop_words_remover,
            'hashing_tf': hashing_tf,
            'hashing_tf_bigram': hashing_tf_bigram,
            'hashing_tf_trigram': hashing_tf_trigram,
            'idf_model': idf_model,
            'bigram': bigram,
            'trigram': trigram,
            'vector_assembler': vector_assembler,
            'label_map': label_to_sentiment
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None


@st.cache_data
def load_dashboard_data():
    """Load pre-computed dashboard data"""
    try:
        with open('saved_models/dashboard_data.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")
        return None


def clean_text(text):
    """Clean and preprocess text"""
    if text is None:
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[@#]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    contractions = {
        "n't": " not", "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'ve": " have", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    
    return text


def predict_sentiment(text, spark, models):
    """Predict sentiment for input text"""
    try:
        # Clean text
        cleaned = clean_text(text)
        
        if not cleaned.strip():
            return None, None, "Text is empty after cleaning. Please enter valid text."
        
        # Create DataFrame
        df = spark.createDataFrame([(1, cleaned)], ["id", "cleaned_text"])
        
        # Tokenization
        df = models['tokenizer'].transform(df)
        
        # Remove stop words
        df = models['stop_words_remover'].transform(df)
        
        # HashingTF for unigrams
        df = models['hashing_tf'].transform(df)
        
        # IDF
        df = models['idf_model'].transform(df)
        
        # Create bigrams
        df = models['bigram'].transform(df)
        df = models['hashing_tf_bigram'].transform(df)
        
        # Create trigrams
        df = models['trigram'].transform(df)
        df = models['hashing_tf_trigram'].transform(df)
        
        # Assemble features
        df = models['vector_assembler'].transform(df)
        
        # Predict
        prediction = models['lr_model'].transform(df)
        
        # Get result
        result = prediction.select('prediction', 'probability').collect()[0]
        predicted_label = int(result['prediction'])
        probability = float(result['probability'][predicted_label])
        sentiment = models['label_map'].get(predicted_label, "Unknown")
        
        return sentiment, probability, None
        
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"


def render_dashboard(data):
    """Render the dashboard tab"""
    st.markdown('<h1 class="main-header">📊 Twitter Sentiment Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Model Performance Metrics
    st.header("🏆 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = data['model_metrics']
    
    with col1:
        st.metric(
            label="Baseline LR Accuracy",
            value=f"{metrics['baseline_lr']['accuracy']:.4f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Tuned LR Accuracy",
            value=f"{metrics['tuned_lr']['accuracy']:.4f}",
            delta=f"+{(metrics['tuned_lr']['accuracy'] - metrics['baseline_lr']['accuracy']):.4f}"
        )
    
    with col3:
        st.metric(
            label="Baseline NB Accuracy",
            value=f"{metrics['baseline_nb']['accuracy']:.4f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Tuned NB Accuracy",
            value=f"{metrics['tuned_nb']['accuracy']:.4f}",
            delta=f"+{(metrics['tuned_nb']['accuracy'] - metrics['baseline_nb']['accuracy']):.4f}"
        )
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Distribution")
        sentiment_df = pd.DataFrame(data['sentiment_distribution'])
        fig = px.pie(
            sentiment_df,
            values='count',
            names='sentiment',
            title='Training Data Sentiment Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎮 Game Distribution")
        game_df = pd.DataFrame(data['game_distribution']).sort_values('count', ascending=False).head(10)
        fig = px.bar(
            game_df,
            x='count',
            y='game',
            orientation='h',
            title='Top 10 Games in Dataset',
            color='count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("🔍 Confusion Matrix - Best Model (Tuned Logistic Regression)")
    
    confusion_matrix = data['confusion_matrix']
    labels = data['sentiment_labels']
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=confusion_matrix,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=700,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top Tokens by Sentiment
    st.subheader("📝 Top Frequent Words by Sentiment")
    
    sentiment_tabs = st.tabs(data['sentiment_labels'])
    
    for idx, sentiment in enumerate(data['sentiment_labels']):
        with sentiment_tabs[idx]:
            tokens_data = data['top_tokens_by_sentiment'][sentiment]
            tokens_df = pd.DataFrame(tokens_data)
            
            fig = px.bar(
                tokens_df,
                x='count',
                y='token',
                orientation='h',
                title=f'Top 15 Words in {sentiment} Tweets',
                color='count',
                color_continuous_scale='Sunset'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Dataset Statistics
    st.markdown("---")
    st.subheader("📊 Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_tweets = sum([item['count'] for item in data['sentiment_distribution']])
        st.metric("Total Tweets", f"{total_tweets:,}")
    
    with col2:
        num_games = len(data['game_distribution'])
        st.metric("Number of Games", num_games)
    
    with col3:
        num_sentiments = len(data['sentiment_distribution'])
        st.metric("Sentiment Classes", num_sentiments)


def render_prediction(spark, models):
    """Render the prediction tab"""
    st.markdown('<h1 class="main-header">🔮 Sentiment Prediction</h1>', 
                unsafe_allow_html=True)
    
    st.write("Enter a tweet or text message to analyze its sentiment using our trained model.")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "Enter your text here:",
            height=150,
            placeholder="e.g., I absolutely love this game! It's amazing!",
            help="Type or paste any text to analyze its sentiment"
        )
    
    with col2:
        st.write("")
        st.write("")
        predict_button = st.button("🚀 Predict Sentiment", type="primary", use_container_width=True)
        
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state['clear_input'] = True
            st.rerun()
    
    # Handle clear button
    if st.session_state.get('clear_input', False):
        user_input = ""
        st.session_state['clear_input'] = False
    
    # Prediction
    if predict_button and user_input.strip():
        with st.spinner("🔄 Analyzing sentiment..."):
            sentiment, confidence, error = predict_sentiment(user_input, spark, models)
            
            if error:
                st.error(f"❌ {error}")
            else:
                # Display result
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
                # Sentiment color mapping
                sentiment_colors = {
                    'Positive': '#28a745',
                    'Negative': '#dc3545',
                    'Neutral': '#6c757d',
                    'Irrelevant': '#ffc107'
                }
                
                color = sentiment_colors.get(sentiment, '#6c757d')
                
                # Prediction box
                st.markdown(
                    f'<div style="background: {color}; padding: 2rem; border-radius: 1rem; '
                    f'color: white; text-align: center; margin: 2rem 0;">'
                    f'<h2 style="margin: 0;">Predicted Sentiment: {sentiment}</h2>'
                    f'<p style="font-size: 1.5rem; margin-top: 1rem;">Confidence: {confidence*100:.2f}%</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Confidence meter
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Confidence Level:**")
                    st.progress(confidence)
                
                with col2:
                    if confidence > 0.8:
                        st.success("🎯 High Confidence")
                    elif confidence > 0.6:
                        st.info("✅ Medium Confidence")
                    else:
                        st.warning("⚠️ Low Confidence")
                
                # Original vs Cleaned text
                st.markdown("---")
                st.subheader("🔧 Text Processing")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Text:**")
                    st.code(user_input)
                
                with col2:
                    st.write("**Cleaned Text:**")
                    st.code(clean_text(user_input))
    
    elif predict_button:
        st.warning("⚠️ Please enter some text to analyze.")


def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/twitter.png", width=100)
        st.title("Twitter Sentiment Analysis")
        st.markdown("---")
        
        st.markdown("""
        ### About
        This application uses **Apache Spark MLlib** and **Logistic Regression** 
        to analyze sentiment in Twitter data.
        
        ### Features
        - 📊 **Dashboard**: View data statistics and visualizations
        - 🔮 **Predict**: Analyze sentiment of custom text
        
        ### Classes
        - ✅ Positive
        - ❌ Negative
        - ➖ Neutral
        - ❔ Irrelevant
        """)
        
        st.markdown("---")
        st.caption("Powered by Apache Spark & Streamlit")
    
    # Initialize
    try:
        spark = initialize_spark()
        models = load_models(spark)
        dashboard_data = load_dashboard_data()
        
        if models is None or dashboard_data is None:
            st.error("Failed to load models or data. Please ensure the models are trained and saved.")
            return
        
        # Main tabs
        tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Predict"])
        
        with tab1:
            render_dashboard(dashboard_data)
        
        with tab2:
            render_prediction(spark, models)
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
