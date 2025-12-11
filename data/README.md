# Data Directory

This directory contains the datasets used for training and validation of the Twitter sentiment analysis model.

## Files

### 📊 twitter_training.csv
**Training Dataset** - Primary dataset for model training

- **Records**: 170,000 tweets
- **Size**: ~10 MB
- **Columns**: `id`, `game`, `sentiment`, `text`
- **Format**: Comma-separated values (CSV)

**Class Distribution**:
- Positive: 80,500 (47.4%)
- Negative: 44,500 (26.2%)
- Neutral: 40,100 (23.6%)
- Irrelevant: 4,900 (2.9%)

**After Balancing** (for training):
- Each class: 12,875 samples (25%)

### 📊 twitter_validation.csv
**Validation Dataset** - Used for model evaluation

- **Records**: 10,000 tweets
- **Size**: ~164 KB
- **Columns**: Same as training data
- **Format**: Comma-separated values (CSV)

## Data Structure

### CSV Format

```
id,game,sentiment,text
1,game_name,Positive,"Tweet text content here..."
2,game_name,Negative,"Another tweet..."
...
```

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Unique identifier for the tweet |
| `game` | String | Name of the game being discussed |
| `sentiment` | String | Sentiment class (Positive, Negative, Neutral, Irrelevant) |
| `text` | String | Tweet content/text |

## Data Preprocessing in Pipeline

```
Raw CSV
    │
    ├─ Load into Spark DataFrame
    ├─ Remove nulls
    ├─ Balance classes (undersampling)
    ├─ Clean text (lowercase, remove URLs, etc.)
    ├─ Tokenize
    ├─ Remove stop words
    ├─ Extract features (TF-IDF, N-grams)
    └─ Train models
```

## Statistics

### Training Set Statistics

```
Total Records: 170,000
Games: Multiple (gaming-related)
Sentiments: 4 classes
Avg Tweet Length: ~120 characters
Min Length: 10 characters
Max Length: 280 characters
```

### Validation Set Statistics

```
Total Records: 10,000
Games: Same distribution as training
Sentiments: 4 classes
Class Balance: Varies (not pre-balanced)
```

## Data Exploration

### Loading Data

```python
import pandas as pd
from pyspark.sql import SparkSession

# Spark method (recommended)
spark = SparkSession.builder.appName("read_data").getOrCreate()
df = spark.read.csv('data/twitter_training.csv', header=False, inferSchema=True)

# Pandas method (for smaller datasets)
df = pd.read_csv('data/twitter_training.csv')
```

### Quick Analysis

```python
# Shape
print(f"Records: {df.count()}")
print(f"Columns: {len(df.columns)}")

# Unique values
df.select('sentiment').distinct().show()
df.select('game').distinct().count()

# Distribution
df.groupBy('sentiment').count().show()
```

## Data Quality

### Potential Issues

1. **Imbalanced Classes**: Fixed with stratified undersampling
2. **Null Values**: Removed during preprocessing
3. **Duplicates**: Some tweets may appear multiple times
4. **Text Encoding**: Handles emoji and special characters
5. **Missing Text**: Rows with empty text column removed

### Cleaning Process

- Remove null values
- Lowercase all text
- Remove URLs (http://, https://, www.)
- Remove mentions (@username) and hashtags (#topic)
- Remove emojis and non-ASCII characters
- Expand contractions (n't → not, 're → are)
- Remove numbers
- Remove special punctuation
- Normalize whitespace

## Data Splits

### Training Workflow

```
Raw Data (170K)
    │
    ├─ Train: 51.5K (balanced - 4 x 12,875)
    └─ Test: 51.5K (balanced - same)

Validation Data (10K)
    └─ Used for final evaluation
```

## Usage

### In Notebook

```python
# From: notebooks/TwitterSentimentAnalysis_Spark.ipynb

# Load training data
spark_train_df = spark.read.csv('file://data/twitter_training.csv', 
                                 header=False, inferSchema=True)
spark_train_df = spark_train_df.toDF('id', 'game', 'sentiment', 'text')

# Load validation data
spark_val_df = spark.read.csv('file://data/twitter_validation.csv',
                               header=False, inferSchema=True)
spark_val_df = spark_val_df.toDF('id', 'game', 'sentiment', 'text')
```

### In Web App

The web app uses pre-trained models to make predictions. Training data is not directly used but the models were trained on this data.

## Adding New Data

### Format Requirements

1. Save as CSV with columns: `id`, `game`, `sentiment`, `text`
2. Use same sentiment values: Positive, Negative, Neutral, Irrelevant
3. Ensure UTF-8 encoding
4. No header row in CSV (data only)

### Processing New Data

```python
# Load new data
new_df = spark.read.csv('path/to/new_data.csv', header=False, inferSchema=True)

# Apply same preprocessing
# Tokenize, remove stop words, extract features
# Use trained models for predictions
```

## Dataset License & Attribution

The Twitter Sentiment Analysis dataset is based on publicly available Twitter data.

- **Source**: Kaggle - Twitter Entity Sentiment Analysis
- **License**: Check original source
- **Attribution**: Required - please credit original source

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- [Dataset Paper](https://arxiv.org/abs/1411.4280)
- [Twitter Data Policy](https://twitter.com/en/developers/twitter-api-documentation)

## Data Privacy

- No personally identifiable information beyond public tweets
- Follow Twitter's Terms of Service
- Respect user privacy and content

## Next Steps

After preprocessing:
1. Data flows to feature engineering
2. Models trained on processed features
3. Predictions stored and visualized
4. Results available in web dashboard

See [main README](../README.md) for complete project documentation.
