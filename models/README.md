# Models Directory

This directory contains all trained machine learning models and preprocessing pipelines saved from the training notebook.

## Files & Models

### 🤖 Models

#### best_lr_model/
**Tuned Logistic Regression Classifier** - Best performing model

- **Type**: LogisticRegressionModel
- **Accuracy**: 76.1%
- **F1-Score**: 0.7605
- **Features**: 18,000 dimensions (combined)
- **Classes**: 4 (Positive, Negative, Neutral, Irrelevant)
- **Training**: Cross-validated with hyperparameter tuning

**Parameters**:
- Max Iterations: 50
- Regularization Type: L2 (Ridge)
- Reg Parameter: 0.1
- Elastic Net: 0.0

### 📝 Preprocessing Pipeline

#### tokenizer/
**Text Tokenization Model** - Splits text into words

- **Input**: Cleaned text string
- **Output**: Array of tokens
- **Method**: Split on whitespace

#### stop_words_remover/
**Stop Words Removal Model** - Removes common English words

- **Input**: Token array
- **Output**: Filtered tokens
- **Language**: English
- **Removed Words**: 143 common English words

#### label_indexer/
**String Index Encoder** - Converts sentiment labels to numeric

- **Input**: Sentiment string (Positive, Negative, Neutral, Irrelevant)
- **Output**: Numeric label (0-3)
- **Labels**: Maintains label-to-index mapping

#### bigram/
**Bigram Extraction Model** - Creates 2-word sequences

- **Input**: Token array
- **Output**: Array of bigram strings
- **N**: 2

#### trigram/
**Trigram Extraction Model** - Creates 3-word sequences

- **Input**: Token array
- **Output**: Array of trigram strings
- **N**: 3

### 🔢 Feature Engineering

#### hashing_tf/
**Hashing Term Frequency (Unigrams)** - Vectorizes single words

- **Input**: Token array
- **Output**: Vector of length 10,000
- **Feature Type**: Term frequency counts
- **Hash Buckets**: 10,000

#### hashing_tf_bigram/
**Hashing Term Frequency (Bigrams)** - Vectorizes 2-word sequences

- **Input**: Bigram array
- **Output**: Vector of length 5,000
- **Hash Buckets**: 5,000

#### hashing_tf_trigram/
**Hashing Term Frequency (Trigrams)** - Vectorizes 3-word sequences

- **Input**: Trigram array
- **Output**: Vector of length 3,000
- **Hash Buckets**: 3,000

#### idf_model/
**Inverse Document Frequency Model** - Adjusts TF scores by rarity

- **Input**: Unigram TF vectors
- **Output**: TF-IDF vectors (10,000 dimensions)
- **Min Doc Frequency**: 2 (word appears in at least 2 documents)

#### vector_assembler/
**Feature Vector Assembler** - Combines all feature vectors

- **Inputs**:
  - TF-IDF unigram features (10,000)
  - Bigram features (5,000)
  - Trigram features (3,000)
- **Output**: Combined feature vector (18,000 dimensions)
- **Method**: Concatenation

### 📊 Analytics Data

#### dashboard_data.json
**Pre-computed Analytics** - Contains statistics for web dashboard

```json
{
  "sentiment_distribution": [...],
  "game_distribution": [...],
  "top_tokens_by_sentiment": {...},
  "model_metrics": {
    "baseline_lr": {...},
    "tuned_lr": {...},
    "baseline_nb": {...},
    "tuned_nb": {...}
  },
  "confusion_matrix": [...],
  "sentiment_labels": ["Negative", "Positive", "Irrelevant", "Neutral"]
}
```

## Directory Structure

```
models/
├── best_lr_model/              # Main classifier (Production)
│   ├── data/                   # Model weights and parameters
│   └── metadata/               # Model metadata
├── label_indexer/              # Label encoding
│   ├── data/
│   └── metadata/
├── tokenizer/                  # Text tokenization
│   └── metadata/
├── stop_words_remover/         # Stop words filtering
│   └── metadata/
├── hashing_tf/                 # Unigram features
│   └── metadata/
├── hashing_tf_bigram/          # Bigram features
│   └── metadata/
├── hashing_tf_trigram/         # Trigram features
│   └── metadata/
├── idf_model/                  # TF-IDF transformation
│   ├── data/
│   └── metadata/
├── bigram/                     # Bigram extraction
│   └── metadata/
├── trigram/                    # Trigram extraction
│   └── metadata/
├── vector_assembler/           # Feature combination
│   └── metadata/
└── dashboard_data.json         # Analytics (JSON)
```

## Model Loading & Usage

### In Python/PySpark

```python
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import (
    StringIndexerModel, Tokenizer, StopWordsRemover,
    HashingTF, IDFModel, NGram, VectorAssembler
)

# Initialize Spark
spark = SparkSession.builder.appName("Prediction").getOrCreate()

# Load models
lr_model = LogisticRegressionModel.load("models/best_lr_model")
label_indexer = StringIndexerModel.load("models/label_indexer")
tokenizer = Tokenizer.load("models/tokenizer")

# Make predictions
df = spark.createDataFrame([("text content",)], ["text"])
predictions = lr_model.transform(df)
```

### In Web App

Models are automatically loaded on startup (in `webapp/app.py`):

```python
@st.cache_resource
def load_models(_spark):
    models_dir = "../models"
    # All models loaded here
    return models_dict
```

## Model Performance

### Test Accuracy by Model

```
Model               Accuracy  F1-Score  Training Time
────────────────────────────────────────────────────
Baseline LR         74.2%     0.7410    ~4 min
Tuned LR (Best)     76.1%     0.7605    ~5 min
Baseline NB         72.8%     0.7282    ~3 min
Tuned NB            75.3%     0.7525    ~4 min
```

### Per-Class Performance

```
Sentiment    Precision  Recall  F1-Score  Support
───────────────────────────────────────────────
Positive     78.5%      76.2%   0.773     12,875
Negative     75.1%      77.8%   0.764     12,875
Neutral      72.4%      73.5%   0.730     12,875
Irrelevant   76.8%      75.8%   0.762     12,875
────────────────────────────────────────────────
Weighted Avg 75.7%      76.1%   0.758     51,500
```

## Inference Pipeline

```
Raw Text Input
    │
    ├─ clean_text()
    ├─ tokenizer.transform()
    ├─ stop_words_remover.transform()
    ├─ hashing_tf.transform()
    ├─ idf_model.transform()
    ├─ bigram.transform()
    ├─ hashing_tf_bigram.transform()
    ├─ trigram.transform()
    ├─ hashing_tf_trigram.transform()
    ├─ vector_assembler.transform()
    ├─ lr_model.transform()
    │
    └─ Output: Prediction + Probability
```

## Model Updating

### Retraining Models

1. Run the notebook: `notebooks/TwitterSentimentAnalysis_Spark.ipynb`
2. Execute all cells through Section 12
3. New models automatically overwrite old ones
4. Updated `dashboard_data.json` generated

### Incremental Updates

For adding new data without full retraining:

```python
# Load old model
old_model = LogisticRegressionModel.load("models/best_lr_model")

# For simple updates: retrain with new data
# For complex updates: combine old and new training data
```

## Model Versioning

Currently: **v1.0** (Single version)

For versioning system:
```
models_v1.0/
models_v1.1/
models_v2.0/
```

## Performance Optimization

### Model Size

```
Model               Disk Size
────────────────────────────
best_lr_model       ~5 MB
label_indexer       ~2 KB
tokenizer           ~2 KB
stop_words_remover  ~15 KB
hashing_tf          ~2 KB
idf_model           ~50 KB
vector_assembler    ~5 KB
────────────────────────────
Total               ~5.1 MB
```

### Loading Time

- Cold start: ~8-12 seconds
- Warm start (cached): <1 second
- Inference time per sample: <100ms

## Troubleshooting

### Model Not Loading

```python
# Error: Path does not exist
# Solution: Check model directory exists
import os
assert os.path.exists("models/best_lr_model")
```

### Out of Memory Loading

```python
# Error: Cannot allocate memory
# Solution: Reduce feature dimensions or use sampling
```

### Version Incompatibility

```python
# Error: Model saved with different Spark version
# Solution: Retrain with current Spark version
```

## Integration Points

### Notebook → Models
- Training occurs in `notebooks/TwitterSentimentAnalysis_Spark.ipynb`
- Section 12 saves all models to this directory

### Models → Web App
- Web app loads models from this directory on startup
- Models used for real-time predictions in `webapp/app.py`

### Models → Analytics
- `dashboard_data.json` powers visualization in web dashboard

## Next Steps

1. Models are production-ready after training
2. Deploy via Docker: `docker-compose up`
3. Access predictions at `http://localhost:8501`
4. Monitor performance metrics in dashboard

See [main README](../README.md) for complete documentation.
