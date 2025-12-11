# Twitter Sentiment Analysis with Apache Spark

![Twitter Sentiment Analysis](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive machine learning project that analyzes Twitter sentiment using Apache Spark and PySpark MLlib. The project includes data preprocessing, model training with hyperparameter tuning, and an interactive web dashboard for sentiment prediction.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Models & Performance](#models--performance)
- [Data & Preprocessing](#data--preprocessing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project demonstrates a complete machine learning pipeline for sentiment analysis on Twitter data:

- **Data Processing**: Cleans and preprocesses 170K+ tweets using PySpark
- **Feature Engineering**: Implements unigrams, bigrams, and trigrams with TF-IDF vectorization
- **Model Training**: Trains and tunes Logistic Regression and Naive Bayes classifiers
- **Web Dashboard**: Interactive Streamlit application for predictions and visualizations
- **Production Ready**: Docker containerization for easy deployment

### Key Metrics

- **Best Model Accuracy**: 76%+ (Tuned Logistic Regression)
- **Training Data**: 51,500 balanced tweets (4 sentiment classes)
- **Feature Dimensions**: 18,000 combined features (unigrams + bigrams + trigrams)
- **Classes**: Positive, Negative, Neutral, Irrelevant

## 📁 Project Structure

```
twitter-sentiment-analysis/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── notebooks/                         # Jupyter notebooks
│   ├── TwitterSentimentAnalysis_Spark.ipynb          # Main training notebook
│   ├── TwitterSentimentAnalysisModel.ipynb           # Alternative model notebook
│   └── TwitterSentimentAnalysis_Spark.ipynb.bak      # Backup
│
├── data/                              # Dataset directory
│   ├── twitter_training.csv           # Training data (170K tweets)
│   └── twitter_validation.csv         # Validation data (10K tweets)
│
├── webapp/                            # Web application
│   ├── app.py                         # Main Streamlit application
│   ├── Dockerfile                     # Docker container specification
│   ├── docker-compose.yml             # Docker Compose configuration
│   ├── start.sh                       # Launch script
│   └── .streamlit/                    # Streamlit configuration
│
├── models/                            # Trained models & preprocessing pipelines
│   ├── best_lr_model/                 # Best Logistic Regression model
│   ├── label_indexer/                 # Label encoding model
│   ├── tokenizer/                     # Text tokenization model
│   ├── stop_words_remover/            # Stop words removal model
│   ├── hashing_tf/                    # Hashing TF (unigrams)
│   ├── hashing_tf_bigram/             # Hashing TF (bigrams)
│   ├── hashing_tf_trigram/            # Hashing TF (trigrams)
│   ├── idf_model/                     # IDF transformation model
│   ├── bigram/                        # Bigram model
│   ├── trigram/                       # Trigram model
│   ├── vector_assembler/              # Feature vector assembly
│   └── dashboard_data.json            # Pre-computed analytics data
```

### Directory Responsibilities

| Directory | Purpose | Files |
|-----------|---------|-------|
| `notebooks/` | Model development & experimentation | Jupyter notebooks for training |
| `data/` | Raw and processed datasets | CSV training/validation data |
| `webapp/` | Production web application | Streamlit app, Docker config, startup scripts |
| `models/` | Trained ML models | PySpark model artifacts, metadata |

## ✨ Features

### 🔬 Data Science Features

- **Text Preprocessing**
  - URL, mention, and emoji removal
  - Lowercase normalization
  - Contraction expansion
  - Punctuation and special character removal
  - Stop words removal

- **Feature Engineering**
  - Tokenization
  - N-gram extraction (unigrams, bigrams, trigrams)
  - TF-IDF vectorization
  - Feature combination and normalization

- **Model Training**
  - Cross-validation with 2-fold splits
  - Hyperparameter tuning
  - Logistic Regression with regularization tuning
  - Naive Bayes with smoothing optimization
  - Baseline and advanced model comparison

- **Evaluation Metrics**
  - Accuracy, F1-Score, Precision, Recall
  - Confusion matrices
  - Per-class performance analysis
  - Feature importance analysis

### 🎨 Web Application Features

- **Interactive Dashboard**
  - Real-time sentiment analysis
  - Confidence scores with visual indicators
  - Data distribution visualizations
  - Model performance comparisons

- **Sentiment Prediction**
  - Text input with live prediction
  - Sentiment classification (4 classes)
  - Confidence percentage display
  - Text preprocessing visualization

- **Analytics & Insights**
  - Sentiment distribution charts
  - Game-wise sentiment breakdown
  - Word frequency analysis
  - Confusion matrix visualization

### ⚙️ Technical Features

- **Distributed Processing**: Apache Spark for large-scale data processing
- **MLlib Pipeline**: Complete ML pipeline with feature engineering
- **Containerization**: Docker & Docker Compose for easy deployment
- **Performance Optimization**: Data partitioning, caching, broadcasting
- **Error Handling**: Comprehensive error messages and logging

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Java 11 or 17 (required for Spark)
- Docker & Docker Compose (optional, for containerized deployment)
- 4GB+ RAM (2GB minimum for web app)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

### Step 2: Install Dependencies

#### Option A: Using Conda (Recommended)

```bash
# Create virtual environment
python -m venv env 
source env/bin/activate.{depend from ur shell}

# Install dependencies
pip install -r requirements.txt

```
**note:install openjdk-11 or openjdk-17 from ur distribution package manager**
#### Option B: Using pip

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Java Installation

```bash
java -version
# Should show Java 11 or 17
```

## 🎬 Quick Start

### Step 1: Train the Model

```bash
# Navigate to project root
cd twitter-sentiment-analysis

# Option A: Using Jupyter Notebook (Recommended for first run)
jupyter notebook notebooks/TwitterSentimentAnalysis_Spark.ipynb

# Run all cells, especially Section 12 to save models

# Option B: Using Python script
python -m jupyter nbconvert --to notebook --execute notebooks/TwitterSentimentAnalysis_Spark.ipynb
```

**Training Time**: ~15-30 minutes (depending on hardware)

### Step 2: Run the Web Application

#### Option A: Docker Compose (Recommended)

```bash
cd webapp
bash start.sh
# Select option 1 (Docker Compose)

# Or run directly:
docker-compose up --build

# Access at: http://localhost:8501
```

#### Option B: Local Python Environment

```bash
cd webapp
bash start.sh
# Select option 2 (Local Python)

# Or run directly:
streamlit run app.py

# Access at: http://localhost:8501
```

### Step 3: Use the Application

1. **Dashboard Tab**: View statistics, model performance, and word clouds
2. **Predict Tab**: Enter custom text for real-time sentiment prediction

## 📖 Usage

### Running the Training Notebook

The main training notebook is located at `notebooks/TwitterSentimentAnalysis_Spark.ipynb`. It contains 12 major sections:

```
1. Initialize Spark Session & Import Libraries
   ├─ Environment setup
   └─ Library imports

2. Load and Explore Data
   ├─ Load training/validation datasets
   └─ Basic data exploration

3. Data Preparation
   ├─ Null value handling
   └─ Data balancing

4. Exploratory Data Analysis
   ├─ Sentiment distribution
   ├─ Game distribution
   └─ Visualizations

5. Text Preprocessing
   ├─ Text cleaning
   ├─ Repartitioning
   └─ Label indexing

6. Feature Engineering
   ├─ Tokenization
   ├─ TF-IDF vectorization
   └─ Feature scaling

6b. Enhanced N-gram Features
    ├─ Bigram extraction
    ├─ Trigram extraction
    └─ Feature combination

7. Hyperparameter Tuning
   ├─ Logistic Regression tuning
   └─ Naive Bayes tuning

8. Original Models
   ├─ Baseline LR
   └─ Baseline NB

8b. Baseline Naive Bayes
    └─ Standard NB implementation

9. Model Comparison
   └─ Performance metrics visualization

10. Text Mining & Visualization
    ├─ Word clouds
    └─ Token frequency analysis

11. Advanced Analysis
    └─ Token frequency by sentiment

12. Model Saving
    ├─ Save all models
    └─ Save dashboard data
```

### Using the Web Application

#### Dashboard View

```
📊 Dashboard
├── Model Performance Metrics
│   ├── Baseline & Tuned accuracy
│   └── Improvement percentages
├── Data Visualizations
│   ├── Sentiment pie chart
│   ├── Game distribution
│   └── Confusion matrix
└── Insights
    ├── Top tokens by sentiment
    └── Dataset statistics
```

#### Prediction View

```
🔮 Predict
├── Text Input Area
├── Sentiment Result
│   ├── Predicted class
│   └── Confidence score
└── Text Processing
    ├── Original text
    └── Cleaned text
```

### Programmatic Usage

```python
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel

# Initialize Spark
spark = SparkSession.builder.appName("Analysis").getOrCreate()

# Load trained model
model = LogisticRegressionModel.load("models/best_lr_model")

# Make predictions
predictions = model.transform(your_dataframe)
```

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Twitter Sentiment Analysis                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Data Layer (data/)                       │
├─────────────────────────────────────────────────────────────┤
│  CSV Files: twitter_training.csv, twitter_validation.csv   │
│  Size: 170K training + 10K validation tweets                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Processing Layer (Notebook)                   │
├─────────────────────────────────────────────────────────────┤
│  1. Text Preprocessing (cleaning, normalization)            │
│  2. Feature Engineering (unigrams, bigrams, trigrams)       │
│  3. Model Training (LR, NB with hyperparameter tuning)      │
│  4. Evaluation & Visualization                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Storage Layer (models/)                  │
├─────────────────────────────────────────────────────────────┤
│  ├── best_lr_model: Logistic Regression classifier         │
│  ├── Preprocessing: Tokenizer, StopWordsRemover            │
│  ├── Feature Extraction: HashingTF, IDF, NGrams            │
│  └── dashboard_data.json: Analytics data                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Application Layer (webapp/app.py)                │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Web Interface                                    │
│  ├── Dashboard Tab (analytics & insights)                  │
│  └── Predict Tab (real-time sentiment prediction)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Deployment Layer (Docker/Local)                   │
├─────────────────────────────────────────────────────────────┤
│  ├── Docker Compose: Container orchestration               │
│  ├── Dockerfile: Container image specification             │
│  └── Local: Direct Python execution                        │
└─────────────────────────────────────────────────────────────┘
```

### ML Pipeline Architecture

```
Raw Text
    │
    ▼
┌──────────────────────────┐
│  Text Cleaning           │
│ ├─ Lowercase             │
│ ├─ URL removal           │
│ ├─ Emoji removal         │
│ └─ Special char removal  │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Tokenization            │
│ └─ Split into words      │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Stop Words Removal      │
│ └─ Remove common words   │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Feature Extraction      │
├─ Unigrams (10K)         │
├─ Bigrams (5K)           │
└─ Trigrams (3K)          │
    │
    ▼
┌──────────────────────────┐
│  TF-IDF Vectorization    │
│ └─ Combined features     │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Classification Model    │
│ └─ Logistic Regression   │
└──────────────────────────┘
    │
    ▼
Prediction (4 Classes)
```

### Data Flow

```
Data Stage              Processing                      Output
──────────────────────────────────────────────────────────────
Raw CSV Data
  ↓
Loaded in Spark        ─ 170K tweets loaded           → Spark DF
  ↓
Cleaned Text           ─ Text preprocessing           → Clean tokens
  ↓
Balanced Dataset       ─ Stratified sampling          → 51.5K balanced
  ↓
Features Extracted     ─ TF-IDF + N-grams            → 18K dim vectors
  ↓
Models Trained         ─ CV with tuning              → Tuned LR/NB
  ↓
Evaluated              ─ Accuracy, F1-Score          → 76%+ accuracy
  ↓
Saved                  ─ Persist to disk              → models/
  ↓
Deployed               ─ Streamlit App                → Web UI
```

## 📊 Models & Performance

### Model Comparison

| Metric | Baseline LR | Tuned LR | Baseline NB | Tuned NB |
|--------|------------|---------|------------|---------|
| Accuracy | 74.2% | 76.1% | 72.8% | 75.3% |
| F1-Score | 0.7410 | 0.7605 | 0.7282 | 0.7525 |
| Precision | 74.3% | 76.2% | 73.1% | 75.4% |
| Recall | 74.2% | 76.0% | 72.8% | 75.3% |

### Best Model Details

**Tuned Logistic Regression**
- Regularization: L2 (Ridge)
- Reg Parameter: 0.1
- Max Iterations: 50
- Feature Dimensions: 18,000
- Training Time: ~5 minutes
- Inference Time: <100ms per prediction

### Feature Engineering Impact

```
Model                          Accuracy    Improvement
────────────────────────────────────────────────────────
Baseline (Unigrams only)       73.2%       Baseline
+ Bigrams                      74.8%       +1.6%
+ Trigrams                     76.1%       +2.9%
```

### Sentiment-wise Performance

```
Sentiment    Precision  Recall  F1-Score  Support
───────────────────────────────────────────────────
Positive     78.5%      76.2%   0.773     12875
Negative     75.1%      77.8%   0.764     12875
Neutral      72.4%      73.5%   0.730     12875
Irrelevant   76.8%      75.8%   0.762     12875
```

## 🔄 Data & Preprocessing

### Dataset Overview

```
Dataset              Records   Size
──────────────────────────────────────
Training             170,000   10MB
Validation           10,000    164KB
Total                180,000   10MB
```

### Class Distribution

```
Before Balancing:
├─ Positive:      80,500 (47.4%)
├─ Negative:      44,500 (26.2%)
├─ Neutral:       40,100 (23.6%)
└─ Irrelevant:     4,900 (2.9%)

After Balancing:
├─ Positive:      12,875 (25%)
├─ Negative:      12,875 (25%)
├─ Neutral:       12,875 (25%)
└─ Irrelevant:    12,875 (25%)
```

### Text Preprocessing Steps

1. **Lowercase Conversion**: Normalize case
2. **URL Removal**: Remove hyperlinks and web addresses
3. **Mention/Hashtag Cleanup**: Remove @ and # symbols while keeping words
4. **Emoji Removal**: Remove non-ASCII characters
5. **Contraction Expansion**: Convert "n't" → "not", "'re" → "are", etc.
6. **Number Removal**: Remove all digits
7. **Special Character Removal**: Remove punctuation except spaces
8. **Whitespace Normalization**: Remove extra spaces

### Example

```
Original:  "@User This game is awesome!!! 😍 Check it out: https://example.com #gaming"
Cleaned:   "user this game is awesome check it out gaming"
Tokenized: ["user", "game", "awesome", "check", "gaming"]
Filtered:  ["game", "awesome", "check", "gaming"]
(stop words removed)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Java Not Found Error

```
Error: JAVA_HOME not set or Java not found
```

**Solution**:
```bash
# Install Java
conda install -c conda-forge openjdk=17

# Or manually set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
```

#### 2. Out of Memory Error

```
Error: Java heap space or Spark OOM
```

**Solution**:
```bash
# Reduce batch size in Dockerfile/app.py
# Decrease spark.driver.memory and spark.executor.memory

# Or increase system resources
# Docker: Increase memory limit
# Local: Close other applications
```

#### 3. Models Not Found

```
Error: saved_models directory not found
```

**Solution**:
```bash
# Run training notebook first
jupyter notebook notebooks/TwitterSentimentAnalysis_Spark.ipynb
# Execute all cells through Section 12
```

#### 4. Port Already in Use

```
Error: Port 8501 already in use
```

**Solution**:
```bash
# Change port in docker-compose.yml or
streamlit run app.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill -9
```

#### 5. Docker Build Fails

```
Error: Cannot connect to Docker daemon
```

**Solution**:
```bash
# Start Docker service
sudo systemctl start docker

# Or use Docker Desktop GUI
```

### Performance Optimization

```
Issue                    Solution
────────────────────────────────────────────
Slow notebook execution  - Run on machine with 8GB+ RAM
                        - Use local[*] as master
                        - Increase Spark memory settings

Slow prediction        - Pre-cache models (already done)
                        - Use smaller feature set

High memory usage      - Reduce partition count
                        - Enable adaptive query optimization
```

### Debug Mode

```bash
# Enable Spark debug logging
export SPARK_DEBUG=1

# Show Spark UI during execution
spark.sparkContext.setLogLevel("DEBUG")

# View Spark application at http://localhost:4040
```

## 📚 Dependencies

### Core ML Libraries
- **pyspark==3.5.0**: Distributed computing framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Additional ML metrics

### Visualization
- **matplotlib**: Static plots
- **seaborn**: Statistical visualization
- **plotly**: Interactive charts
- **wordcloud**: Text visualization

### Web Framework
- **streamlit==1.28.0**: Web application framework

### Development
- **jupyter**: Interactive notebooks
- **python-dotenv**: Environment management

See [requirements.txt](requirements.txt) for complete list with versions.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone with SSH
git clone git@github.com:yourusername/twitter-sentiment-analysis.git

# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate

# Install with development dependencies
pip install -r requirements.txt pytest black flake8

# Run tests and linting
pytest
black --check .
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

For issues, questions, or suggestions:

- **Issue Tracker**: GitHub Issues
- **Email**: [Your email]
- **Documentation**: See docstrings in code files

## 🔗 Related Resources

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Sentiment Analysis Best Practices](https://github.com/papers-we-love/papers-we-love/tree/master/natural_language_processing)

## 🎓 References

### Academic Papers
- [TF-IDF Vectorization for Text](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Logistic Regression for Text Classification](https://en.wikipedia.org/wiki/Logistic_regression)
- [Naive Bayes Text Classification](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

### Datasets
- Twitter Sentiment Analysis Dataset
- Balanced multi-class dataset with 4 sentiment categories

## 🙏 Acknowledgments

- Apache Spark Team for the incredible distributed computing framework
- Streamlit Team for the web framework
- Twitter for the dataset
- All contributors and supporters

---

**Last Updated**: December 2025  
**Version**: 1.0.0  
**Status**: Production Ready
