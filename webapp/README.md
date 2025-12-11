# Web Application Directory

This directory contains the production Streamlit web application for real-time sentiment prediction and analytics dashboard.

## Files

### 🐍 app.py
**Main Streamlit Application** - Web interface for sentiment analysis

**Features**:
- Interactive sentiment prediction interface
- Real-time analytics dashboard
- Model performance visualization
- Text preprocessing display

**Key Functions**:
- `initialize_spark()`: Initialize Spark session
- `load_models()`: Load all trained models
- `load_dashboard_data()`: Load pre-computed analytics
- `clean_text()`: Text preprocessing
- `predict_sentiment()`: Make predictions
- `render_dashboard()`: Render analytics tab
- `render_prediction()`: Render prediction tab

### 🐳 Dockerfile
**Container Image Definition** - Specifies Docker container setup

**Base Image**: `debian:bookworm-slim`

**Contents**:
- Python 3 installation
- Java 17 (required for Spark)
- PySpark and dependencies
- Environment configuration
- Streamlit application

**Exposed Port**: 8501

### 🔧 docker-compose.yml
**Docker Compose Configuration** - Orchestrates container setup

**Services**:
- `twitter-sentiment-app`: Main application container

**Resources**:
- Memory: 2GB limit, 1.5GB reservation
- CPU: No limit specified
- Port: 8501 (Streamlit)

**Volumes**:
- `../models/:/app/models:ro` - Read-only access to models

### 🚀 start.sh
**Launch Script** - Simplifies application startup

**Features**:
- Checks for model files
- Docker and Java verification
- Interactive menu for startup options
- Two modes: Docker Compose or Local Python

**Usage**:
```bash
bash start.sh
# Select option 1 (Docker) or 2 (Local)
```

### ⚙️ .streamlit/
**Streamlit Configuration Directory** - Application settings

## Application Architecture

```
┌──────────────────────────────────────────────┐
│         Streamlit Web Interface              │
├──────────────────────────────────────────────┤
│  Tab 1: Dashboard          Tab 2: Predict    │
│  ├─ Model metrics          ├─ Text input     │
│  ├─ Visualizations         ├─ Prediction     │
│  ├─ Confusion matrix       ├─ Confidence     │
│  └─ Token analysis         └─ Text preview   │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│      PySpark ML Pipeline                     │
├──────────────────────────────────────────────┤
│  ├─ Text Cleaning                            │
│  ├─ Tokenization                             │
│  ├─ Feature Extraction                       │
│  └─ Model Prediction                         │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│      Trained Models (../models/)             │
├──────────────────────────────────────────────┤
│  ├─ Logistic Regression (Main)              │
│  ├─ Preprocessing Pipeline                   │
│  └─ Feature Engineering Models               │
└──────────────────────────────────────────────┘
```

## Pages & Tabs

### 📊 Dashboard Tab

Displays analytics and model performance:

```
Dashboard
│
├── Model Performance Metrics
│   ├─ Baseline LR Accuracy
│   ├─ Tuned LR Accuracy
│   ├─ Baseline NB Accuracy
│   └─ Tuned NB Accuracy
│
├── Visualizations
│   ├─ Sentiment Distribution (Pie Chart)
│   ├─ Game Distribution (Bar Chart)
│   └─ Confusion Matrix (Heatmap)
│
├── Top Tokens by Sentiment
│   ├─ Positive words
│   ├─ Negative words
│   ├─ Neutral words
│   └─ Irrelevant words
│
└── Dataset Statistics
    ├─ Total Tweets
    ├─ Number of Games
    └─ Sentiment Classes
```

### 🔮 Prediction Tab

Real-time sentiment prediction interface:

```
Prediction
│
├── Input Section
│   ├─ Text Area (for input)
│   ├─ Predict Button
│   └─ Clear Button
│
├── Results
│   ├─ Sentiment (color-coded)
│   ├─ Confidence Percentage
│   ├─ Confidence Progress Bar
│   └─ Confidence Level Badge
│
└── Text Processing
    ├─ Original Text
    └─ Cleaned Text
```

## Running the Application

### Option 1: Docker Compose (Recommended)

```bash
# From webapp directory
cd webapp

# Run start script
bash start.sh
# Select option 1

# Or directly
docker-compose up --build

# Access at http://localhost:8501
```

### Option 2: Local Python

```bash
# From webapp directory
cd webapp

# Run start script
bash start.sh
# Select option 2

# Or directly
streamlit run app.py

# Access at http://localhost:8501
```

## Configuration

### Java Configuration
```bash
JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
_JAVA_OPTIONS=-Xmx1g -Xms256m -Duser.country=US -Duser.language=en
```

### Spark Configuration
```python
.config("spark.driver.memory", "1g")
.config("spark.executor.memory", "1g")
.config("spark.sql.shuffle.partitions", "4")
```

### Streamlit Configuration
Location: `.streamlit/config.toml`

```toml
[client]
showErrorDetails = true

[server]
port = 8501
maxUploadSize = 200
enableXsrfProtection = true
```

## Environment Variables

### Docker Environment
```bash
JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
HADOOP_USER_NAME=$USER
_JAVA_OPTIONS=-Xmx1g -Xms256m ...
JAVA_TOOL_OPTIONS=-Xmx1g -Xms256m
```

### Python Application
```python
os.environ['_JAVA_OPTIONS'] = '...'
os.environ['JAVA_TOOL_OPTIONS'] = '...'
```

## Dependencies

### Core
- streamlit==1.28.0
- pyspark==3.5.0
- pandas
- numpy

### Visualization
- plotly
- matplotlib
- seaborn

### Other
- scikit-learn (metrics)

See `../requirements.txt` for complete list.

## Performance Tuning

### Memory Usage
- Base: ~1GB
- Model loading: ~500MB
- Per prediction: <10MB

### Response Time
- Model loading: ~8-12s (cold start)
- Prediction: <100ms per sample
- Dashboard rendering: ~2-3s

### Optimization Tips
1. Pre-cache models (already done)
2. Reduce Spark shuffle partitions
3. Limit concurrent users
4. Monitor memory with `streamlit run ... --logger.level=debug`

## Troubleshooting

### Port Already in Use
```bash
# Change port
streamlit run app.py --server.port 8502

# Or kill process using 8501
lsof -ti:8501 | xargs kill -9
```

### Models Not Loading
```bash
# Verify models exist
ls ../models/best_lr_model

# Rebuild Docker image
docker-compose build --no-cache
```

### Out of Memory
```bash
# Reduce memory allocation in docker-compose.yml
# Or increase system resources

# Reduce Spark memory in app.py
.config("spark.driver.memory", "512m")
```

### Java Not Found
```bash
# In Docker:
# Already included in Dockerfile

# Locally:
conda install -c conda-forge openjdk=17
export JAVA_HOME=/path/to/java
```

## Development

### Running in Debug Mode
```bash
streamlit run app.py --logger.level=debug
```

### Adding New Features
1. Edit `app.py`
2. Add function for new feature
3. Add to UI rendering functions
4. Test locally before deploying

### Custom CSS
Modify the CSS in `app.py` under `st.markdown(<style>...)`

## Deployment

### Docker Deployment
```bash
# Build image
docker build -t twitter-sentiment:latest .

# Run container
docker run -p 8501:8501 -v ../models:/app/models:ro twitter-sentiment:latest

# With Docker Compose
docker-compose up -d
```

### Cloud Deployment
- AWS: ECS, Fargate
- Google Cloud: Cloud Run
- Azure: Container Instances
- Heroku: Via Docker

### Health Check
```bash
curl http://localhost:8501/_stcore/health
```

## Monitoring

### Logs
```bash
# Docker
docker logs twitter_sentiment_web

# Local
# Printed to console
```

### Metrics
- User access logs (Streamlit built-in)
- Model inference metrics
- Resource usage (CPU, memory)

## Security

### Current Setup
- No authentication (local use)
- Read-only model access
- Input validation (text cleaning)

### For Production
- Add authentication (SSO, OAuth)
- Rate limiting
- HTTPS/SSL
- Input sanitization
- CORS configuration

## Backups & Recovery

### Model Backup
```bash
# Backup models
cp -r ../models ../models.backup

# Restore
cp -r ../models.backup ../models
```

### Configuration Backup
```bash
# Backup Streamlit config
cp -r .streamlit .streamlit.backup
```

## Maintenance

### Regular Tasks
1. Monitor disk space for logs
2. Check model performance drift
3. Update dependencies quarterly
4. Review error logs weekly

### Scaling
- Load balancer for multiple instances
- Shared model storage (NFS, S3)
- Model versioning system

## Next Steps

1. Start application: `bash start.sh`
2. Access dashboard: `http://localhost:8501`
3. Test prediction functionality
4. Monitor performance
5. Plan for scaling/deployment

See [main README](../README.md) for complete documentation.
