# Notebooks Directory

This directory contains Jupyter notebooks for model training, experimentation, and analysis.

## Files

### 📓 TwitterSentimentAnalysis_Spark.ipynb
**Main Training Notebook** - Production-ready notebook with complete ML pipeline

**Sections**:
1. Initialize Spark Session & Import Libraries
2. Load and Explore Data
3. Data Preparation
4. Exploratory Data Analysis
5. Text Preprocessing
6. Feature Engineering
6b. Enhanced N-gram Features
7. Hyperparameter Tuning
8. Original Models (Baseline)
9. Model Comparison
10. Text Mining & Visualization
11. Advanced Analysis
12. Model Saving

**Runtime**: ~15-30 minutes on 4GB RAM machine
**Output**: Trained models saved to `../models/`

### 📓 TwitterSentimentAnalysisModel.ipynb
**Alternative Model Notebook** - Different approach or previous version

### 📓 TwitterSentimentAnalysis_Spark.ipynb.bak
**Backup** - Backup of main notebook

## How to Use

### Interactive Development
```bash
jupyter notebook TwitterSentimentAnalysis_Spark.ipynb
```

### Execute All Cells
```bash
jupyter nbconvert --to notebook --execute TwitterSentimentAnalysis_Spark.ipynb
```

### Export to PDF
```bash
jupyter nbconvert --to pdf TwitterSentimentAnalysis_Spark.ipynb
```

## Notebook Structure

Each notebook follows this general structure:

```
1. Imports & Setup
   └─ Libraries, environment variables, configurations

2. Data Loading
   └─ Load CSV files into Spark DataFrames

3. Exploration
   └─ Statistics, distributions, visualizations

4. Preprocessing
   └─ Cleaning, normalization, feature engineering

5. Modeling
   └─ Training, hyperparameter tuning, evaluation

6. Analysis
   └─ Results, visualizations, insights

7. Persistence
   └─ Save models and artifacts
```

## Key Functions

### Text Cleaning
```python
def clean_text(text):
    # Lowercase, remove URLs, emojis, special chars, etc.
    # Returns clean text string
```

### Model Training
```python
# Logistic Regression with Cross-Validation
lr_cv_model = CrossValidator(...).fit(training_data)

# Naive Bayes with Cross-Validation
nb_cv_model = CrossValidator(...).fit(training_data)
```

### Evaluation
```python
# Accuracy
accuracy = evaluator.evaluate(predictions)

# F1-Score
f1_score = evaluator_f1.evaluate(predictions)

# Confusion Matrix
confusion_matrix(y_true, y_pred)
```

## Output Files

After running Section 12 (Model Saving), the following files are created:

```
models/
├── best_lr_model/              # Tuned Logistic Regression
├── label_indexer/              # Label encoding
├── tokenizer/                  # Text tokenization
├── stop_words_remover/         # Stop words removal
├── hashing_tf/                 # Unigram features
├── hashing_tf_bigram/          # Bigram features
├── hashing_tf_trigram/         # Trigram features
├── idf_model/                  # IDF transformation
├── bigram/                     # Bigram generation
├── trigram/                    # Trigram generation
├── vector_assembler/           # Feature assembly
└── dashboard_data.json         # Analytics data
```

## Troubleshooting

### Out of Memory
- Reduce `spark.driver.memory` in notebook config
- Use `coalesce()` to reduce partitions
- Cache only necessary DataFrames

### Slow Execution
- Increase `spark.executor.memory`
- Use `local[*]` to utilize all cores
- Check for shuffle operations

### Missing Dependencies
```bash
pip install -r ../requirements.txt
```

### Java Not Found
```bash
# Install Java
conda install -c conda-forge openjdk=17

# Set JAVA_HOME
export JAVA_HOME=/path/to/java
```

## Performance Tips

1. **Caching**: Use `.cache()` on DataFrames used multiple times
2. **Partitioning**: Repartition data for better parallelism
3. **Feature Selection**: Drop unnecessary columns early
4. **Sampling**: Use `.sample()` for quick testing
5. **Logging**: Adjust Spark log level to reduce overhead

## Next Steps

After training:
1. Models are automatically saved to `models/`
2. Run the web app from `webapp/` directory
3. Access predictions at `http://localhost:8501`

See [main README](../README.md) for complete documentation.
