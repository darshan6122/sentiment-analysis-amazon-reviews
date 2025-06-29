# Sentiment Analysis Project Documentation

## Project Overview
This project implements various machine learning and deep learning models for sentiment analysis on Amazon product reviews. The goal is to classify reviews as either positive or negative sentiment using different approaches and compare their performance.

## Dataset

The dataset used for training and evaluation is sourced from Kaggle:

- **Title:** [Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data)
- **Author:** Kritanjali Jain
- **License:** Provided by Kaggle for educational use

The dataset includes labeled reviews marked as **positive** or **negative**, making it suitable for binary sentiment classification tasks.

## Project Structure
```
Final_Submission/
├── models/                  # Trained model files
│   ├── cnn_lstm_model.h5   # CNN-LSTM hybrid model
│   ├── lstm_model.h5       # LSTM model
│   ├── rnn_model.h5        # RNN model
│   ├── mlp_model.h5        # Multi-layer Perceptron model
│   ├── naive_bayes.pkl     # Naive Bayes model
│   ├── gradient_boosting.pkl # Gradient Boosting model
│   ├── logistic_regression.pkl # Logistic Regression model
│   ├── tokenizer.pkl       # Text tokenizer
│   └── vectorizer.pkl      # TF-IDF vectorizer
├── notebooks/              # Jupyter notebooks for training and testing
│   ├── training_cnn_lstm.ipynb
│   ├── training_gradient_boosting.ipynb
│   ├── training_naive_bayes.ipynb
│   ├── testing_lstm.ipynb
│   └── testing_rnn.ipynb
└── results/               # Model evaluation results and visualizations
    ├── ResultsCNN_GB_NB_RNN.xlsx
    ├── model_all_metrics_comparison.png
    ├── model_accuracy_comparison.png
    └── Various confusion matrices
```

## Models Implemented

### 1. Traditional Machine Learning Models
- **Logistic Regression**: A linear model for binary classification
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
- **Gradient Boosting**: Ensemble learning method using decision trees

### 2. Deep Learning Models
- **LSTM (Long Short-Term Memory)**: Recurrent neural network for sequence modeling
- **CNN-LSTM Hybrid**: Combines convolutional and LSTM layers for better feature extraction
- **RNN (Recurrent Neural Network)**: Basic recurrent neural network implementation
- **MLP (Multi-layer Perceptron)**: Feed-forward neural network

## Data Preprocessing
The project implements comprehensive text preprocessing steps:
1. Text cleaning (removing URLs, special characters)
2. Lowercase conversion
3. Stop word removal
4. Lemmatization
5. Tokenization
6. Sequence padding for deep learning models

## Model Training and Evaluation
Each model is trained and evaluated using:
- 80-20 train-test split
- Binary cross-entropy loss for deep learning models
- Various evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrices

## Results and Visualizations
The project includes comprehensive result analysis:
- Excel spreadsheet with detailed metrics (ResultsCNN_GB_NB_RNN.xlsx)
- Visual comparison of model accuracies
- Confusion matrices for each model
- Overall metrics comparison visualization

## Technical Implementation
- **Framework**: TensorFlow/Keras for deep learning models
- **Libraries**: 
  - scikit-learn for traditional ML models
  - NLTK for text preprocessing
  - pandas for data manipulation
  - numpy for numerical operations
- **Model Persistence**: Models are saved in appropriate formats (.h5 for deep learning, .pkl for traditional ML)

## Usage
1. The notebooks in the `notebooks/` directory contain the complete implementation
2. Models can be trained using the respective training notebooks
3. Evaluation can be performed using the testing notebooks
4. Results are automatically saved and visualized

## Dependencies
- Python 3.x
- TensorFlow
- scikit-learn
- pandas
- numpy
- NLTK
- matplotlib (for visualizations)

## Future Improvements
1. Hyperparameter tuning for better model performance
2. Implementation of more advanced architectures (e.g., BERT, Transformer)
3. Cross-validation for more robust evaluation
4. Real-time prediction API
5. Model interpretability analysis 