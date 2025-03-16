# Language Detector

A machine learning model for detecting Azerbaijan, Russian, and English languages from text input.

## Overview

This repository contains an improved language classification system that can accurately identify between Azerbaijani, Russian, and English text. The system uses character n-grams and machine learning algorithms to provide fast and accurate language detection. The trained model achieves 98.6% accuracy on test data.

## Features

- Character-level n-gram analysis for robust language detection
- Support for both TF-IDF and Count vectorization
- Multiple classification algorithms (Naive Bayes and SVM)
- High accuracy across Azerbaijan, Russian, and English languages
- Fast classification (milliseconds per prediction)
- Probability estimation for confidence levels
- Tools for model evaluation and optimization

## Requirements

- Python 3.6+
- NumPy
- Pandas
- scikit-learn
- joblib
- Matplotlib
- Seaborn

## Installation

```bash
# Clone the repository
git clone https://github.com/vrashad/language_detect_bayes.git
cd language-detector

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a new model

```python
from language_classifier import ImprovedLanguageClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data['word'], data['language'], 
    test_size=0.2, random_state=42, stratify=data['language']
)

# Create and train the classifier
classifier = ImprovedLanguageClassifier(
    ngram_range=(1, 4),  # Character n-grams from 1 to 4 characters
    alpha=0.05,          # Smoothing parameter
    max_features=10000,  # Maximum number of features to consider
    use_tfidf=True,      # Use TF-IDF instead of count vectors
    model_type='naive_bayes'  # Use 'svm' for SVM classifier
)

classifier.fit(X_train, y_train)

# Evaluate the model
evaluation = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {evaluation['accuracy']:.4f}")
print(evaluation['report'])

# Save the model
classifier.save_model("models/my_language_classifier.joblib")
```

### Using a pre-trained model

```python
import joblib
import time

# Load the saved model
model_path = "models/improved_language_classifier.joblib"
model = joblib.load(model_path)

# Classify a word
word_to_classify = "acliq"
start_time = time.time()
predicted_language = model.predict([word_to_classify])[0]
probabilities = model.predict_proba([word_to_classify])[0]
confidence = max(probabilities) * 100
elapsed_ms = (time.time() - start_time) * 1000

print(f"Word: '{word_to_classify}'")
print(f"Language: {predicted_language}")
print(f"Confidence: {confidence:.2f}%")
print(f"Classification time: {elapsed_ms:.2f} ms")
```

## Model Details

The language classifier uses character-level n-grams to capture the patterns and features of different languages. This approach is particularly effective for language detection as it can identify subtle differences in character distributions and sequences.

### Key Components:

1. **Character-Level Analysis**: Words are broken down into character sequences of varying lengths (n-grams).
2. **Feature Extraction**: Either TF-IDF or Count Vectorization to convert text into numerical features.
3. **Classification Algorithm**: 
   - Naive Bayes (default): Fast and effective for text classification
   - SVM: High accuracy but potentially slower

### Hyperparameters:

- `ngram_range`: The range of n-gram sizes to consider (default: 1-4 characters)
- `alpha`: Smoothing parameter for Naive Bayes (default: 0.05)
- `max_features`: Maximum number of features to extract (default: 10,000)
- `use_tfidf`: Whether to use TF-IDF or simple count vectors (default: True)
- `model_type`: Classification algorithm ('naive_bayes' or 'svm')
- `class_prior`: Prior probabilities of classes (calculated automatically if not provided)

## Datasets

The repository includes two versions of the training dataset:

1. **cleaned_dataset.csv** - Contains only grammatically correct words in all languages.

2. **cleaned_dataset_with_translit_error.csv** - Contains both correct words and words with transliteration errors common in informal text (like social media posts).

The second dataset includes common Azerbaijani character substitutions such as:
```
'ş' → 's'
'ç' → 'c'
'ğ' → 'g'
'ö' → 'o'
'ü' → 'u'
'ı' → 'i'
'ə' → 'e'
```

Examples:
- üçün → ucun
- gəlmək → gelmek
- danışmaq → danismaq

This approach enables the model to correctly classify text even when it contains spelling errors or lacks proper diacritical marks, which is especially helpful for analyzing informal text from social media and other online sources.

Choose the appropriate dataset based on your specific use case.

## Performance

When trained on the included dataset, the model achieves:
- 98.6% accuracy on test data
- Fast classification times (typically < 10ms per word)

## Example Output

```
Word: 'acliq'
Language: azerbaijani
Confidence: 98.75%
Classification time: 5.23 ms
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
