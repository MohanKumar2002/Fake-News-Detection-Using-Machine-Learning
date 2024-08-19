# Fake News Detection Using Machine Learning

![Fake News Detection](fake_news_detection.png)

This project implements a machine learning approach to detect fake news articles using the Multinomial Naive Bayes classifier with CountVectorizer for text feature extraction.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Fake news has become a significant issue in the era of digital media, influencing public opinion and trust. This project aims to address this problem by building a classification model that can distinguish between fake and real news articles based on their textual content. The model is trained on a labeled dataset and evaluates its performance using metrics such as accuracy and a confusion matrix.

## Dataset

The dataset used for this project contains 6,335 news articles labeled as "FAKE" or "REAL". Each article includes a title and text content, which are utilized to train and test the machine learning model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MohanKumar2002/Fake-News-Detection-Using-Machine-Learning
   ```
   
2. Navigate into the project directory:
   ```bash
   cd fake-news-detection
   ```
   
3. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have Python installed (preferably Python 3.x).

2. Navigate to the project directory and activate your virtual environment (if used).

3. Run the main script:
   ```bash
   python fake_news_classification.py
   ```

4. The script will:
   - Load and preprocess the dataset.
   - Train a Multinomial Naive Bayes classifier using CountVectorizer.
   - Evaluate the model's accuracy and plot a confusion matrix.

## Results

- **Accuracy**: The model achieves an accuracy of 89.3% in distinguishing between fake and real news articles.
- **Confusion Matrix**: Visual representation of the model's performance with metrics such as precision, recall, and F1-score.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your improvements or suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
