# Import necessary libraries
import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('fake_or_real_news.csv')

# Display dataset shape and first few rows
print("Dataset shape:", df.shape)
print(df.head())

# Set 'Unnamed: 0' column as index
df = df.set_index('Unnamed: 0')

# Extract labels
y = df.label

# Drop 'label' column from features
df = df.drop('label', axis=1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform TF-IDF Vectorizer on training set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform test set using fitted TF-IDF Vectorizer
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print some feature names
print("Some feature names:", tfidf_vectorizer.get_feature_names_out()[-10:])

# Convert TF-IDF transformed data to DataFrame for visualization
tfidf_df = pd.DataFrame(tfidf_train.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Define function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Initialize Passive Aggressive Classifier
linear_clf = PassiveAggressiveClassifier(max_iter=50)

# Train the classifier on TF-IDF transformed training data
linear_clf.fit(tfidf_train, y_train)

# Predict on TF-IDF transformed test data
pred = linear_clf.predict(tfidf_test)

# Calculate accuracy score
score = metrics.accuracy_score(y_test, pred)
print("Accuracy:   %0.3f" % score)

# Compute confusion matrix
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'], title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'], normalize=True, title='Normalized confusion matrix')

# Show plots
plt.show()
