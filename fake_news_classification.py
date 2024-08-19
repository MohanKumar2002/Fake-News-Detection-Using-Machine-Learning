# Import necessary libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Add this line to import NumPy
import itertools

# Load dataset
df = pd.read_csv('fake_or_real_news.csv')

# Display dataset information
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

# Initialize CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform CountVectorizer on training set
count_train = count_vectorizer.fit_transform(X_train)

# Transform test set using fitted CountVectorizer
count_test = count_vectorizer.transform(X_test)

# Initialize Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model on CountVectorized training data
model.fit(count_train, y_train)

# Predict on CountVectorized test data
pred = model.predict(count_test)

# Calculate accuracy score
score = metrics.accuracy_score(y_test, pred)
print("Accuracy:   %0.3f" % score)

# Compute confusion matrix
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])

# Define function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))  # Ensure 'np' is defined here
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

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

# Show plot
plt.show()
