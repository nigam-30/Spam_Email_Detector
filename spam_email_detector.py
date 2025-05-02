import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')  # Or use 'Qt5Agg' if you prefer


# Load the dataset
data = pd.read_csv('spam.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Show the first few rows
print("\nFirst few rows of the dataset:")
print(data.head(20))

# Data preprocessing
# Replace missing values (if any)
data.dropna(inplace=True)

# Convert the 'Label' column to numeric values (ham = 0, spam = 1)
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})

# Vectorize the 'EmailText' column using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['EmailText'])
y = data['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optional: Plotting confusion matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

