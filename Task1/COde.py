import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (with correct encoding)
df = pd.read_csv("spam.csv", encoding="latin-1")

# Drop the unnecessary columns
df = df[['v1', 'v2']]  # keep only the useful ones
df.columns = ['label', 'message']  # rename them

# Confirm missing values
print("Missing Values:\n", df.isnull().sum())
print("Dataset Shape:", df.shape)

# Class distribution
print("Class Distribution:\n", df['label'].value_counts())

# Visualize class distribution
sns.countplot(x='label', data=df)
plt.title("Spam vs Ham Distribution")
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Encode labels: ham → 0, spam → 1
le = LabelEncoder()
df['label_num'] = le.fit_transform(df['label'])

# Vectorize the message column using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['message'])

# Labels
y = df['label_num']

# Show shape of TF-IDF matrix and example
print("TF-IDF shape:", X.shape)
print("Sample features:", vectorizer.get_feature_names_out()[100:110])

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
