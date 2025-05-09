'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.linear_model import LogisticRegression

# Load the cleaned data
df = pd.read_csv('icd_sample_data_clean.csv')

# Encode the ICD codes
le = LabelEncoder()
df['label'] = le.fit_transform(df['icd_code'])

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
#model = MultinomialNB()
model = LogisticRegression(max_iter=1000,solver='liblinear')
print("Training Logistic Regression model... please wait ⏳")
model.fit(X_train, y_train)
print("✅ Training complete!")
# Predict
y_pred = model.predict(X_test)

# Evaluation
print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(df['icd_code'].value_counts())

# Save the model and vectorizer
joblib.dump(model, 'icd_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the enhanced dataset
df = pd.read_csv('icd_sample_data_clean.csv')

# Encode the ICD codes
le = LabelEncoder()
df['label'] = le.fit_transform(df['icd_code'])

# TF-IDF vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # 1-grams and 2-grams
X = vectorizer.fit_transform(df['note_text'])
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save the model and vectorizer
joblib.dump(model, 'icd_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')
