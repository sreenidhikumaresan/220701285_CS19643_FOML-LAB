import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not already done

nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
#df = pd.read_csv('icd_sample_data.csv')
df = pd.read_csv('icd_sample_data_full_enhanced.csv')
df.columns = df.columns.str.strip().str.lower()

# Initialize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = df['note_text'].apply(preprocess)

# Display sample results
print("Original vs Cleaned:")
print(df[['note_text', 'clean_text']].head())
df.to_csv('icd_sample_data_clean.csv', index=False)
