'''#batch testing
import joblib

# Load the saved model, vectorizer, and label encoder
model = joblib.load('icd_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# List of multiple test inputs
input_texts = [
    "Yellow skin and eyes with extreme tiredness",
    "Passing watery stools frequently with stomach cramps",
    "Feeling of nausea and vomiting after meals",
    "Persistent dizziness and mild headache during office work",
    "Shortness of breath after running short distance",
    "Sharp ear pain worsening while lying down",
    "Feeling heaviness and pain around eyes with blocked nose",
    "Severe headache getting worse with bright lights",
    "Swollen ankles and lower back pain with changes in urination",
    "Loss of smell and dry cough for several days"
]

# Vectorize the input sentences
X_input = vectorizer.transform(input_texts)

# Predict
predicted_labels = model.predict(X_input)

# Decode the predicted labels back to ICD codes
predicted_icd_codes = label_encoder.inverse_transform(predicted_labels)

# Show results
print("\n✅ Batch Prediction Results:\n")
for text, code in zip(input_texts, predicted_icd_codes):
    print(f"Input: {text}")
    print(f"Predicted ICD Code: {code}\n")

'''


#single line output cheching
import joblib

# Load saved objects
model = joblib.load('icd_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Sample diagnosis input
#input_text = "Complains of burning sensation during urination and lower stomach cramps"
#input_text = "Diagnosed with high blood pressure and dizziness"  #I10
#input_text = "Reports itchy skin with red rashes on arms"L29.9

#input_text = "Patient complains of difficulty in breathing and chest tightness"J45.909  smallpox=B03
#input_text = "The person is currently suffering from high cold and cough"
#input_text = "The patient experiences pain and burning sensation while urinating, with a feeling of urgency."
input_text = "A 50-year-old male reports severe pain in the upper right abdomen, with nausea after fatty meals."



# Preprocess it like before (manually for now)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
clean_input = preprocess(input_text)
input_vec = vectorizer.transform([clean_input])

# Predict
predicted_label = model.predict(input_vec)
predicted_icd = label_encoder.inverse_transform(predicted_label)

print(f"✅ Predicted ICD Code: {predicted_icd[0]}")
