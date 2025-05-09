from flask import Flask, render_template, request
import joblib

# Load model, vectorizer, and label encoder
model = joblib.load('icd_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_icd():
    if request.method == 'POST':
        input_text = request.form['input_text']
        X_input = vectorizer.transform([input_text])
        y_pred = model.predict(X_input)
        icd_code = label_encoder.inverse_transform(y_pred)[0]
        return render_template('index.html', prediction=icd_code, input_text=input_text)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
