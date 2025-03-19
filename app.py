from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app) 

# Load the trained model and vectorizer
with open('phishing_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the prediction endpoint
@app.route('/predict_phishing', methods=['POST'])
def predict_phishing():
    email_content = request.json.get('email_content')

    # Preprocess and vectorize the email content
    email_vectorized = vectorizer.transform([email_content])

    # Make prediction
    prediction = model.predict(email_vectorized)
    label_map = {1: 'phishing', 0: 'legitimate'}  # Change based on your dataset labels
    predicted_label = label_map.get(prediction[0], "unknown")  # Handle unexpected values
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
