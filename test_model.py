import pickle

# Load the trained model and vectorizer
with open('phishing_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Test email (Phishing Example)
test_email = ["Your account has been compromised! Click here to reset your password: http://fake-bank.com"]

# Convert to feature vector
test_vector = vectorizer.transform(test_email)

# Predict
prediction = model.predict(test_vector)

print("Prediction:", prediction)  # Should print 'phishing' or 'legitimate'
