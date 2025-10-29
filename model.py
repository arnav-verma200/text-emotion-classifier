import joblib

# Load trained model & vectorizer
model = joblib.load("svm_emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Emotion label names (same ordering as training dataset)
emotion_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

while True:
    text = input("\nEnter a sentence (or type 'exit' to quit): ")
    
    if text.lower() == "exit":
        print("Goodbye!")
        break
    
    # Convert input text to numerical vector form
    features = vectorizer.transform([text])
    
    # Predict the emotion class index
    pred_index = model.predict(features)[0]
    
    # Convert index → label name
    predicted_emotion = emotion_names[pred_index]
    
    print(f"Predicted Emotion: 🟦 {predicted_emotion.upper()}")
