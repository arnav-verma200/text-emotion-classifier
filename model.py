import joblib

#loading model and stuff
model = joblib.load("svm_emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

#emotion names
emotion_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

while True:
    text = input("\nEnter a sentence (or type 'exit' to quit): ")
    
    if text.lower() == "exit":
        print("Goodbye!")
        break
    
    #txt to numbers
    features = vectorizer.transform([text])
    
    #prediction of emotion no.
    pred_index = model.predict(features)[0]
    
    #convert index to label name using stuff
    predicted_emotion = emotion_names[pred_index]
    
    print(f"Predicted Emotion: 🟦 {predicted_emotion.upper()}")
