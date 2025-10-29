from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dataset = load_dataset("dair-ai/emotion")

#split into train test
train_data = dataset["train"]
test_data = dataset["test"]

#tried printing stuff
print(train_data[0])

#tain test ts
train_texts = [x['text'] for x in train_data]
train_labels = [x['label'] for x in train_data]

test_texts = [x['text'] for x in test_data]
test_labels = [x['label'] for x in test_data]

#try printing stuff to see
print(train_texts[3])
print(train_labels[3])

#emotions
emotion_names = dataset["train"].features["label"].names
print(emotion_names)
# ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

#converting the txt into numbers type shit so machine can understand stuff
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

#testing and seeing stuff
print(X_test[1])


#training model type shit
svm_model = LinearSVC()
svm_model.fit(X_train, train_labels)

#prediction
svm_pred = svm_model.predict(X_test)
print("SVM Classification Report:")
print(classification_report(test_labels, svm_pred, target_names=emotion_names))


# Assuming emotion_names is your label list
cm = confusion_matrix(test_labels, svm_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_names)

plt.figure(figsize=(8, 6))
disp.plot(xticks_rotation='vertical')
plt.title("Confusion Matrix - SVM Emotion Classifier")
plt.show()
