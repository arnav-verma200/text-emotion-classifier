from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB



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


#traing model using Multinomial naive bayes for now 
model = MultinomialNB()
model.fit(X_train, train_labels)