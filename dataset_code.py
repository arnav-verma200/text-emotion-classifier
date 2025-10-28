from datasets import load_dataset

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

